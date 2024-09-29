import torch
from torch.utils.data import Dataset, DataLoader
# from new_preprocess import CustomImageDataset
import time
from PIL import Image
import pdb
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import clip
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from get_args import get_args
from runner import train_epoch, validate_epoch
import torch.optim as optim
import toml
import pickle
from types import SimpleNamespace
# from process_images import get_feature_loaders
# from process_images import CustomImageDataset
# from process_images import FeatureDataset
import torch
from torch.utils.data import DataLoader, Dataset, Subset
import pdb
import os
import pickle
import sys
import argparse
import json
from typing import Tuple, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from enum import Enum
torch.autograd.set_detect_anomaly(True)

class MappingType(Enum):
    MLP = 'mlp'
    Transformer = 'transformer'


class InverseResidualLayer(nn.Module):
    def __init__(self, in_dim, h_dim, res_h_dim):
        super(InverseResidualLayer, self).__init__()
        self.res_block = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(in_dim, res_h_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.ConvTranspose2d(res_h_dim, h_dim, kernel_size=1, stride=1, bias=False),
        )

    def forward(self, x):
        return x + self.res_block(x)

class ResidualLayer(nn.Module):
    def __init__(self, in_dim, h_dim, res_h_dim):
        super(ResidualLayer, self).__init__()
        self.res_block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_dim, res_h_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(res_h_dim, h_dim, kernel_size=1, stride=1, bias=False),
        )

    def forward(self, x):
        return x + self.res_block(x)

class ResidualStack(nn.Module):
    def __init__(self, in_dim, h_dim, res_h_dim, n_res_layers, inverse=False):
        super(ResidualStack, self).__init__()
        self.n_res_layers = n_res_layers
        layer_class = InverseResidualLayer if inverse else ResidualLayer
        self.stack = nn.ModuleList(
            [layer_class(in_dim, h_dim, res_h_dim) for _ in range(n_res_layers)]
        )

    def forward(self, x):
        for layer in self.stack:
            x = layer(x)
            x = F.relu(x)
        return x


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.initial_conv1 = nn.ConvTranspose2d(768, 256, kernel_size=3, stride=2, padding=0) # 1*1 -> 3*3
        self.initial_conv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1) # 3x3 -> 5*5
        self.residual_stack = ResidualStack(128, 128, 64, n_res_layers=3, inverse=True)
        self.pre_quant_conv = nn.Conv2d(128, 12, kernel_size=3, stride=1, padding=1)  # 5*5 -> 5*5  # conv2d?
        self.tanh = nn.Tanh()
    def forward(self, x):
        x = x.view(-1, 768, 1, 1).float()  # B*768 → B*768*1*1
        x = F.relu(self.initial_conv1(x))  # B*768*1*1 → B*512*3*3
        x = F.relu(self.initial_conv2(x))  # B*512*3*3 → B*256*5*5
        x = self.residual_stack(x)  # B*256*5*5 → B*256*5*5
        x = self.tanh(self.pre_quant_conv(x))  # 76.53
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.post_quant_conv = nn.Conv2d(12, 128, kernel_size=3, stride=1, padding=1)  # Bx128x5*5 -> Bx256x5*5
        self.residual_stack = ResidualStack(128, 128, 64, n_res_layers=3, inverse=False)
        self.final_conv1 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)  # Bx256x5*5 -> Bx512x3x3
        self.final_conv2 = nn.Conv2d(256, 768, kernel_size=3, stride=2, padding=0)  # Bx512x3x3 -> Bx768x1x1
    def forward(self, x):
        x = self.post_quant_conv(x)  # Bx128x5*5 -> Bx256x5*5
        x = self.residual_stack(x)  # 通过残差块处理
        x = F.relu(self.final_conv1(x)) # Bx256x5*5 -> Bx512x3x3
        x = self.final_conv2(x) # Bx512x3x3 -> Bx768x1x1
        x = x.view(-1, 768)  # Bx768x1x1 -> Bx768
        return x
 
 
def get_indices(z_q):
    unique_vectors = {}
    current_index = 0
    batch_size, height, width, vector_length = z_q.shape
    indices = torch.empty(batch_size, height, width, dtype=torch.long, device=z_q.device)
    for b in range(batch_size):
        for h in range(height):
            for w in range(width):
                vector = tuple(z_q[b, h, w].tolist())
                if vector not in unique_vectors:
                    unique_vectors[vector] = current_index
                    current_index += 1
                indices[b, h, w] = unique_vectors[vector]
    return indices


class MLP(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)


class ClipCaptionModel(nn.Module):

    def get_dummy_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)

    def forward(self, tokens: torch.Tensor, prefix: torch.Tensor, mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None):
        embedding_text = self.gpt.transformer.wte(tokens)
        prefix_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.gpt_embedding_size)
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        return out

    def __init__(self, prefix_length: int, clip_length: Optional[int] = None, prefix_size: int = 768,
                 num_layers: int = 8, mapping_type: MappingType = MappingType.MLP):
        super(ClipCaptionModel, self).__init__()
        self.prefix_length = prefix_length # 10
        self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1] # 768
        if mapping_type == MappingType.MLP:
            self.clip_project = MLP((prefix_size, (self.gpt_embedding_size * prefix_length) // 2,
                                     self.gpt_embedding_size * prefix_length))  # (10, 10*768/2, 10*768)
        # else:
        #     self.clip_project = TransformerMapper(prefix_size, self.gpt_embedding_size, prefix_length,
        #                                                              clip_length, num_layers)
            

class Clip_autoencoder_cap(nn.Module):
    def get_dummy_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)

    def __init__(self, cfg, args, device, prefix_length: int, clip_length: Optional[int] = None, prefix_size: int = 768,
                 num_layers: int = 8, mapping_type: MappingType = MappingType.MLP):
        super(Clip_autoencoder_cap, self).__init__()
        self.device = device
        self.beta = args.beta
        self.power = cfg.CHANNEL.power
        self.args = args
        print('parameters ready')

        self.bandwidth = args.bw   # bandwidth/e_dim
        self.e_dim = self.bandwidth # bandwidth/e_dim  (1倍，也就是直接等于)
        self.n_e = args.n_e    # n_e
        self.num_bits = int(math.log2(self.n_e))

        # self.SNR_dB = args.snr_db
        # self.noise_std = 10 ** (-self.SNR_dB / 10)

        self.encoder = Encoder()
        self.decoder = Decoder()

        self.tanh = nn.Tanh()

        self.levels = 4

        self.prefix_length = prefix_length # 10
        self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1] # 768
        if mapping_type == MappingType.MLP:
            self.clip_project = MLP((prefix_size, (self.gpt_embedding_size * prefix_length) // 2,
                                     self.gpt_embedding_size * prefix_length))  # (10, 10*768/2, 10*768)
            # 整个clip_project都是仅针对clip features的
            

    def forward(self, tokens, x, mask: Optional[torch.Tensor] = None,  
                labels: Optional[torch.Tensor] = None):  # x: tensor
        
        feature_normed = F.normalize(x, p=2, dim=1).float()
        encoded_x = self.encoder(feature_normed)

        encoded_x = F.normalize(encoded_x, p=2, dim=1).float()

        # codebook_loss, z_q_reconstructed = self.quantize(encoded_x)

        min_val = -0.5; max_val = 0.5
        levels_tensor = torch.linspace(min_val, max_val, self.levels).to(self.device)
        quantized_tensor = levels_tensor[torch.argmin(torch.abs(encoded_x.unsqueeze(-1) - levels_tensor), dim=-1)]
        quantized_tensor = encoded_x + (quantized_tensor - encoded_x).detach()

        decoded_x = self.decoder(quantized_tensor)
        decoded_x = F.normalize(decoded_x, p=2, dim=1)

        prefix = decoded_x

        # pdb.set_trace()
        embedding_text = self.gpt.transformer.wte(tokens)  # self.gpt.transformer.wte.weight: [50257, 768]
        prefix_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.gpt_embedding_size)  
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        # embedding_text: [batch_size, 40, 768] 每句句子有40个tokens，每个token768维度, 这跟CLIP没关系, 是GPT2对于每个token要求是768维度
        # 具体来说, gpt2词汇库里面50257个words,每个word都会被映射为768维度
        # prefix_projections: [batch_size, 10, 768] （每句句子对应的）每个sample是1*768，通过映射到10*768
        # embdding_cat: [batch_size, 50, 768]
        if labels is not None: 
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        return out

