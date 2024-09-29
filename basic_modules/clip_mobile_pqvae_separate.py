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

from get_args import get_args
from runner import train_epoch, validate_epoch
import torch.optim as optim
import toml
import pickle
from types import SimpleNamespace
from process_images import get_feature_loaders
# from process_images import CustomImageDataset
from process_images import FeatureDataset
import torch
from torch.utils.data import DataLoader, Dataset, Subset
import pdb
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from huffman import huffman_encoding, huffman_decoding
torch.autograd.set_detect_anomaly(True)

"""
# HANWEI's paper's method
"""


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
    def __init__(self, encoder_output_dimension):
        super(Encoder, self).__init__()
        self.initial_conv1 = nn.ConvTranspose2d(512, 384, kernel_size=3, stride=2, padding=0) # 1*1 -> 3*3
        self.initial_conv2 = nn.ConvTranspose2d(384, 256, kernel_size=3, stride=2, padding=1) # 3x3 -> 5*5
        self.residual_stack = ResidualStack(256, 256, 128, n_res_layers=3, inverse=True)
        self.pre_quant_conv = nn.Conv2d(256, encoder_output_dimension, kernel_size=3, stride=1, padding=1)  # 5*5 -> 5*5  # conv2d?
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 512, 1, 1).float()  # B*768 → B*768*1*1
        x = F.relu(self.initial_conv1(x))  # B*768*1*1 → B*512*3*3
        x = F.relu(self.initial_conv2(x))  # B*512*3*3 → B*256*5*5
        x = self.residual_stack(x)  # B*256*5*5 → B*256*5*5
        x = self.pre_quant_conv(x)
        return x
    

class Decoder(nn.Module):
    def __init__(self, encoder_output_dimension):
        super(Decoder, self).__init__()
        self.post_quant_conv = nn.Conv2d(encoder_output_dimension, 256, kernel_size=3, stride=1, padding=1)  # Bx128x5*5 -> Bx256x5*5
        self.residual_stack = ResidualStack(256, 256, 128, n_res_layers=3, inverse=False)
        self.final_conv1 = nn.Conv2d(256, 384, kernel_size=3, stride=2, padding=1)  # Bx256x5*5 -> Bx512x3x3
        self.final_conv2 = nn.Conv2d(384, 512, kernel_size=3, stride=2, padding=0)  # Bx512x3x3 -> Bx768x1x1

    def forward(self, x):
        x = self.post_quant_conv(x)  # Bx128x5*5 -> Bx256x5*5
        x = self.residual_stack(x)  # 通过残差块处理
        x = F.relu(self.final_conv1(x)) # Bx256x5*5 -> Bx512x3x3
        x = self.final_conv2(x) # Bx512x3x3 -> Bx768x1x1
        x = x.view(-1, 512)  # Bx768x1x1 -> Bx768
        return x

class VectorQuantizer(nn.Module):
    def __init__(self, n_e, num_bits, e_dim, beta, snr_db, device, encoder_output_dimension):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.num_bits = num_bits
        self.e_dim = e_dim
        self.beta = beta
        self.snr_db = snr_db
        self.device = device
        self.encoder_output_dimension = encoder_output_dimension

        self.num_segments = encoder_output_dimension // e_dim
        
        self.embeddings = nn.ModuleList([nn.Embedding(self.n_e, self.e_dim) for _ in range(self.num_segments)])
        for embedding in self.embeddings:
            embedding.weight.data.uniform_(-1.0, 1.0)


    def forward(self, z):
        z = z.permute(0, 2, 3, 1).contiguous() 

        z_segments = torch.split(z, self.e_dim, dim=-1)
        
        z_q_segments = []
        total_loss = 0

        for i, z_segment in enumerate(z_segments):
            z_flattened = z_segment.view(-1, self.e_dim)
            
            embedding = self.embeddings[i]
            dis = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
                  torch.sum(embedding.weight ** 2, dim=1) - 2 * \
                  torch.matmul(z_flattened, embedding.weight.t())

            min_encoding_indices = torch.argmin(dis, dim=1).unsqueeze(1)
            min_encodings = torch.zeros(min_encoding_indices.shape[0], self.n_e).to(self.device)
            min_encodings.scatter_(1, min_encoding_indices, 1)

            z_q = torch.matmul(min_encodings, embedding.weight).view(z_segment.shape)  
            
            codebook_loss = torch.mean((z_q - z_segment.detach()) ** 2)
            commitment_loss = torch.mean((z_q.detach() - z_segment) ** 2)
            loss = codebook_loss + self.beta * commitment_loss
            total_loss += loss
            
            z_q_segments.append(z_q)

            # pdb.set_trace()
            
            # entropy
            # min_encoding_indices = min_encoding_indices + i * self.n_e
            # if i == 0:
            #     min_encoding_indices_all = min_encoding_indices
            # else:
            #     min_encoding_indices_all = torch.cat((min_encoding_indices_all, min_encoding_indices), dim=1)

        z_q_combined = torch.cat(z_q_segments, dim=-1)
        z_q_combined = z_q_combined.permute(0, 3, 1, 2).contiguous()
        z = z.permute(0, 3, 1, 2).contiguous()

        z_q_combined = z + (z_q_combined - z).detach()

        # entropy
        # tmp = [int(e) for e in min_encoding_indices_all.view(-1)]
        # huff_coded_bits, huff_dict = huffman_encoding(tmp)
        # huffman_length = len(huff_coded_bits); print(huffman_length/z_q.shape[0])
        # pdb.set_trace()

        # return total_loss / self.num_segments, z_q_combined
        return total_loss, z_q_combined
    

class Clip_autoencoder(nn.Module):
    def __init__(self, cfg, args, device):
        super(Clip_autoencoder, self).__init__()
        print("dither4")
        self.device = device
        self.beta = args.beta
        self.power = cfg.CHANNEL.power
        self.args = args
        print('parameters ready')

        self.bandwidth = args.bw   # bandwidth/e_dim
        self.e_dim = self.bandwidth # bandwidth/e_dim  (1倍，也就是直接等于)
        self.n_e = args.n_e    # n_e
        self.num_bits = int(math.log2(self.n_e))

        self.SNR_dB = args.snr_db
        self.noise_std = 10 ** (-self.SNR_dB / 10)

        self.encoder_output_dimension = 64
        # 64的时候: 512-384-256(128)-64  (128)指的是residual
        # 32的时候: 512-384-256(128)-32  (32)指的是residual
        self.encoder = Encoder(self.encoder_output_dimension)
        self.decoder = Decoder(self.encoder_output_dimension)

        self.quantize = VectorQuantizer(self.n_e, self.num_bits, self.e_dim, self.beta, self.SNR_dB, self.device, self.encoder_output_dimension)

        self.tanh = nn.Tanh()


    def forward(self, x, indices):
        feature_normed = F.normalize(x, p=2, dim=1).float()
        encoded_x = self.encoder(feature_normed)

        encoded_x = F.normalize(encoded_x, p=2, dim=1).float()

        codebook_loss, z_q_reconstructed = self.quantize(encoded_x)

        decoded_x = self.decoder(z_q_reconstructed)
        decoded_x = F.normalize(decoded_x, p=2, dim=1)

        if self.training:
            similarity_matrix = torch.mm(feature_normed, decoded_x.t())
            channel_loss = torch.mean(1-torch.diag(similarity_matrix)) 
            total_loss = 10 * channel_loss + codebook_loss
            return decoded_x, total_loss
        else: 
            with torch.no_grad():
                return decoded_x


def get_logits(image_features, text_features, logit_scale):
    logits_per_image = (logit_scale * image_features @ text_features.T)
    logits_per_text = (logit_scale * text_features @ image_features.T)
    return logits_per_image, logits_per_text
