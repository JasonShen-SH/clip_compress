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
# from huffman import huffman_encoding, huffman_decoding
torch.autograd.set_detect_anomaly(True)

"""
self.embedding.weight.data.uniform_(-1.0, 1.0)的时候,能够达到最高76.29%
self.embedding.weight.data.uniform_(-1.5, 1.5)的时候,能够达到最高76.07% (但实际上是达到过76.62%的,但具体怎么达到的不太记得了)
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
    def __init__(self):
        super(Encoder, self).__init__()
        self.initial_conv1 = nn.ConvTranspose2d(768, 512, kernel_size=3, stride=2, padding=0) # 1*1 -> 3*3
        self.initial_conv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1) # 3x3 -> 5*5
        self.residual_stack = ResidualStack(256, 256, 128, n_res_layers=3, inverse=True)
        self.pre_quant_conv = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)  # 5*5 -> 5*5  # conv2d?
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 768, 1, 1).float()  # B*768 → B*768*1*1
        x = F.relu(self.initial_conv1(x))  # B*768*1*1 → B*512*3*3
        x = F.relu(self.initial_conv2(x))  # B*512*3*3 → B*256*5*5
        x = self.residual_stack(x)  # B*256*5*5 → B*256*5*5
        x = self.pre_quant_conv(x)
        return x
    

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.post_quant_conv = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)  # Bx128x5*5 -> Bx256x5*5
        self.residual_stack = ResidualStack(256, 256, 128, n_res_layers=3, inverse=False)
        self.final_conv1 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)  # Bx256x5*5 -> Bx512x3x3
        self.final_conv2 = nn.Conv2d(512, 768, kernel_size=3, stride=2, padding=0)  # Bx512x3x3 -> Bx768x1x1

    def forward(self, x):
        x = self.post_quant_conv(x)  # Bx128x5*5 -> Bx256x5*5
        x = self.residual_stack(x)  # 通过残差块处理
        x = F.relu(self.final_conv1(x)) # Bx256x5*5 -> Bx512x3x3
        x = self.final_conv2(x) # Bx512x3x3 -> Bx768x1x1
        x = x.view(-1, 768)  # Bx768x1x1 -> Bx768
        return x


class VectorQuantizer(nn.Module):

    def __init__(self, n_e, num_bits, e_dim, beta, snr_db, device):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.num_bits = num_bits
        self.e_dim = e_dim
        self.beta = beta
        self.snr_db = snr_db
        self.device = device

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        # self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        self.embedding.weight.data.uniform_(-1.2, 1.2)
        # torch.nn.init.xavier_uniform_(self.embedding.weight)

        # 0.9, 1.2, 1.5, 2.0

    def forward(self, z):
        z = z.permute(0, 2, 3, 1).contiguous()

        z_flattened = z.view(-1, self.e_dim)
        # pdb.set_trace()

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.n_e).to(self.device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)  
        z_q = z_q.permute(0, 3, 1, 2).contiguous(); z = z.permute(0, 3, 1, 2).contiguous()

        codebook_loss = torch.mean((z_q - z.detach()) ** 2)
        commitment_loss = torch.mean((z_q.detach() - z) ** 2)
        loss = codebook_loss + self.beta * commitment_loss
        
        z_q_1 = z + (z_q - z).detach()

        return loss, z_q_1

    
class Discriminator(nn.Module):
    def __init__(self, input_dim=768):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 384),
            nn.ReLU(),
            nn.Linear(384, 1),
            nn.Sigmoid()
        )
        # self.model = nn.Sequential(
        #     nn.Linear(input_dim, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 1),
        #     nn.Sigmoid()
        # )

    def forward(self, x):
        output = self.model(x)
        return output
    

class Clip_autoencoder(nn.Module):
    def __init__(self, cfg, args, device):
        super(Clip_autoencoder, self).__init__()
        print("gan")
        self.device = device
        self.beta = args.beta
        self.power = cfg.CHANNEL.power
        self.args = args

        self.bandwidth = args.bw 
        self.e_dim = self.bandwidth 
        self.n_e = args.n_e
        self.num_bits = int(math.log2(self.n_e))

        self.SNR_dB = args.snr_db
        self.noise_std = 10 ** (-self.SNR_dB / 10)

        self.encoder = Encoder()
        self.decoder = Decoder()

        self.quantize = VectorQuantizer(self.n_e, self.num_bits, self.e_dim, self.beta, self.SNR_dB, self.device)

        self.discriminator = Discriminator()
        self.tanh = nn.Tanh()


    def forward(self, x, indices):
        feature_normed = F.normalize(x, p=2, dim=1).float()
        encoded_x = self.encoder(feature_normed)

        encoded_x = F.normalize(encoded_x, p=2, dim=1).float()

        codebook_loss, z_q_reconstructed = self.quantize(encoded_x)

        decoded_x = self.decoder(z_q_reconstructed)
        decoded_x = F.normalize(decoded_x, p=2, dim=1)

        # feature_normed_part1 = feature_normed[:, :128]
        # feature_normed_part2 = feature_normed[:, 128:256]
        # feature_normed_part3 = feature_normed[:, 256:384]
        # feature_normed_part4 = feature_normed[:, 384:512]
        # feature_normed_part5 = feature_normed[:, 512:640]
        # feature_normed_part6 = feature_normed[:, 640:768]
        
        # decoded_x_part1 = decoded_x[:, :128]
        # decoded_x_part2 = decoded_x[:, 128:256]
        # decoded_x_part3 = decoded_x[:, 256:384]
        # decoded_x_part4 = decoded_x[:, 384:512]
        # decoded_x_part5 = decoded_x[:, 512:640]
        # decoded_x_part6 = decoded_x[:, 640:768]

        # real_outputs_part1 = self.discriminator(feature_normed_part1)
        # real_outputs_part2 = self.discriminator(feature_normed_part2)
        # real_outputs_part3 = self.discriminator(feature_normed_part3)
        # real_outputs_part4 = self.discriminator(feature_normed_part4)
        # real_outputs_part5 = self.discriminator(feature_normed_part5)
        # real_outputs_part6 = self.discriminator(feature_normed_part6)
        real_outputs = self.discriminator(feature_normed)

        # fake_outputs_part1 = self.discriminator(decoded_x_part1)
        # fake_outputs_part2 = self.discriminator(decoded_x_part2)
        # fake_outputs_part3 = self.discriminator(decoded_x_part3)
        # fake_outputs_part4 = self.discriminator(decoded_x_part4)
        # fake_outputs_part5 = self.discriminator(decoded_x_part5)
        # fake_outputs_part6 = self.discriminator(decoded_x_part6)
        fake_outputs = self.discriminator(decoded_x)

        # d_loss_real_part1 = F.binary_cross_entropy(real_outputs_part1, torch.ones_like(real_outputs_part1))
        # d_loss_real_part2 = F.binary_cross_entropy(real_outputs_part2, torch.ones_like(real_outputs_part2))
        # d_loss_real_part3 = F.binary_cross_entropy(real_outputs_part3, torch.ones_like(real_outputs_part3))
        # d_loss_real_part4 = F.binary_cross_entropy(real_outputs_part4, torch.ones_like(real_outputs_part4))
        # d_loss_real_part5 = F.binary_cross_entropy(real_outputs_part5, torch.ones_like(real_outputs_part5))
        # d_loss_real_part6 = F.binary_cross_entropy(real_outputs_part6, torch.ones_like(real_outputs_part6))
        d_loss_real = F.binary_cross_entropy(real_outputs, torch.ones_like(real_outputs))

        # d_loss_fake_part1 = F.binary_cross_entropy(fake_outputs_part1, torch.zeros_like(fake_outputs_part1))
        # d_loss_fake_part2 = F.binary_cross_entropy(fake_outputs_part2, torch.zeros_like(fake_outputs_part2))
        # d_loss_fake_part3 = F.binary_cross_entropy(fake_outputs_part3, torch.zeros_like(fake_outputs_part3))
        # d_loss_fake_part4 = F.binary_cross_entropy(fake_outputs_part4, torch.zeros_like(fake_outputs_part4))
        # d_loss_fake_part5 = F.binary_cross_entropy(fake_outputs_part5, torch.zeros_like(fake_outputs_part5))
        # d_loss_fake_part6 = F.binary_cross_entropy(fake_outputs_part6, torch.zeros_like(fake_outputs_part6))
        d_loss_fake = F.binary_cross_entropy(fake_outputs, torch.zeros_like(fake_outputs))

        # d_loss_real = (d_loss_real_part1 + d_loss_real_part2)/2 # + d_loss_real_part4 + d_loss_real_part5 + d_loss_real_part6)/6
        # d_loss_fake = (d_loss_fake_part1 + d_loss_fake_part2)/2 # + d_loss_fake_part4 + d_loss_fake_part5 + d_loss_fake_part6)/6
        
        gan_loss = d_loss_real + d_loss_fake

        if self.training:
            similarity_matrix = torch.mm(feature_normed, decoded_x.t())
            channel_loss = torch.mean(1-torch.diag(similarity_matrix)) # mean within a single batch
            total_loss = 10*channel_loss + codebook_loss + gan_loss
            return decoded_x, total_loss
        else: 
            with torch.no_grad():
                return decoded_x


def get_logits(image_features, text_features, logit_scale):
    logits_per_image = (logit_scale * image_features @ text_features.T)
    logits_per_text = (logit_scale * text_features @ image_features.T)
    return logits_per_image, logits_per_text
