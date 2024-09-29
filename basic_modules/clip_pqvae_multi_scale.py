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
    # 发送：index(0-255) → (0&1)string → (0&1)tensor → (-1&1)tensor
    # 真正传输的是(-1&1)tensor,因为能量为1
    # 接受：先把噪声去掉(1.121→1),这步非常重要,绝不能直接(1.121+1)/2 → 去噪后的(-1&1)tensor → (0&1)tensor → (0&1)string → index(0-255)

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

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.n_e).to(self.device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)  # 256个星座点，每个星座点64维度
        z_q = z_q.permute(0, 3, 1, 2).contiguous(); z = z.permute(0, 3, 1, 2).contiguous()

        codebook_loss = torch.mean((z_q - z.detach()) ** 2)
        commitment_loss = torch.mean((z_q.detach() - z) ** 2)
        loss = codebook_loss + self.beta * commitment_loss
        
        z_q_1 = z + (z_q - z).detach()

        return loss, z_q_1


# class Discriminator(nn.Module):
#     def __init__(self, input_dim=768):
#         super(Discriminator, self).__init__()
#         # self.frontEncoder = nn.TransformerEncoderLayer(d_model=input_dim, nhead=8)
#         # self.encoder = nn.TransformerEncoder(self.frontEncoder, num_layers=3)
#         self.model = nn.Sequential(
#             nn.Linear(input_dim, 256),
#             nn.ReLU(),
#             nn.Linear(256, 1),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         output = self.model(x)
#         return output
    

class Clip_autoencoder(nn.Module):
    def __init__(self, cfg, args, device):
        super(Clip_autoencoder, self).__init__()
        print("multi-scale")
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
        # self.power_constraint = power_constraint(self.bandwidth, self.noise_std, self.power, self.device)

        # self.embedding = nn.Embedding(self.n_e, self.e_dim)
        # # self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        # self.embedding.weight.data.uniform_(-1.5, 1.5)

        self.encoder = Encoder()
        self.decoder = Decoder()
        # self.quantization = VectorQuantizer(self.n_e, self.e_dim, self.beta, self.embedding, device)
        # self.indices2units = indices2units(self.num_bits, device)
        # self.units2indices = units2indices(self.num_bits, device)
        # self.reverse_quantization = Reverse_quantize(self.n_e, self.e_dim, self.embedding, device)

        # self.big_machine = VectorQuantizerWithChannel(self.n_e, self.num_bits, self.e_dim, self.beta, self.SNR_dB, self.device)
        self.quantize = VectorQuantizer(self.n_e, self.num_bits, self.e_dim, self.beta, self.SNR_dB, self.device)

        # if cfg.CHANNEL.channel_type == 'awgn':
        #     self.channel = awgn_channel()
        
        # self.alpha = nn.Parameter(torch.tensor(1.0))  
        # self.beta = nn.Parameter(torch.tensor(1.0))

        # self.discriminator = Discriminator()
        self.tanh = nn.Tanh()


    def forward(self, x, indices):
        feature_normed = F.normalize(x, p=2, dim=1).float()
        encoded_x = self.encoder(feature_normed)
        # encoded_x = self.encoder(x)

        encoded_x = F.normalize(encoded_x, p=2, dim=1).float()

        # codebook_loss, z_q_reconstructed = self.big_machine(encoded_x)
        codebook_loss, z_q_reconstructed = self.quantize(encoded_x)

        decoded_x = self.decoder(z_q_reconstructed)
        decoded_x = F.normalize(decoded_x, p=2, dim=1)

        # real_outputs = self.discriminator(feature_normed)
        # fake_outputs = self.discriminator(decoded_x)

        # d_loss_real = F.binary_cross_entropy(real_outputs, torch.ones_like(real_outputs))
        # d_loss_fake = F.binary_cross_entropy(fake_outputs, torch.zeros_like(fake_outputs))
        # gan_loss = d_loss_real + d_loss_fake
        

        if self.training:
            # mse
            # total_loss = F.mse_loss(decoded_x, feature_normed)

            # cosine similarity
            similarity_matrix = torch.mm(feature_normed, decoded_x.t())
            channel_loss = torch.mean(1-torch.diag(similarity_matrix)) # mean within a single batch
            # total_loss = 10*channel_loss + codebook_loss + 0.1*gan_loss
            total_loss = 10 * channel_loss + codebook_loss

            # variational loss
            # 结构, 
            # bpp vs performance 
            # 10 change to alpha  （效果不好，会突然断崖式性能下降）
            # fixed dimension (close to upper bound)
            # (5*5*128) 尝试不同的bits (quantization bits, levels, 不同curve)
            # 5*5*64 20bits 
            # 5*5*128 10bits
            # 5*5*256 5bits

            # contrastive loss
            # unique_elements, counts = torch.unique(indices, return_counts=True)
            # repeated_indices = torch.where(counts > 1)[0]
            # repeated_elements = unique_elements[repeated_indices]
            # to_delete_positions = []
            # for element in repeated_elements:
            #     positions = (indices == element).nonzero(as_tuple=True)[0]
            #     to_delete_positions.extend(positions[1:].tolist())

            # for pos in to_delete_positions:
            #     indices = torch.cat((indices[:pos], indices[pos+1:]))
            #     decoded_x = torch.cat((decoded_x[:pos], decoded_x[pos+1:]), dim=0)
            #     feature_normed = torch.cat((feature_normed[:pos], feature_normed[pos+1:]), dim=0)

            # new_length = int(decoded_x.shape[0])
            # # pdb.set_trace()

            # logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).exp()
            # logits_per_image, logits_per_text = get_logits(feature_normed, decoded_x, logit_scale)
            # labels = torch.arange(logits_per_image.shape[0], device=self.device, dtype=torch.long)
            # contrastive_loss = (F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text, labels)) / 2

            # total_loss = contrastive_loss + codebook_loss
            # pdb.set_trace()

            return decoded_x, total_loss
        else: # test
            # 对于test, 只计算mse_loss
            with torch.no_grad():
                # val_mse_loss = F.mse_loss(decoded_x, feature_normed)
                # return decoded_x, val_mse_loss
                return decoded_x


def get_logits(image_features, text_features, logit_scale):
    # 计算image_features @ text_features.T相似度矩阵
    logits_per_image = (logit_scale * image_features @ text_features.T)
    logits_per_text = (logit_scale * text_features @ image_features.T)
    return logits_per_image, logits_per_text
