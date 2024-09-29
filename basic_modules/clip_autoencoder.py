from uu import decode
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


# class Encoder(nn.Module):
#     def __init__(self):
#         super(Encoder, self).__init__()
#         self.initial_conv1 = nn.ConvTranspose2d(768, 512, kernel_size=3, stride=2, padding=0) # 1*1 -> 3*3
#         self.initial_conv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1) # 3x3 -> 5*5
#         self.residual_stack = ResidualStack(256, 256, 128, n_res_layers=3, inverse=True)
#         self.pre_quant_conv = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)  # 5*5 -> 5*5  # conv2d?
#         self.tanh = nn.Tanh()
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         x = x.view(-1, 768, 1, 1).float()  # B*768 → B*768*1*1
#         x = F.relu(self.initial_conv1(x))  # B*768*1*1 → B*512*3*3
#         x = F.relu(self.initial_conv2(x))  # B*512*3*3 → B*256*5*5
#         x = self.residual_stack(x)  # B*256*5*5 → B*256*5*5
#         x = self.pre_quant_conv(x)
#         return x

# class Decoder(nn.Module):
#     def __init__(self):
#         super(Decoder, self).__init__()
#         self.post_quant_conv = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)  # Bx128x5*5 -> Bx256x5*5
#         self.residual_stack = ResidualStack(256, 256, 128, n_res_layers=3, inverse=False)
#         self.final_conv1 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)  # Bx256x5*5 -> Bx512x3x3
#         self.final_conv2 = nn.Conv2d(512, 768, kernel_size=3, stride=2, padding=0)  # Bx512x3x3 -> Bx768x1x1

#     def forward(self, x):
#         x = self.post_quant_conv(x)  # Bx128x5*5 -> Bx256x5*5
#         x = self.residual_stack(x)  # 通过残差块处理
#         x = F.relu(self.final_conv1(x)) # Bx256x5*5 -> Bx512x3x3
#         x = self.final_conv2(x) # Bx512x3x3 -> Bx768x1x1
#         x = x.view(-1, 768)  # Bx768x1x1 -> Bx768
#         return x
    
class Encoder(nn.Module):
    def __init__(self, dimension):
        super(Encoder, self).__init__()
        self.initial_conv1 = nn.ConvTranspose2d(768, 256, kernel_size=3, stride=2, padding=0) # 1*1 -> 3*3
        self.initial_conv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1) # 3x3 -> 5*5
        self.residual_stack = ResidualStack(128, 128, 64, n_res_layers=3, inverse=True)
        self.pre_quant_conv = nn.Conv2d(128, dimension, kernel_size=3, stride=1, padding=1)  # 5*5 -> 5*5  # conv2d?
        self.tanh = nn.Tanh()
    def forward(self, x):
        x = x.view(-1, 768, 1, 1).float()  # B*768 → B*768*1*1
        x = F.relu(self.initial_conv1(x))  # B*768*1*1 → B*512*3*3
        x = F.relu(self.initial_conv2(x))  # B*512*3*3 → B*256*5*5
        x = self.residual_stack(x)  # B*256*5*5 → B*256*5*5
        x = self.tanh(self.pre_quant_conv(x))  # 76.53
        return x

class Decoder(nn.Module):
    def __init__(self, dimension):
        super(Decoder, self).__init__()
        self.post_quant_conv = nn.Conv2d(dimension, 128, kernel_size=3, stride=1, padding=1)  # Bx128x5*5 -> Bx256x5*5
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

# class ResidualQuantizer(nn.Module):
#     def __init__(self, num_quantizers, n_e, e_dim, device):
#         super(ResidualQuantizer, self).__init__()
#         self.num_quantizers = num_quantizers
#         self.quantizers = nn.ModuleList(
#             [nn.Embedding(n_e, e_dim) for _ in range(num_quantizers)]
#         )
#         for quantizer in self.quantizers:
#             quantizer.weight.data.uniform_(-1.0, 1.0)
#         self.device = device
#         self.n_e = n_e
#         self.e_dim = e_dim

#     def forward(self, z):
#         residual = z
#         quantized_outputs = []
#         for quantizer in self.quantizers:
#             z_flattened = residual.view(-1, self.e_dim)
#             d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
#                 torch.sum(quantizer.weight ** 2, dim=1) - 2 * \
#                 torch.matmul(z_flattened, quantizer.weight.t())

#             min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
#             min_encodings = torch.zeros(min_encoding_indices.shape[0], quantizer.num_embeddings).to(self.device)
#             min_encodings.scatter_(1, min_encoding_indices, 1)

#             z_q = torch.matmul(min_encodings, quantizer.weight).view(residual.shape)
#             quantized_outputs.append(z_q)
#             residual = residual - z_q
#             # pdb.set_trace()
#         return residual
    

# class VectorQuantizer(nn.Module):
#     def __init__(self, n_e, num_bits, e_dim, beta, device, num_quantizers):
#         super(VectorQuantizer, self).__init__()
#         self.n_e = n_e
#         self.num_bits = num_bits
#         self.e_dim = e_dim
#         self.beta = beta
#         # self.snr_db = snr_db
#         self.device = device
#         self.num_quantizers = num_quantizers
#         self.tanh = nn.Tanh()
#         self.levels = 16

#         self.residual_quantizer = ResidualQuantizer(self.num_quantizers, n_e, e_dim, device)

#     def forward(self, z):
#         z = z.permute(0, 2, 3, 1).contiguous()
#         output = self.residual_quantizer(z)

#         # output = self.tanh(output) # 不用tanh
#         min_val = -0.2
#         max_val = 0.2
#         quantized_tensor = torch.round((output - min_val) / (max_val - min_val) * (self.levels-1)) / (self.levels-1) * (max_val - min_val) + min_val
#         quantized_tensor = output + (quantized_tensor - output).detach()

#         z_q = quantized_tensor
#         z_q = z_q.permute(0, 3, 1, 2).contiguous()
#         z = z.permute(0, 3, 1, 2).contiguous()

#         codebook_loss = torch.mean((z_q - z.detach()) ** 2)
#         commitment_loss = torch.mean((z_q.detach() - z) ** 2)
#         loss = codebook_loss + self.beta * commitment_loss

#         # huffman 1
#         # quantized_list_flatten = [int(e) for e in quantized_tensor.flatten()]
#         # huff_coded_bits, huff_dict = huffman_encoding(quantized_list_flatten)
#         # huffman_length = len(huff_coded_bits); print(huffman_length)

#         # huffman 2
#         # z_q_reshape = z_q.view(z_q.shape[0], z_q.shape[2], z_q.shape[3], -1)
#         # indices = get_indices(z_q_reshape)
#         # quantized_list_flatten = [int(e) for e in indices.flatten()]
#         # huff_coded_bits, huff_dict = huffman_encoding(quantized_list_flatten)
#         # huffman_length = len(huff_coded_bits); print(huffman_length)

#         return loss, z_q
    
    
def round_huffman_dict(huff_dict, decimal_places=4):
    rounded_huff_dict = {round(key, decimal_places): value for key, value in huff_dict.items()}
    return rounded_huff_dict


class Clip_autoencoder(nn.Module):
    def __init__(self, cfg, args, device, dimension, levels):
        super(Clip_autoencoder, self).__init__()
        # print("rqvae")
        self.device = device
        self.beta = args.beta
        self.power = cfg.CHANNEL.power
        self.args = args
        # print('parameters ready')

        self.bandwidth = args.bw   # bandwidth/e_dim
        self.e_dim = self.bandwidth # bandwidth/e_dim  (1倍，也就是直接等于)
        self.n_e = args.n_e    # n_e
        self.num_bits = int(math.log2(self.n_e))

        # self.SNR_dB = args.snr_db
        # self.noise_std = 10 ** (-self.SNR_dB / 10)

        self.dimension = dimension
        self.levels = levels

        self.encoder = Encoder(dimension)
        self.decoder = Decoder(dimension)

        # self.num_quantizers = 1
        # self.quantize = VectorQuantizer(self.n_e, self.num_bits, self.e_dim, self.beta, self.device, self.num_quantizers)
        self.tanh = nn.Tanh()

        # self.levels = 4

    def forward(self, x, indices):
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

        # # # dict
        # huff_dict = torch.load("huff_dicts/encoder-output-dimension-16-levels-8.pt")
        # round_huff_dict = round_huffman_dict(huff_dict)
        # # data
        # flattened_tensor = quantized_tensor.flatten()
        # flattened_list = flattened_tensor.cpu().numpy().tolist()  # 将其转换为列表
        # rounded_flattened_list = [round(val, 4) for val in flattened_list]  # 保留四位小数
        # encoded_data = ''.join(round_huff_dict[symbol] for symbol in rounded_flattened_list)
        # avg_length = len(encoded_data)/decoded_x.shape[0]; print(avg_length/768)
        # pdb.set_trace()

        # quantized_list_flatten = [e.item() for e in quantized_tensor.flatten()]
        # huff_coded_bits, huff_dict = huffman_encoding(quantized_list_flatten)
        # huffman_length = len(huff_coded_bits); print(huffman_length/decoded_x.shape[0])
        # pdb.set_trace()

        # huffman 2
        # z_q_reshape = quantized_tensor.view(quantized_tensor.shape[0], quantized_tensor.shape[2], quantized_tensor.shape[3], -1)
        # # bits_per_vector = int(math.ceil(math.log2(self.levels**z_q_reshape.shape[-1])))
        # indices = get_indices(z_q_reshape)
        # quantized_list_flatten = [int(e) for e in indices.flatten()]
        # huff_coded_bits, huff_dict = huffman_encoding(quantized_list_flatten)
        # huffman_length = len(huff_coded_bits); print(huffman_length/z_q_reshape.shape[0])

        # quantized_tensor = torch.round(math.floor(self.levels/2) * self.tanh(encoded_x)) # FSQ formula
        # quantized_tensor = encoded_x + (quantized_tensor - encoded_x).detach()
        # decoded_x = self.decoder(quantized_tensor)

        # quantized_list_flatten = [e.item() for e in quantized_tensor.flatten()]
        # huff_coded_bits, huff_dict = huffman_encoding(quantized_list_flatten)
        # huffman_length = len(huff_coded_bits); print(huffman_length)
        # pdb.set_trace()

        if self.training:
            # cosine similarity
            similarity_matrix = torch.mm(feature_normed, decoded_x.t())
            channel_loss = torch.mean(1-torch.diag(similarity_matrix))
            # total_loss = 10 * channel_loss + codebook_loss
            total_loss = channel_loss

            return decoded_x, total_loss
        else:
            with torch.no_grad():
                return decoded_x


def get_logits(image_features, text_features, logit_scale):
    logits_per_image = (logit_scale * image_features @ text_features.T)
    logits_per_text = (logit_scale * text_features @ image_features.T)
    return logits_per_image, logits_per_text
