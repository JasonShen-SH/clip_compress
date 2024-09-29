import torch
from torch.utils.data import Dataset, DataLoader
from clip_finetune_jscc2 import get_logits
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
from runner import train_epoch, validate_epoch, train_epoch_random, validate_epoch_random
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
torch.autograd.set_detect_anomaly(True)

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
            nn.Conv2d(res_h_dim, h_dim, kernel_size=1, stride=1, bias=False)
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
        self.initial_conv1 = nn.ConvTranspose2d(512, 384, kernel_size=3, stride=2, padding=0) # 1*1 -> 3*3
        self.initial_conv2 = nn.ConvTranspose2d(384, 256, kernel_size=3, stride=2, padding=1) # 3x3 -> 5*5
        self.residual_stack = ResidualStack(256, 256, 128, n_res_layers=3, inverse=True)
        self.pre_quant_conv = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)  # 5*5 -> 5*5  # conv2d?

    def forward(self, x):
        x = x.view(-1, 512, 1, 1).float()  # Bx768 -> Bx768x1x1
        x = F.relu(self.initial_conv1(x))
        x = F.relu(self.initial_conv2(x))  
        x = self.residual_stack(x)
        # x = F.relu(self.pre_quant_conv(x))  # Bx256x5*5 -> Bx128x5*5
        # x = self.pre_quant_conv2(x)  # Bx256x5*5 -> Bx128x5*5
        x = self.pre_quant_conv(x)
        return x
    
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.post_quant_conv = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)  # Bx128x5*5 -> Bx256x5*5
        self.residual_stack = ResidualStack(256, 256, 128, n_res_layers=3, inverse=False)
        self.final_conv1 = nn.Conv2d(256, 384, kernel_size=3, stride=2, padding=1)  # Bx256x5*5 -> Bx512x3x3
        self.final_conv2 = nn.Conv2d(384, 512, kernel_size=3, stride=2, padding=0)  # Bx512x3x3 -> Bx768x1x1

    def forward(self, x):
        x = self.post_quant_conv(x)  # Bx128x5*5 -> Bx256x5*5
        x = self.residual_stack(x)  # 通过残差块处理
        x = F.relu(self.final_conv1(x))  # Bx256x5*5 -> Bx512x3x3
        x = self.final_conv2(x) # Bx512x3x3 -> Bx768x1x1
        x = x.view(-1, 512)  # Bx768x1x1 -> Bx768
        return x
    

class VectorQuantizer(nn.Module):
    def __init__(self, n_e, e_dim, beta, embedding, device):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.embedding = embedding
        self.device = device

    def forward(self, z):
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        # find closest encodings
        # pdb.set_trace()
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1) 
        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.n_e).to(self.device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        # compute loss for embedding (codebook + committment loss)
        # loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * \
        #     torch.mean((z_q - z.detach()) ** 2)
        loss = torch.mean((z_q - z.detach()) ** 2) + self.beta * torch.mean((z_q.detach() - z) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return loss, z_q,  min_encodings, min_encoding_indices



class indices2units(nn.Module):
    def __init__(self, num_bits, device):
        super(indices2units, self).__init__()
        self.num_bits = num_bits
        self.device = device
    def forward(self, min_encoding_indices): # indices→binarystring→tensor
        min_encoding_indices = min_encoding_indices.flatten().tolist()
        binary_string = ''.join([format(int(index), f'0{self.num_bits}b') for index in min_encoding_indices])
        binary_tensor = torch.tensor([float(bit) for bit in binary_string], dtype=torch.float32)
        bipolar_binary_tensor = 2 * binary_tensor - 1
        return bipolar_binary_tensor

class power_constraint(nn.Module):
    def __init__(self, n_channel, noise_std, power, device):
        super(power_constraint, self).__init__()
        self.n_channel = n_channel # self.bandwidth 见deepjscc-Q关于power constraint的定义
        self.noise_std = noise_std # unused for power constraint, used for AWGN 
        self.power = power # Is power normalized ?
        self.device = device 

    def forward(self, x1):
        # pdb.set_trace()
        norm = torch.norm(x1, p=2, dim=1, keepdim=True)  # B*C*H*W
        # normalized_x1 = x1 / (norm + 1e-5)
        normalized_x1 = x1 / (norm + 1e-8) # 此时每个sample能量为1(64/128维度)
        # scale = np.sqrt(self.power * self.n_channel) 
        scale = np.sqrt(self.power * 2 * self.n_channel) 
        output11 = scale * normalized_x1
        return output11

class awgn_channel(nn.Module):
    def __init__(self, ):
        super(awgn_channel, self).__init__()
    def forward(self, input_signal, snr_db):
        snr_linear = 10 ** (snr_db / 10.0)
        signal_power = 1.0
        noise_power = torch.tensor(signal_power / snr_linear)
        noise_std = torch.sqrt(noise_power)
        noise = torch.randn_like(input_signal) * noise_std
        output_signal = input_signal + noise
        # proportion = (torch.abs(noise) > 1).float().mean().item()
        return output_signal


class units2indices(nn.Module):
    def __init__(self, num_bits, device):
        super(units2indices, self).__init__()
        self.num_bits = num_bits
        self.device = device
    def forward(self, bipolar_binary_tensor):
        unipolar_binary_tensor = (bipolar_binary_tensor + 1) / 2
        binary_string = ''.join(unipolar_binary_tensor.int().cpu().numpy().astype(str))
        min_encoding_indices = [int(binary_string[i:i + self.num_bits], 2) for i in range(0, len(binary_string), self.num_bits)]  # 每num_bits位的二进制序列，转为十进制
        return min_encoding_indices
    

class Reverse_quantize(nn.Module):  # 不是找最近，而是直接映射
    def __init__(self, n_e, e_dim, embedding, device):
        super(Reverse_quantize, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.embedding = embedding
        self.device = device

    def forward(self, min_encoding_indices, shape):
        min_encoding_indices = torch.tensor(min_encoding_indices).reshape(-1,1).to(self.device)
        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.n_e).to(self.device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        z_q = torch.matmul(min_encodings, self.embedding.weight).view(shape)
        z_q = z_q + (z_q - z_q).detach()  # 保留梯度信息
        # z: 64*128*5*5
        return z_q


class VectorQuantizerWithChannel(nn.Module):
    # 发送：index(0-255) → (0&1)string → (0&1)tensor → (-1&1)tensor
    # 真正传输的是(-1&1)tensor,因为能量为1
    # 接受：先把噪声去掉(1.121→1),这步非常重要,绝不能直接(1.121+1)/2 → 去噪后的(-1&1)tensor → (0&1)tensor → (0&1)string → index(0-255)

    def __init__(self, n_e, num_bits, e_dim, beta, snr_db, device):
        super(VectorQuantizerWithChannel, self).__init__()
        self.n_e = n_e
        self.num_bits = num_bits
        self.e_dim = e_dim
        self.beta = beta
        self.snr_db = snr_db
        self.device = device

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def quantize(self, z):
        # pdb.set_trace()
        z = z.permute(0, 2, 3, 1).contiguous()

        z_flattened = z.view(-1, self.e_dim)

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.n_e).to(self.device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # pdb.set_trace()
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)  # 256个星座点，每个星座点64维度
        z_q = z_q.permute(0, 3, 1, 2).contiguous(); z = z.permute(0, 3, 1, 2).contiguous()

        # loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean((z_q - z.detach()) ** 2)
        loss = torch.mean((z_q - z.detach()) ** 2) + self.beta * torch.mean((z_q.detach() - z) ** 2)
        
        z_q_1 = z + (z_q - z).detach()
        # pdb.set_trace()
        return loss, z_q_1, min_encoding_indices

    def indices2units(self, min_encoding_indices):  
        # pdb.set_trace()
        min_encoding_indices = min_encoding_indices.flatten().tolist()
        binary_string = ''.join([format(int(index), f'0{self.num_bits}b') for index in min_encoding_indices])
        unipolar_binary_tensor = torch.tensor([float(bit) for bit in binary_string], dtype=torch.float32)
        bipolar_binary_tensor = 2 * unipolar_binary_tensor - 1
        return bipolar_binary_tensor, unipolar_binary_tensor

    def units2indices(self, bipolar_binary_tensor):
        unipolar_binary_tensor = (bipolar_binary_tensor + 1) / 2
        binary_string = ''.join(unipolar_binary_tensor.int().cpu().numpy().astype(str))
        min_encoding_indices = [int(binary_string[i:i + self.num_bits], 2) for i in range(0, len(binary_string), self.num_bits)]
        data = [int(char) for char in binary_string]
        r_unipolar_binary_tensor = torch.tensor(data, dtype=torch.float32)
        return min_encoding_indices, r_unipolar_binary_tensor

    def awgn_channel(self, input_signal):
        snr_linear = 10 ** (self.snr_db / 10.0)
        signal_power = 1.0
        noise_power = torch.tensor(signal_power / snr_linear)
        noise_std = torch.sqrt(noise_power)
        noise = torch.randn_like(input_signal) * noise_std
        output_signal = input_signal + noise
        return output_signal

    def forward(self, z):
        loss, z_q_1, min_encoding_indices = self.quantize(z)
        # z_q.requires_grad_(True)  
        
        with torch.no_grad():
            bipolar_binary_tensor, original_unipolar_binary_tensor = self.indices2units(min_encoding_indices)
            r_bipolar_binary_tensor = self.awgn_channel(bipolar_binary_tensor)
            r_bipolar_binary_tensor = torch.where(r_bipolar_binary_tensor >= 0, torch.tensor(1.0), torch.tensor(-1.0))
            r_min_encoding_indices, r_unipolar_binary_tensor = self.units2indices(r_bipolar_binary_tensor)
            r_min_encoding_indices = torch.tensor(r_min_encoding_indices).reshape(-1, 1).to(self.device)
            # correct_num = torch.sum(torch.eq(r_min_encoding_indices, min_encoding_indices)); print(correct_num)
            # pdb.set_trace()
            # 这里的correct_num是针对每个batch的，如何针对每个epoch，也就是使之遍历一遍整个数据集
            
            min_encodings = torch.zeros(r_min_encoding_indices.shape[0], self.n_e).to(self.device)
            min_encodings.scatter_(1, r_min_encoding_indices, 1)
            # pdb.set_trace()
            # z_q_reconstructed = torch.matmul(min_encodings, self.embedding.weight).view(z_q.shape)
            z_q_reconstructed = torch.matmul(min_encodings, self.embedding.weight).view(-1, 5, 5, 128)
            z_q_reconstructed = z_q_reconstructed.permute(0, 3, 1, 2).contiguous()

        # pdb.set_trace()
        z_q_reconstructed_1 = z_q_1 + (z_q_reconstructed - z_q_1).detach()
        # if torch.sum(torch.eq(z_q_reconstructed_1, z_q_reconstructed)) not in [204800, 320000]:
        #     print(torch.sum(torch.eq(z_q_reconstructed_1, z_q_reconstructed)))

        # z_q_reconstructed.requires_grad_(True)

        return loss, z_q_reconstructed_1
    

class Clip_autoencoder(nn.Module):
    def __init__(self, cfg, args, device):
        super(Clip_autoencoder, self).__init__()
        self.device = device
        self.beta = args.beta
        self.power = cfg.CHANNEL.power
        print('parameters ready')

        self.bandwidth = args.bw   # bandwidth/e_dim
        self.e_dim = self.bandwidth # bandwidth/e_dim  (1倍，也就是直接等于)
        self.n_e = args.n_e    # n_e
        self.num_bits = int(math.log2(self.n_e))

        self.SNR_dB = args.snr_db
        self.noise_std = 10 ** (-self.SNR_dB / 10)
        self.power_constraint = power_constraint(self.bandwidth, self.noise_std, self.power, self.device)

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        self.encoder = Encoder()  
        self.decoder = Decoder()
        # self.quantization = VectorQuantizer(self.n_e, self.e_dim, self.beta, self.embedding, device)
        # self.indices2units = indices2units(self.num_bits, device)
        # self.units2indices = units2indices(self.num_bits, device)
        # self.reverse_quantization = Reverse_quantize(self.n_e, self.e_dim, self.embedding, device)

        self.quantize = VectorQuantizer(self.n_e, self.e_dim, self.beta, self.embedding, self.device)
        self.big_machine = VectorQuantizerWithChannel(self.n_e, self.num_bits, self.e_dim, self.beta, self.SNR_dB, self.device)

        if cfg.CHANNEL.channel_type == 'awgn':
            self.channel = awgn_channel()


    def forward(self, text_features, img_features):
        text_features = text_features.squeeze()
        text_features = F.normalize(text_features, p=2, dim=1).float()
        img_features = img_features.squeeze()
        img_features = F.normalize(img_features, p=2, dim=1).float()

        tokens = text_features
        # x 应该是 [batch_size, 768], 而且是text features
        # feature_normed = F.normalize(tokens, p=2, dim=1).float()
        encoded_x = self.encoder(tokens)

        encoded_x = self.power_constraint(encoded_x)
        noisy_encoded_x = self.channel(encoded_x, self.SNR_dB)  # dither一般只对big_machien中的AWGN信道传输有作用

        codebook_loss, z_q_reconstructed = self.big_machine(noisy_encoded_x)
        # # codebook_loss, z_q_reconstructed,  min_encodings, min_encoding_indices = self.quantize(noisy_encoded_x)

        decoded_x = self.decoder(z_q_reconstructed)
        decoded_x = F.normalize(decoded_x, p=2, dim=1) 

        if self.training:
            # pdb.set_trace()
            similarity_matrix = torch.mm(decoded_x, text_features.t())
            channel_loss = torch.mean(1-torch.diag(similarity_matrix)) # mean within a single batch
            total_loss = 10 * channel_loss + codebook_loss  # 不应该1：10，1：1，也不应该20：1，最好10：1
            return decoded_x, total_loss
        else: # test
            with torch.no_grad():
                similarity_matrix = torch.mm(decoded_x, img_features.t())
                channel_loss = torch.mean(1-torch.diag(similarity_matrix))
                return channel_loss


