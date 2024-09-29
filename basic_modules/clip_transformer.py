from __future__ import absolute_import
from __future__ import division
import torch
from torch import nn
from torch.nn import functional as F
import torchvision
import numpy as np
import math
import sys
import clip
import toml
import pdb

__all__ = ['Clip4']

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

        loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean((z_q - z.detach()) ** 2)
        
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
        signal_power = 0.5 # 1.0
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
    def scalar_quantize(self, tensor, args):
        levels = 2 ** args.levels
        # flat_tensor = tensor.view(-1)
        # min_val = torch.min(flat_tensor)
        # max_val = torch.max(flat_tensor)
        min_val = -1
        max_val = 1
        range_val = max_val - min_val
        step = range_val / levels
        quantized_tensor = torch.round((tensor - min_val) / step) * step + min_val
        # pdb.set_trace()
        quantized_tensor = tensor + (quantized_tensor - tensor).detach()
        return quantized_tensor
    
    def learnable_scalar_quantize(self, tensor, args):
        levels = 2 ** args.levels
        init_min = -1.0
        init_max = 1.0
        min_val = nn.Parameter(torch.tensor([init_min])).to(self.device)  
        max_val = nn.Parameter(torch.tensor([init_max])).to(self.device)  
        range_val = max_val - min_val 
        # step = nn.Parameter(torch.tensor([(init_max - init_min) / (2 ** levels)]))  # 可学习的步长
        step = range_val / levels
        quantized_tensor = torch.round((tensor - min_val) / step) * step + min_val
        quantized_tensor = tensor + (quantized_tensor - tensor).detach()
        return quantized_tensor
    
    def learnable_scalar_quantize_2(self, tensor, args):
        levels = 2 ** args.levels
        init_min = -1.0
        init_max = 1.0

        boundaries = nn.Parameter(torch.linspace(init_min, init_max, levels + 1).to(self.device))
        boundaries = torch.sort(boundaries)[0] # must sort all boundaries before each quantization process
        # pdb.set_trace()

        quantized_tensor = torch.zeros_like(tensor)
        for i in range(levels):
            min_val = boundaries[i]
            max_val = boundaries[i + 1]
            mask = (tensor >= min_val) & (tensor < max_val)
            quantized_tensor[mask] = ((min_val + max_val) / 2.0)

        mask = (tensor >= boundaries[-2]) & (tensor <= boundaries[-1])
        quantized_tensor[mask] = ((boundaries[-2] + boundaries[-1]) / 2.0)

        quantized_tensor = tensor + (quantized_tensor - tensor).detach()
        return quantized_tensor

    def dither_quantize(self, tensor, args):
        # dither_quantize的前提是，先dither，然后再去quantize，两步一个都不可少；
        # dither当然是精髓
        # quantize是前提，是目的，否则直接传的话，根本就没有量化噪声，还需要dither干嘛
        # dither是quantization之前的步骤，因此做dither的时候，必须保证quantization是uniform的
        tensor = tensor.to(self.device)
        init_min_value = -1.0 
        init_max_value = 1.0 
        min_value = nn.Parameter(torch.tensor([init_min_value])).to(self.device)  
        max_value = nn.Parameter(torch.tensor([init_max_value])).to(self.device)  

        levels = 2 ** args.levels
        interval_size = (max_value - min_value) / levels
        dither_range = (interval_size / 2).to("cpu")
        dither_noise = (torch.rand(tensor.size()) * 2 - 1) * dither_range
        dither_noise = dither_noise.to(self.device)
        dithered_vector = tensor + dither_noise
        
        quantized_vector = torch.round((dithered_vector - min_value) / interval_size) * interval_size + min_value
        quantized_vector = tensor + (quantized_vector - tensor).detach()
        # quantized_vector = dithered_vector + (quantized_vector - dithered_vector).detach()
        return quantized_vector


    def __init__(self, cfg, args, device):
        super(Clip_autoencoder, self).__init__()
        self.args = args
        self.device = device
        self.bandwidth = args.bw_old
        self.power = cfg.CHANNEL.power
        print('parameters ready')

        # self.clip, self.preprocess = clip.load('/cpfs01/user/chengxuxin.cxx/clip-jscc-ali/ViT-L-14-336px.pt',device=self.device) # 学长这里是直接调用已经fine_tine过了的
        # self.clip, _ = clip.load("ViT-L/14@336px", device=self.device)  # 我们已经preprocess至适合于clip的形式了，因此不再使用self.preprocess
        # print('CLIP model ready')

        self.frontEncoder = nn.TransformerEncoderLayer(d_model=768, nhead=8)  
        self.encoder = nn.TransformerEncoder(self.frontEncoder, num_layers=3)
        self.encoder1 = nn.TransformerEncoder(self.frontEncoder, num_layers=3)

        self.denseEncoder1 = torch.nn.Sequential(nn.Linear(768, 256),nn.ReLU())
        self.denseEncoder2 = torch.nn.Sequential(nn.Linear(256, 2*self.bandwidth),nn.ReLU())

        self.denseDecoder1 = torch.nn.Sequential(nn.Linear(2*self.bandwidth, 256),nn.ReLU())
        self.denseDecoder2 = torch.nn.Sequential(nn.Linear(256, 768),nn.ReLU())

        self.frontDecoder = nn.TransformerDecoderLayer(d_model=768, nhead=8)
        self.decoder = nn.TransformerDecoder(self.frontDecoder, num_layers=3)
    
        self.tanh = nn.Tanh()

        self.embedding_dim = self.bandwidth 

        # self.constellation_points = cluster_centers
        # # self.register_buffer('constellation_points', cluster_centers)
        # self.constellation_points = torch.randn(32768, 384)

        self.SNR_dB = args.snr_db
        self.noise_std = 10 ** (-self.SNR_dB / 10)
        self.power_constraint = power_constraint(self.bandwidth, self.noise_std, self.power, self.device)

        if cfg.CHANNEL.channel_type == 'awgn':
            self.channel = awgn_channel()

    def forward(self, image_features, indices):
        image_features_normed = F.normalize(image_features, p=2, dim=1).float()
        code = self.encoder(image_features_normed.unsqueeze(0))
        denseCode = self.denseEncoder1(code)
        channel_encoder_output = self.denseEncoder2(denseCode)
        channel_encoder_output = self.tanh(channel_encoder_output)
        channel_encoder_output = torch.squeeze(channel_encoder_output)

        # 定义quantize
        # quantized_output = self.scalar_quantize(channel_encoder_output, self.args)
        # quantized_output = self.learnable_scalar_quantize_2(channel_encoder_output, self.args)
        quantized_output = self.dither_quantize(channel_encoder_output, self.args)

        # channel_input = self.power_constraint(quantized_output) # batch-size * 384
        channel_input = self.power_constraint(quantized_output)

        channel_output = self.channel(channel_input, self.SNR_dB)

        codeReceived = self.denseDecoder1(channel_output)
        codeReceived = self.denseDecoder2(codeReceived)

        feat_decoded = self.encoder1(codeReceived.unsqueeze(0))
        feat_decoded = torch.squeeze(feat_decoded)

        feat_decoded = F.normalize(feat_decoded, p=2, dim=1)

        if self.training:
            # logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).exp()
            # logits_per_image, logits_per_text = get_logits(feat_decoded, image_features_normed, logit_scale)
            # labels = torch.arange(logits_per_image.shape[0], device=self.device, dtype=torch.long) # 0到63
            # channel_loss = (F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text, labels)) / 2

            similarity_matrix = torch.mm(image_features_normed, feat_decoded.t())
            channel_loss = torch.mean(1-torch.diag(similarity_matrix)) # mean within a single batch
            total_loss = channel_loss
            return feat_decoded, total_loss
        else: # test
            with torch.no_grad():
                return feat_decoded
    
    # def quantize(self, inputs):
    #     input_shape = inputs.shape # batch-size * 384
        
    #     self.constellation_points = self.constellation_points.float() # 16 * 384
    #     distances = torch.cdist(inputs.unsqueeze(0), self.constellation_points.float().unsqueeze(0).to(self.device)).squeeze(0) + 1e-8
    #     encoding_indices = torch.argmin(distances, dim=-1) # 0-15

    #     topk_distances, topk_indices = torch.topk(distances, k=2, dim=-1, largest=False, sorted=True)

    #     quantized = self.constellation_points[encoding_indices].view(input_shape) # batch-size * 384

    #     quantized = inputs + (quantized.to(self.device) - inputs).detach()

    #      # Calculate entropy regularization term
    #     unique, counts = torch.unique(encoding_indices, return_counts=True)
    #     probabilities = counts.float() / encoding_indices.size(0)
    #     entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-8))  # Calculate entropy
    #     entropy_reg = -entropy  # Minimize the negative entropy to encourage uniform usage

    #     return quantized, encoding_indices, entropy_reg


    
def get_logits(image_features, text_features, logit_scale):
    logits_per_image = (logit_scale * image_features @ text_features.T) # batch_size * batch_size
    logits_per_text = (logit_scale * text_features @ image_features.T)  
    # two matrixes are transpose
    return logits_per_image, logits_per_text            


class power_constraint(nn.Module):
    def __init__(self, n_channel, noise_std, power, device):
        super(power_constraint, self).__init__()
        # self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.n_channel = n_channel # self.bandwidth 见deepjscc-Q关于power constraint的定义
        # self.subcarriers = subcarriers # unused for power constraint & AWGN
        self.noise_std = noise_std # unused for power constraint, used for AWGN 
        self.power = power # Is power normalized ?
        self.device = device 

    def forward(self, x1):
        output11 = np.sqrt(self.power * self.n_channel) * (torch.div(x1.t(),torch.norm(x1, p=2, dim=1)+1e-5).t()).to(self.device) #power constraint
        return output11


class awgn_channel(nn.Module):
    def __init__(self, ):
        super(awgn_channel, self).__init__()
    def forward(self, input_signal, snr_db):
        snr_linear = 10 ** (snr_db / 10.0)
        signal_power = torch.mean(torch.abs(input_signal) ** 2)  # 0.5 (0.4999)
        noise_power = signal_power / snr_linear
        noise_std = torch.sqrt(noise_power)

        noise = torch.randn_like(input_signal) * noise_std
        output_signal = input_signal + noise

        return output_signal