# from clip_cap import Clip_autoencoder

import tqdm
import sys
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
import random
from coco_dataset import ClipCocoDataset
import torch
import torch.nn as nn
from torch.nn import functional as nnf
from torch.utils.data import Dataset, DataLoader
from enum import Enum
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import os
import pickle
import sys
import argparse
import json
from typing import Tuple, Optional, Union
# import torch.multiprocessing as mp
# # mp.set_start_method('spawn', force=True)

"""
三个层面提升：
1. 5*5, 7*7 但是这个层面上的提升，代价很大，因为compression rate上升速度很高，不建议
2. 星座点个数，例如2**12到2**13，compression rate上升速度较慢，
3. ResNet+Conv2d → Transformer系列
"""
# device = "cuda:2"
torch.autograd.set_detect_anomaly(True)

def load_config(path):
    with open(path, 'r') as f:
        config = toml.load(f)
    # dict to SimpleNamespace, i.e. cfg['CHANNEL'] → cfg.CHANNEL
    config = {k: SimpleNamespace(**v) if isinstance(v, dict) else v for k, v in config.items()}
    return SimpleNamespace(**config)

def save_model(model, path):
    torch.save(model.state_dict(), path)
    # print(f"Model saved to {path}")

def save_codebook(vector_quantizer, path):
    torch.save(vector_quantizer.embedding.weight.data, path)
    # print(f"Codebook saved to {path}")

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
        # tokens: (batch_size, 40) ; the maximum length of a single caption is 40
        # prefix: (batch_size, 768)
        # mask: (batch_size, 50)
        embedding_text = self.gpt.transformer.wte(tokens) # 
        prefix_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.gpt_embedding_size) 
        # self.prefix_length=10, self.gpt_embedding_size=768
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        # pdb.set_trace()
        if labels is not None:  # default: None
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        return out

    def __init__(self, prefix_length: int, prefix_size: int = 768):
        super(ClipCaptionModel, self).__init__()
        self.prefix_length = prefix_length
        self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1] # 768
        self.clip_project = MLP((prefix_size, (self.gpt_embedding_size * prefix_length) // 2,
                                     self.gpt_embedding_size * prefix_length))

if __name__ == '__main__':
    # args & cfgs & SNR(k)
    args = get_args()
    cfgs = load_config("cfg.toml")
    # model =  Clip_autoencoder(cfgs, args, device, args.prefix_length).to(device)

    # device_ids = [2,3,4]
    # model = ClipCaptionModel(args.prefix_length)
    # model = nn.DataParallel(model, device_ids).to(device_ids[0])  
    # model.train()

    device = "cuda:6"
    model = ClipCaptionModel(args.prefix_length).to(device)
    model.train()

    data_path = 'data/coco2014/train_data.pkl'  # 已经是768维度了
    prefix_length = 10
    gpt2_type = "gpt2"
    normalize_prefix = False

    dataset = ClipCocoDataset(data_path, prefix_length, gpt2_type, normalize_prefix)
    print("dataset ready!")
    train_dataloader = DataLoader(dataset, batch_size=40, shuffle=True, drop_last=True)
    print("dataloader ready!")

    # for (tokens, masks, prefixes) in train_dataloader:
    #     pdb.set_trace()

    # train and validate
    lr = 2e-5
    warmup_steps = 5000
    total_epochs = 10
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_epochs * len(train_dataloader)
    )

    output_dir = "clipcap_original"
    os.makedirs(output_dir, exist_ok=True)
    output_prefix = ""

    epoch = 0
    while epoch < total_epochs:
        sys.stdout.flush()
        progress = tqdm(total=len(train_dataloader), desc="")
        for idx, (tokens, mask, prefix) in enumerate(train_dataloader):
            model.zero_grad()
            # tokens, mask, prefix = tokens.to(device_ids[0]), mask.to(device_ids[0]), prefix.to(device_ids[0], dtype=torch.float32)
            tokens, mask, prefix = tokens.to(device), mask.to(device), prefix.to(device, dtype=torch.float32)
            outputs = model(tokens, prefix, mask)  # 我们的prefix不是事先给定的，而是要经过quantize并变换的
            # mask指的是每句句子的真实元素位置（例如真实长度是28，padding后是40，那么前28是1，后12是0）(28就是28个tokens,也就是说明这句句子由28个tokens组成)
            # 这个outputs的shape是[64, 40, 50257]，它意思是预测出的本句句子的内容（这句句子已知（也是要求）由40个tokens组成，每个token可能从整个vocabulary中选择，
            # 而整个vocabulary中有50257个元素，e.g.第22个位置可能是第0个word，也可能是第50256个word
            # pdb.set_trace()
            logits = outputs.logits[:, dataset.prefix_length - 1: -1] # [64, 40, 50257]
            loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0)  # logits: [64, 40, 50257];  tokens: [64, 40]
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            progress.set_postfix({"loss": loss.item()})
            progress.update()
            # if (idx + 1) % 10000 == 0:
            #     torch.save(
            #         model.state_dict(),
            #         os.path.join(output_dir, f"{output_prefix}_original_latest.pt"),
            #     )
        progress.close()
        if epoch % args.save_every == 0 or epoch == total_epochs - 1:
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, f"{output_prefix}-{epoch:03d}-original.pt"),
            )

        epoch += 1
        