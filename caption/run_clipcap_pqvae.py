from clip_cap_pqvae import Clip_autoencoder_cap
from clip_dither_3 import Clip_autoencoder  # import pretrained model

import tqdm
import sys
from get_args import get_args
# from runner import train_epoch, validate_epoch, train_epoch_random, validate_epoch_random
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

if __name__ == '__main__':
    # args & cfgs & SNR(k)
    args = get_args()
    cfgs = load_config("cfg.toml")

    print(f"bw-{args.bw}-n_e-{args.n_e}")

    device = "cuda:6"
    model =  Clip_autoencoder_cap(cfgs, args, device, args.prefix_length).to(device)


    # load pretrained (if needed)
    # 我发现这个pretrained-model还是可以的，效果不错，目前默认要加载它
    model_pretrain = Clip_autoencoder(cfgs, args, device).to(device)  # from clip_dither_2
    model_pretrain.load_state_dict(torch.load(f"latest_models/vqvae/{args.bw}-n_e-{args.n_e}-1.2.pt", map_location=device))
    model.encoder.load_state_dict(model_pretrain.encoder.state_dict())
    model.decoder.load_state_dict(model_pretrain.decoder.state_dict())
    model.quantize.embedding.weight.data.copy_(model_pretrain.quantize.embedding.weight.data)

    for param in model.encoder.parameters():
        param.requires_grad = False
    for param in model.decoder.parameters():
        param.requires_grad = False
    for param in model.quantize.embedding.parameters():
        param.requires_grad = False
 
    model.train()

    data_path = 'data/coco2014/train_data.pkl'  # 已经是768维度了
    prefix_length = 10
    gpt2_type = "gpt2"
    normalize_prefix = False

    dataset = ClipCocoDataset(data_path, prefix_length, gpt2_type, normalize_prefix)
    print("dataset ready!")
    train_dataloader = DataLoader(dataset, batch_size=args.train_batch, shuffle=True, drop_last=True)
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

    output_dir = "clipcap_pqvae"
    os.makedirs(output_dir, exist_ok=True)
    subfolder_dir = os.path.join(output_dir, "freeze_all")
    os.makedirs(subfolder_dir, exist_ok=True)
    subsubfolder_dir = os.path.join(subfolder_dir, f"{args.bw}-n_e-{args.n_e}")
    os.makedirs(subsubfolder_dir, exist_ok=True)
    output_prefix = ""
    
    epoch = 0
    while epoch < total_epochs:
        sys.stdout.flush()
        progress = tqdm(total=len(train_dataloader), desc="")
        for idx, (tokens, mask, prefix) in enumerate(train_dataloader):
            model.zero_grad()
            tokens, mask, prefix = tokens.to(device), mask.to(device), prefix.to(device, dtype=torch.float32)
            outputs, codebook_loss = model(tokens, prefix, mask)  # 我们的prefix不是事先给定的，而是要经过quantize并变换的
            # mask指的是每句句子的真实元素位置（例如真实长度是28，padding后是40，那么前28是1，后12是0）(28就是28个tokens,也就是说明这句句子由28个tokens组成)
            # 这个outputs的shape是[64, 40, 50257]，它意思是预测出的本句句子的内容（这句句子已知（也是要求）由40个tokens组成，每个token可能从整个vocabulary中选择，
            # 而整个vocabulary中有50257个元素，e.g.第22个位置可能是第0个word，也可能是第50256个word
            # pdb.set_trace()
            logits = outputs.logits[:, dataset.prefix_length - 1: -1] # [64, 40, 50257]
            # pdb.set_trace()
            ce_loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0)  # logits: [64, 40, 50257];  tokens: [64, 40]
            loss = ce_loss + 10 * codebook_loss
            # 对于直接对于浮点数的quantize, 一定要加上10倍的codebook_loss
            # 对于真正的0/1传输(含dither), 我们先加上10倍的codebook_loss试一试
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            progress.set_postfix({"loss": loss.item()})
            progress.update()
            # if (idx + 1) % 10000 == 0:
            #     torch.save(
            #         model.state_dict(),
            #         os.path.join(output_dir, f"{output_prefix}_codebook_dither_pretrain_latest.pt"),
            #     )
        progress.close()
        if epoch % args.save_every == 0 or epoch == total_epochs - 1:
            torch.save(
                model.state_dict(),
                # os.path.join(output_dir, f"{output_prefix}-{epoch:03d}-codebook_dither_pretrain_all_train.pt"),
                os.path.join(subsubfolder_dir, f"{output_prefix}-{epoch:03d}-codebook.pt"),
                # 如果在indices中，那么这里的dither仅是indices
            )

        epoch += 1
        # if epoch == 9:
        #     print("all_train")
