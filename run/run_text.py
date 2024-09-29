from clip_text import Clip_autoencoder
import tqdm
import sys
from get_args import get_args
from runner import train_epoch, validate_epoch_text
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

    
if __name__ == '__main__':
    # args & cfgs & SNR(k)
    args = get_args()
    cfgs = load_config("cfg.toml")

    device = "cuda:7"
    model =  Clip_autoencoder(cfgs, args, device).to(device)
    # model = nn.DataParallel(model, device_ids).to(device_ids[0])  
    model.train()

    # data_path = 'train_text_data.pkl'  # 已经是768维度了
    # data_dict = pickle.load(open(data_path, 'rb'))

    
    full_features, full_labels = torch.load("train_coco_features_labels_512.pt", map_location = device) 
    full_dataset = FeatureDataset(full_features, full_labels)
    train_loader = DataLoader(full_dataset, batch_size=args.train_batch, shuffle=True)

    test_features, test_labels = torch.load("val_coco_features_labels_512.pt", map_location = device) 
    test_dataset = FeatureDataset(test_features, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch, shuffle=False)

    print("dataloader ready!")


    # train and validate

    model_save_path = "latest_models/benchmark"
    os.makedirs(model_save_path, exist_ok=True)

    lowest_val_loss = 100

    epoch = 0
    while epoch < 20:
        # loss = train_epoch_random(args, model, train_loader, device)
        loss = train_epoch(args, model, train_loader, device)
        # loss = train_epoch_validate(args, model, train_loader, text_features, device)
        print(f'Epoch {epoch}, Loss: {loss:.4f}')

        val_loss = validate_epoch_text(model, test_loader, device)
        if val_loss < lowest_val_loss:
            lowest_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(model_save_path, "latest_text_512_model.pt"))

        epoch = epoch + 1