from clip_text import Clip_autoencoder
import tqdm
import sys
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


data_path = 'val_text_data_512.pkl'  # 已经是768维度了
data_dict = pickle.load(open(data_path, 'rb'))

class ClipEmbeddingDataset(Dataset):
    def __init__(self, data_dict):
        self.image_embeddings = data_dict['clip_image_embedding']
        self.text_embeddings = data_dict['clip_text_embedding']
        self.keys = list(self.image_embeddings.keys())  # 确保key的顺序是对应的

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        image_embedding = torch.tensor(self.image_embeddings[key], dtype=torch.float32)
        text_embedding = torch.tensor(self.text_embeddings[key], dtype=torch.float32)
        return image_embedding, text_embedding


dataset = ClipEmbeddingDataset(data_dict)

def extract_features(dataset):
    img_feats = []
    txt_feats = []
    # pdb.set_trace()

    for i in range(len(dataset)):
        img_feat, txt_feat = dataset[i]
        img_feat = img_feat.unsqueeze(0)
        txt_feat = txt_feat.unsqueeze(0)
        img_feats.append(img_feat)
        txt_feats.append(txt_feat)

    img_feats = torch.stack(img_feats)
    txt_feats = torch.stack(txt_feats)

    return img_feats, txt_feats

features, labels = extract_features(dataset)

torch.save((features, labels), "val_coco_features_labels_512.pt")