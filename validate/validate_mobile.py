from clip_mobile_dither4 import Clip_autoencoder

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

device = "cuda"
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

    # # import the entire model
    model =  Clip_autoencoder(cfgs, args, device).to(device)
    model.load_state_dict(torch.load("latest_models/mobile_pqvae_seperate/full/64-32-n_e-8.pt"))
    model.to(device)

    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    encoder_size_in_bytes = encoder_params * 4  
    encoder_size_in_mb = encoder_size_in_bytes / (1024 * 1024)

    decoder_params = sum(p.numel() for p in model.decoder.parameters())
    decoder_size_in_bytes = decoder_params * 4  
    decoder_size_in_mb = decoder_size_in_bytes / (1024 * 1024)

    total_params = sum(p.numel() for p in model.parameters())
    total_size_in_bytes = total_params * 4  
    total_size_in_mb = total_size_in_bytes / (1024 * 1024)

    print(total_size_in_mb)

    pdb.set_trace()


    test_features = torch.load("data/mobile/mobile_test_full.pt", map_location = device) 
    test_labels = torch.arange(1000).repeat_interleave(100)
    test_dataset = FeatureDataset(test_features, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch, shuffle=False)

    text_features = torch.load("data/labels/mobile_text_features.pt", map_location = device)

    test_accuracy = validate_epoch(model, test_loader, text_features, device) 
