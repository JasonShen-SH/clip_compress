from clip_dither_3_huffman import Clip_autoencoder

from get_args import get_args
from runner import train_epoch, validate_epoch, validate_epoch_snr
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
import re
# import torch.multiprocessing as mp
# # mp.set_start_method('spawn', force=True)

"""
三个层面提升：
1. 5*5, 7*7 但是这个层面上的提升，代价很大，因为compression rate上升速度很高，不建议
2. 星座点个数，例如2**12到2**13，compression rate上升速度较慢，
3. ResNet+Conv2d → Transformer系列
"""
device = "cuda:4"
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

    pattern = re.compile(r'(\d+)-n_e-(\d+)\-1.2.pt')

    # for filename in os.listdir("latest_models/vqvae"):
    for filename in os.listdir("huff_dicts_vqvae"):
        match = pattern.search(filename)
        if match:
            bw = int(match.group(1))  
            n_e = int(match.group(2))   

            # output_file = f"huff_dicts_vqvae/{bw}-n_e-{n_e}-1.2.pt"
            # if os.path.exists(output_file):
            #     print(f"File {output_file} already exists. Skipping this combination.")
            #     continue  
            
            args = get_args()
            cfgs = load_config("cfg.toml")

            try:
                model = Clip_autoencoder(cfgs, args, device, bw, n_e).to(device)
                model.load_state_dict(torch.load(f"latest_models/vqvae/{bw}-n_e-{n_e}-1.2.pt", map_location=device)) 
                model.to(device)
                huff_dict = torch.load(f"huff_dicts_vqvae/{bw}-n_e-{n_e}-1.2.pt")

                full_features, full_labels = torch.load("test_features_labels.pt", map_location = device) 
                full_dataset = FeatureDataset(full_features, full_labels)
                train_dataset = Subset(full_dataset, list(range(0, 800 * 50)))
                test_dataset = Subset(full_dataset, list(range(800 * 50, 1000 * 50)))
                train_loader = DataLoader(train_dataset, batch_size=args.train_batch, shuffle=True)
                test_loader = DataLoader(test_dataset, batch_size=args.test_batch, shuffle=False)

                text_features = torch.load("data/labels/text_features.pt", map_location = device)

                model.eval()
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    # huff_dict = model(images, labels)
                    # torch.save(huff_dict, f"huff_dicts_vqvae/{bw}-n_e-{n_e}-1.2.pt")
                    avg_length = model(images, labels, huff_dict)
                    print(f"{bw}-n_e-{n_e}-1.2, bpd is: {avg_length/768}, bits used is: {avg_length}")
                
            except RuntimeError as e:
                print(f"skipping huff_dicts {bw}-n_e-{n_e}-1.2.pt")
                torch.cuda.empty_cache()
                continue