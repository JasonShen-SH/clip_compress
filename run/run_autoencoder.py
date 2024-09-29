# from clip_rqvae import Clip_autoencoder  
from clip_rqvae_learn import Clip_autoencoder_learn
# from clip_rqvae_512_ground import Clip_autoencoder
# from clip_rqvae_fsr import Clip_autoencoder
# from clip_rqvae_lfq import Clip_autoencoder
# from clip_rqvae2 import Clip_autoencoder

from clip_rqvae import Clip_autoencoder

from get_args import get_args
from runner import train_epoch, validate_epoch, train_epoch_snr, validate_epoch_snr
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
device = "cuda:2"
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

def extract_numbers(filename):
    match = re.search(r'dimension-(\d+)-levels-(\d+)', filename)
    if match:
        dimension = int(match.group(1))
        levels = int(match.group(2))
        return dimension, levels
    return float('inf'), float('inf') 

def custom_sort_key(filename):
    if filename.startswith('encoder'):
        priority = 0  
        dimension, levels = extract_numbers(filename)
        return (priority, dimension, levels)
    else:
        return (float('inf'), float('inf'), float('inf')) 
    
if __name__ == '__main__':
    folder_path = "latest_models/improved_fsr"
    for models in sorted(os.listdir(folder_path), key=custom_sort_key):
        model_path = os.path.join(folder_path, models)
        dimension, levels = map(int, re.search(r'dimension-(\d+)-levels-(\d+)', model_path).groups())

        args = get_args()
        cfgs = load_config("cfg.toml")

        tau = 0.00001
        model =  Clip_autoencoder_learn(cfgs, args, device, dimension, levels, tau=tau).to(device)
        model.quantization_points.requires_grad = True

        model_pretrain = Clip_autoencoder(cfgs, args, device, dimension, levels).to(device)
        model_pretrain.load_state_dict(torch.load(f"latest_models/improved_fsr/encoder-output-dimension-{dimension}-levels-{levels}.pt", map_location=device))
        model.encoder.load_state_dict(model_pretrain.encoder.state_dict())
        model.decoder.load_state_dict(model_pretrain.decoder.state_dict())

        for param in model.encoder.parameters():
            param.requires_grad = False
        for param in model.decoder.parameters():
            param.requires_grad = False

        full_features, full_labels = torch.load("test_features_labels.pt", map_location = device) 
        full_dataset = FeatureDataset(full_features, full_labels)
        train_dataset = Subset(full_dataset, list(range(0, 800 * 50)))
        test_dataset = Subset(full_dataset, list(range(800 * 50, 1000 * 50)))
        train_loader = DataLoader(train_dataset, batch_size=args.train_batch, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.test_batch, shuffle=False)

        # train_features, train_labels = torch.load("cifar10_train_features_labels.pt", map_location = device) 
        # train_dataset = FeatureDataset(train_features, train_labels)
        # train_loader = DataLoader(train_dataset, batch_size=args.train_batch, shuffle=True)
        # test_features, test_labels = torch.load("cifar10_test_features_labels.pt", map_location = device) 
        # test_dataset = FeatureDataset(test_features, test_labels)
        # test_loader = DataLoader(test_dataset, batch_size=args.test_batch, shuffle=False)

        # train and validate
        epoch = 0
        # text_features = torch.load("data/labels/text_features.pt", map_location = device)
        text_features = torch.load("data/labels/text_features.pt", map_location = device)
        # text_features = torch.load("data/labels/cifar10_text_features.pt", map_location = device)

        # save model
        model_save_path = f"latest_models/improved_fsr_tau_{tau}"
        os.makedirs(model_save_path, exist_ok=True)

        point_save_path = f"quantization_points_tau_{tau}"
        os.makedirs(point_save_path, exist_ok=True)

        highest_test_accuracy = 0

        # print("encoder_output_dimension: ", dimension)
        # print("levels: ", levels)

        # print("think twice, we are going to save the model !!!")

        while epoch < args.epoch:
            loss = train_epoch(args, model, train_loader, device)
            # print(f'Epoch {epoch}, Loss: {loss:.4f}')

            test_accuracy = validate_epoch(model, test_loader, text_features, device) 

            if test_accuracy > highest_test_accuracy:
                highest_test_accuracy = test_accuracy
                torch.save(model.state_dict(), os.path.join(model_save_path, f"encoder-output-dimension-{dimension}-levels-{levels}.pt"))
                torch.save(model.quantization_points.data, f"quantization_points_tau_{tau}/encoder-output-dimension-{dimension}-levels-{levels}.pt")
        
            epoch += 1

        print(f"highest_test_accuracy is {highest_test_accuracy}","encoder_output_dimension: ", dimension, "levels: ", levels, "tau: ", tau)