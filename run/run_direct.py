from clip_amazing import Clip_autoencoder

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
    
    args = get_args()
    cfgs = load_config("cfg.toml")

    for levels in [2,4,8,16,32]:
        model =  Clip_autoencoder(cfgs, args, device, levels).to(device)
        model.quantization_points.requires_grad = True

        # full_features, full_labels = torch.load("test_features_labels.pt", map_location = device) 
        # full_dataset = FeatureDataset(full_features, full_labels)
        # train_dataset = Subset(full_dataset, list(range(0, 800 * 50)))
        # test_dataset = Subset(full_dataset, list(range(800 * 50, 1000 * 50)))
        # train_loader = DataLoader(train_dataset, batch_size=args.train_batch, shuffle=True)
        # test_loader = DataLoader(test_dataset, batch_size=args.test_batch, shuffle=False)

        # full_features, full_labels = torch.load("test_features_labels.pt", map_location = device) 
        # full_dataset = FeatureDataset(full_features, full_labels)
        # train_dataset = Subset(full_dataset, list(range(0, 800 * 50)))
        # test_dataset = Subset(full_dataset, list(range(800 * 50, 1000 * 50)))
        # train_loader = DataLoader(train_dataset, batch_size=args.train_batch, shuffle=True)
        # test_loader = DataLoader(test_dataset, batch_size=args.test_batch, shuffle=False)

        features, labels = torch.load("cifar10_features_labels.pt", map_location = device) 
        dataset = FeatureDataset(features, labels)
        dataloader = DataLoader(dataset, batch_size=args.train_batch, shuffle=False)

        # features, labels = torch.load("stl10_features_labels.pt", map_location = device) 
        # dataset = FeatureDataset(features, labels)
        # dataloader = DataLoader(dataset, batch_size=args.train_batch, shuffle=False)

        # features, labels = torch.load("tiny_imagenet_features_labels.pt", map_location = device) 
        # dataset = FeatureDataset(features, labels)
        # dataloader = DataLoader(dataset, batch_size=args.train_batch, shuffle=False)


        # train and validate
        epoch = 0
        # text_features = torch.load("data/labels/text_features.pt", map_location = device)           
        text_features = torch.load("data/labels/cifar10_text_features.pt", map_location = device)
        # text_features = torch.load("data/labels/cifar100_text_features.pt", map_location = device)
        # text_features = torch.load("tiny_imagenet_text_features.pt", map_location = device)
        # text_features = torch.load("stl_text_features.pt")

        # save model

        highest_test_accuracy = 0

        # print("encoder_output_dimension: ", dimension)
        # print("levels: ", levels)

        # print("think twice, we are going to save the model !!!")

        while epoch < 10:
            loss = train_epoch(args, model, dataloader, device)
            # print(f'Epoch {epoch}, Loss: {loss:.4f}')

            test_accuracy = validate_epoch(model, dataloader, text_features, device) 

            if test_accuracy > highest_test_accuracy:
                highest_test_accuracy = test_accuracy
                # torch.save(model.state_dict(), os.path.join(model_save_path, f"encoder-output-dimension-{dimension}-levels-{levels}.pt"))
                torch.save(model.quantization_points.data, f"quantization_points_cifar10_{levels}.pt")
            epoch += 1

        print(f"saved {levels}!")
        # print(f"highest_test_accuracy is {highest_test_accuracy}","encoder_output_dimension: ", dimension, "levels: ", levels)