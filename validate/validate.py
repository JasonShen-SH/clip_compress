# from clip_rqvae import Clip_autoencoder
# from clip_rqvae_512_ground import Clip_autoencoder
from clip_rqvae_learn import Clip_autoencoder_learn
# from clip_dither_3 import Clip_autoencoder
# from clip_dither3_learn import Clip_autoencoder
# from clip_dither_4 import Clip_autoencoder
# from clip_dither_gan import Clip_autoencoder

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
# import torch.multiprocessing as mp
# # mp.set_start_method('spawn', force=True)

"""
三个层面提升：
1. 5*5, 7*7 但是这个层面上的提升，代价很大，因为compression rate上升速度很高，不建议
2. 星座点个数，例如2**12到2**13，compression rate上升速度较慢，
3. ResNet+Conv2d → Transformer系列
"""
device = "cuda:6"
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
    # pqvae: tiny
    # autoencoder: cifar10, cifar100, tiny, (stl失败)

    # args & cfgs & SNR(k)
    args = get_args()
    cfgs = load_config("cfg.toml")

    # # import the entire model
    encoder_output_dimension = 16
    levels = 8

    model = Clip_autoencoder_learn(cfgs, args, device, encoder_output_dimension, levels, tau=1).to(device)
    model.load_state_dict(torch.load(f"latest_models/improved_fsr_tau_1/encoder-output-dimension-{encoder_output_dimension}-levels-{levels}.pt", map_location=device)) # rq
    
    # model = Clip_autoencoder(cfgs, args, device).to(device)
    # model.load_state_dict(torch.load(f"latest_models/vqvae/{args.bw}-n_e-{args.n_e}-1.2.pt", map_location=device)) # vqvae
    # model.load_state_dict(torch.load("latest_models/gan/768-16-8.pt", map_location=device))
    # model.to(device)

    # load pretrained parts (if needed)
    # model_pretrain = Clip_autoencoder_original(cfgs, args, device).to(device)
    # model_pretrain.load_state_dict(torch.load("latest_models/benchmark/256-128-res-3-encoder-decoder-20-contrastive.pt", map_location=device))
    # model.encoder.load_state_dict(model_pretrain.encoder.state_dict())
    # model.decoder.load_state_dict(model_pretrain.decoder.state_dict())

    # for param in model.encoder.parameters():
    #     param.requires_grad = False
    # for param in model.decoder.parameters():
    #     param.requires_grad = False
    # for param in model.big_machine.embedding.parameters():
    #     param.requires_grad = False


    # import train/test dataloader
    # full_features, full_labels = torch.load("test_features_labels.pt", map_location = device) 
    # full_dataset = FeatureDataset(full_features, full_labels)
    # train_dataset = Subset(full_dataset, list(range(0, 800 * 50)))
    # test_dataset = Subset(full_dataset, list(range(800 * 50, 1000 * 50)))
    # train_loader = DataLoader(train_dataset, batch_size=args.train_batch, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=args.test_batch, shuffle=False)

    full_features, full_labels = torch.load("caltech101_features_labels.pt", map_location = device) 
    full_dataset = FeatureDataset(full_features, full_labels)
    dataloader = DataLoader(full_dataset, batch_size=args.test_batch, shuffle=False)

    # features, labels = torch.load("cifar10_features_labels.pt", map_location = device) 
    # dataset = FeatureDataset(features, labels)
    # dataloader = DataLoader(dataset, batch_size=args.test_batch, shuffle=False)

    # features, labels = torch.load("cifar100_features_labels.pt", map_location = device) 
    # dataset = FeatureDataset(features, labels)
    # dataloader = DataLoader(dataset, batch_size=args.test_batch, shuffle=False)

    # features, labels = torch.load("stl10_features_labels.pt", map_location = device) 
    # dataset = FeatureDataset(features, labels)
    # dataloader = DataLoader(dataset, batch_size=args.test_batch, shuffle=False)

    # features, labels = torch.load("tiny_imagenet_features_labels.pt", map_location = device) 
    # dataset = FeatureDataset(features, labels)
    # dataloader = DataLoader(dataset, batch_size=args.test_batch, shuffle=False)


    # train and validate
    # text_features = torch.load("data/labels/text_features.pt", map_location = device)
    # text_features = torch.load("512_text_features.pt", map_location = device)
    text_features = torch.load("caltech101_text_features.pt", map_location = device)
    # text_features = torch.load("data/labels/cifar10_text_features.pt", map_location = device)
    # text_features = torch.load("data/labels/cifar100_text_features.pt", map_location = device)
    # text_features = torch.load("data/labels/tiny_imagenet_text_labels.pt", map_location = device)
    # text_features = torch.load("stl_text_features.pt", map_location = device)

    epoch = 0
    highest_test_accuracy = 0

    # print("VQ-VAE!")
    # print(f"256-128-res-3-{args.n_e}")
    # print(f"snr: {args.snr_db}")
    # print("bw:16, n_e:8")

    while epoch < args.epoch:
        test_accuracy = validate_epoch(model, dataloader, text_features, device) 

        if test_accuracy > highest_test_accuracy:
            highest_test_accuracy = test_accuracy

        epoch += 1

    print(f"highest_test_accuracy is {highest_test_accuracy}")
    # print(f"256-128-res-3-{args.n_e}")
    # print(f"snr: {args.snr_db}")