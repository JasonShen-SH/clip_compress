from clip_mobile_dither4 import Clip_autoencoder

# mobile + pqvae_separate

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
    # pretrained_embedding = torch.load("cluster_centers/new/cluster_centers__256_train.pt")
    # pretrained_embedding = pretrained_embedding.float()

    # # import the entire model
    model =  Clip_autoencoder(cfgs, args, device).to(device)
    # model.load_state_dict(torch.load("latest_models/benchmark/transformer-20-contrastive-1.pt"))
    # model.to(device)

    # load pretrained parts (if needed)
    # model_pretrain = Clip_autoencoder_original(cfgs, args, device).to(device)
    # model_pretrain.load_state_dict(torch.load("latest_models/benchmark/256-128-res-3-encoder-decoder-20-contrastive.pt", map_location=device))
    # model.encoder.load_state_dict(model_pretrain.encoder.state_dict())
    # model.decoder.load_state_dict(model_pretrain.decoder.state_dict())
    # model.big_machine.embedding.weight.data.copy_(model_pretrain.big_machine.embedding.weight.data)

    # for param in model.encoder.parameters():
    #     param.requires_grad = False
    # for param in model.decoder.parameters():
    #     param.requires_grad = False
    # for param in model.big_machine.embedding.parameters():
    #     param.requires_grad = False

    # import train/test dataloader 
    train_features = torch.load("data/mobile/mobile_train.pt", map_location = device) 
    train_labels = torch.arange(1000).repeat_interleave(100)
    train_dataset = FeatureDataset(train_features, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch, shuffle=True)

    # train_features, train_labels = torch.load("data/mobile/mobile_train_full.pt", map_location = device) 
    # train_dataset = FeatureDataset(train_features, train_labels)
    # train_loader = DataLoader(train_dataset, batch_size=args.train_batch, shuffle=True)

    test_features = torch.load("data/mobile/mobile_test_full.pt", map_location = device) 
    test_labels = torch.arange(1000).repeat_interleave(100)
    test_dataset = FeatureDataset(test_features, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch, shuffle=False)
    # mobile_test是每类随机选取的50个, mobile_test_full是全部的最后100个（每类）

    # train and validate
    epoch = 0
    text_features = torch.load("data/labels/mobile_text_features.pt", map_location = device)

    # save model
    model_save_path = "latest_models/mobile_pqvae_seperate"
    os.makedirs(model_save_path, exist_ok=True)

    highest_test_accuracy = 0

    while epoch < args.epoch:
        loss = train_epoch(args, model, train_loader, device)
        print(f'Epoch {epoch}, Loss: {loss:.4f}')

        test_accuracy = validate_epoch(model, test_loader, text_features, device) 

        if test_accuracy > highest_test_accuracy:
            highest_test_accuracy = test_accuracy
            torch.save(model.state_dict(), os.path.join(model_save_path, f"64-{args.bw}-n_e-{args.n_e}.pt"))
        epoch += 1

    print(f"highest_test_accuracy is {highest_test_accuracy}, bw:{args.bw}, n_e:{args.n_e}")