# from clip_dither_3 import Clip_autoencoder
# from clip_dither_4 import Clip_autoencoder
# from clip_dither_gan import Clip_autoencoder
from clip_multi_scale import Clip_autoencoder

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
    new_full_features, new_full_labels = torch.load("multi_test_features_labels_2_real.pt", map_location = device)
    full_features = new_full_features.reshape(-1, 768); full_labels = new_full_labels.reshape(-1)
    # pdb.set_trace()
    full_features = full_features.view(-1, 5, 768)
    full_features = full_features[:, 1:, :].reshape(-1, 768)
    full_labels = full_labels.view(-1, 5)
    full_labels = full_labels[:, 1:].reshape(-1)
    full_dataset = FeatureDataset(full_features, full_labels)
    
    train_dataset = Subset(full_dataset, list(range(0, 800 * 50 * 4)))
    test_dataset = Subset(full_dataset, list(range(800 * 50 * 4, 1000 * 50 * 4)))
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
    text_features = torch.load("data/labels/text_features.pt", map_location = device)
    # text_features = torch.load("data/labels/cifar10_text_features.pt", map_location = device)

    # save model
    model_save_path = "latest_models"
    os.makedirs(model_save_path, exist_ok=True)

    highest_test_accuracy = 0

    print("VQ-VAE!")
    print(f"bw:{args.bw}, n_e:{args.n_e}")
    print("multi-2")

    # print("think twice, we are going to save the model !!!")

    while epoch < args.epoch:
        loss = train_epoch(args, model, train_loader, device)
        print(f'Epoch {epoch}, Loss: {loss:.4f}')

        test_accuracy = validate_epoch(model, test_loader, text_features, device) 

        if test_accuracy > highest_test_accuracy:
            highest_test_accuracy = test_accuracy
            # torch.save(model.state_dict(), os.path.join(model_save_path, f"vqvae_hanwei/{args.bw}-n_e-{args.n_e}-1.2.pt"))
        epoch += 1

    print(f"highest_test_accuracy is {highest_test_accuracy}")
    print(f"bw:{args.bw}, n_e:{args.n_e}")
    print("multi-2")