import clip
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.datasets import CIFAR100
import numpy as np
import pdb

class FeatureDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
    
class CustomCIFAR10Dataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform
        model, _ = clip.load("ViT-L/14@336px", device="cuda")
        self.model = model
        self.device = "cuda"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = transforms.ToPILImage()(img)

        if self.transform is not None:
            img = self.transform(img)

        with torch.no_grad():
            img = img.unsqueeze(0).to(self.device)
            feature_vector = self.model.encode_image(img).squeeze(0) 

        return feature_vector, target


def extract_features(dataset):
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False) 
    features, labels = [], []

    for images, lbls in dataloader:
        features_batch = images.view(images.size(0), -1)  # Example feature extraction: flatten images
        features.append(features_batch)
        labels.append(lbls)

    features = torch.cat(features)
    labels = torch.cat(labels)
    return features, labels


def get_feature_loaders():
    preprocess = transforms.Compose([
        transforms.Resize(336),
        transforms.CenterCrop((336, 336)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    ])

    # 加载 CIFAR-10 数据集
    train_dataset = CIFAR100(root='./data', train=True, download=True)
    # test_dataset = CIFAR100(root='./data', train=False, download=True)

    # 提取测试集特征
    train_dataset = CustomCIFAR10Dataset(train_dataset.data, train_dataset.targets, transform=preprocess)
    train_features, train_labels = extract_features(train_dataset)
    torch.save((train_features, train_labels), "cifar100_train_features_labels.pt")

    # test_dataset = CustomCIFAR10Dataset(test_dataset.data, test_dataset.targets, transform=preprocess)
    # test_features, test_labels = extract_features(test_dataset)
    # torch.save((test_features, test_labels), "cifar100_test_features_labels.pt")

    return (train_features, train_labels), (test_features, test_labels)

if __name__ == '__main__':
    (train_features, train_labels), (test_features, test_labels) = get_feature_loaders()
    print("Train and test features extracted and saved.")
