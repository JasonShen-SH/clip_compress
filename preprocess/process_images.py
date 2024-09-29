import clip
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
# from zmq import device
from get_args import get_args
import pdb

class FeatureDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
    
class CustomImageDataset(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform)
        device = "cuda:6"
        # model, _ = clip.load("ViT-L/14@336px", device=device)
        model, _ = clip.load("ViT-B/32", device=device)
        self.model = model
        self.device = device
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)

        # img → img
        if self.transform is not None:
            sample = self.transform(sample)

        # img → feature
        with torch.no_grad():
            sample = sample.unsqueeze(0).to(self.device)
            feature_vector = self.model.encode_image(sample).squeeze(0) 

        return feature_vector, target

def extract_features(dataset):
    features = []
    labels = []

    for i in range(len(dataset)):
        feature, label = dataset[i]
        features.append(feature)
        labels.append(label)

    features = torch.stack(features)
    labels = torch.tensor(labels)

    return features, labels

def get_feature_loaders():
    # preprocess = transforms.Compose([
    #     transforms.Resize(336),
    #     transforms.CenterCrop((336, 336)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    # ])

    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])

    # train_dataset = CustomImageDataset(root='/home/cb222/rs1923/deepjscc-clip/data/imagenet/train', transform=preprocess)
    # print("a")

    test_dataset = CustomImageDataset(root='data/imagenet/val', transform=preprocess)
    print("b")

    # train_features, train_labels = extract_features(train_dataset)
    # print("real train_dataset prepared!")
    
    test_features, test_labels = extract_features(test_dataset)
    print("real test_dataset prepared!")

    # torch.save((train_features, train_labels), "train_features_labels.pt")
    torch.save((test_features, test_labels), "512_test_features_labels.pt")

    return (test_features, test_labels)

if __name__ == '__main__':
    # args = get_args()
    test_data = get_feature_loaders()