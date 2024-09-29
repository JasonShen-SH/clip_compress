import clip
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
import os
import pdb
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)


class CustomImageDataset(datasets.ImageFolder):
    def __init__(self, root, transform=None, patch_size=336):
        super().__init__(root, transform)
        self.device = "cuda:2"
        self.model, _ = clip.load("ViT-L/14@336px", device=self.device)
        self.transform = transform
        self.patch_size = patch_size

    def get_patches(self, image, scale): # 对于某种具体的维度
        w, h = image.size
        patch_size = int(w / scale)  # w 和 h 是等尺寸的
        patches = []
        for i in range(0, w, patch_size):
            for j in range(0, h, patch_size):
                patch = image.crop((i, j, i + patch_size, j + patch_size))
                # resized_patch = patch.resize((self.patch_size, self.patch_size), Image.BICUBIC)
                # patches.append(resized_patch)
                patches.append(patch)
        return patches # 4个 / 9个 / 16个

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = self.loader(path)

        scales = [1, 2, 3] # 不同的尺度
        all_features = []  #  最终有4个768维度的features

        with torch.no_grad():
            for scale in scales:
                if scale == 1:
                    transformed_img = self.transform(image).unsqueeze(0).to(self.device)
                    feature = self.model.encode_image(transformed_img).squeeze(0)
                    all_features.append(feature)
                else:
                    patches = self.get_patches(image, scale)
                    features = []
                    for patch in patches: # 4/9/16 个 patches
                        # patch = transforms.ToTensor()(patch).unsqueeze(0).to(self.device)
                        # patch = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                        #                              std=[0.26862954, 0.26130258, 0.27577711])(patch)
                        patch = self.transform(patch).unsqueeze(0).to(self.device)
                        feature = self.model.encode_image(patch).squeeze(0)
                        features.append(feature)
                        
                    average_feature = torch.mean(torch.stack(features), dim=0)
                    all_features.append(average_feature)

        return torch.stack(all_features), target

def extract_features(dataset):
    batch_size = 128
    loader = DataLoader(dataset, batch_size, num_workers=4, shuffle=False)

    all_features = []
    all_labels = []

    for batch in loader:
        images, labels = batch
        features = []

        for i in range(images.size(0)):
            feature, _ = dataset[i]
            features.append(feature)
        
        all_features.append(torch.stack(features))
        all_labels.append(labels)

        print(f"Processed {len(all_features) * batch_size}/{len(dataset)}")

    all_features = torch.cat(all_features)
    all_labels = torch.cat(all_labels)

    return all_features, all_labels

def get_feature_loaders():
    preprocess = transforms.Compose([
        transforms.Resize(336),
        transforms.CenterCrop((336, 336)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    ])

    test_dataset = CustomImageDataset(root='data/imagenet/val', transform=preprocess)
    print("b")

    test_features, test_labels = extract_features(test_dataset)
    print("real test_dataset prepared!")

    torch.save((test_features, test_labels), "multi_test_features_labels_3.pt")

    return (test_features, test_labels)

if __name__ == '__main__':
    test_data = get_feature_loaders()
