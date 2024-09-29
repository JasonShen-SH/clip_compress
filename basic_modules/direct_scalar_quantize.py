import torch
from torch.nn import functional as F
import pdb
from process_images import FeatureDataset
from torch.utils.data import DataLoader, Dataset, Subset
from get_args import get_args

device = "cuda"
args = get_args()

def get_logits(image_features, text_features, logit_scale):
    logits_per_image = (logit_scale * image_features @ text_features.T) # batch_size * batch_size
    logits_per_text = (logit_scale * text_features @ image_features.T)  
    # two matrixes are transpose
    return logits_per_image, logits_per_text    


# full_features, full_labels = torch.load("test_features_labels.pt", map_location = device) 
# full_dataset = FeatureDataset(full_features, full_labels)
# dataloader = DataLoader(full_dataset, batch_size=args.train_batch, shuffle=True)
# train_dataset = Subset(full_dataset, list(range(0, 800 * 50)))
# test_dataset = Subset(full_dataset, list(range(800 * 50, 1000 * 50)))
# train_loader = DataLoader(train_dataset, batch_size=args.train_batch, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=args.test_batch, shuffle=False)

# test_features, test_labels = torch.load("cifar10_features_labels.pt", map_location = device) 
# test_dataset = FeatureDataset(test_features, test_labels)
# dataloader = DataLoader(test_dataset, batch_size=args.test_batch, shuffle=False)

# features, labels = torch.load("cifar100_features_labels.pt", map_location = device) 
# dataset = FeatureDataset(features, labels)
# dataloader = DataLoader(dataset, batch_size=args.test_batch, shuffle=False)

# full_features, full_labels = torch.load("regularized_caltech101_features_labels.pt", map_location = device) 
# full_dataset = FeatureDataset(full_features, full_labels)
# dataloader = DataLoader(full_dataset, batch_size=args.test_batch, shuffle=False)

# features, labels = torch.load("stl10_features_labels.pt", map_location = device) 
# dataset = FeatureDataset(features, labels)
# dataloader = DataLoader(dataset, batch_size=args.test_batch, shuffle=False)

# features, labels = torch.load("tiny_imagenet_features_labels.pt", map_location = device) 
# dataset = FeatureDataset(features, labels)
# dataloader = DataLoader(dataset, batch_size=args.test_batch, shuffle=False)

test_features = torch.load("data/mobile/mobile_test_full.pt", map_location = device) 
test_labels = torch.arange(1000).repeat_interleave(100)
test_dataset = FeatureDataset(test_features, test_labels)
test_loader = DataLoader(test_dataset, batch_size=args.test_batch, shuffle=False)

def scalar_quantize(tensor, levels):
    # flat_tensor = tensor.view(-1)
    # min_val = -0.5224
    # max_val = 0.5224
    # range_val = max_val - min_val
    # step = range_val / (levels-1)
    # quantization_points = torch.tensor([min_val + i * step for i in range(levels)]).to(device)
    quantization_points = torch.load("quantization_points_16.pt", map_location=device)
    quantization_points_repeat = quantization_points.repeat(tensor.shape[0], tensor.shape[1], 1)
    quantization_points_repeat = quantization_points_repeat.view(-1,quantization_points_repeat.shape[-1])
    tensor_flat = tensor.reshape(-1).unsqueeze(1)
    diffs = torch.abs(quantization_points_repeat - tensor_flat)
    min_indices = torch.argmin(diffs, dim=1)
    quantized_tensor = quantization_points[min_indices].unsqueeze(1)
    quantized_tensor = quantized_tensor.view(tensor.shape[0], tensor.shape[1])

    # pdb.set_trace()
    quantized_tensor = tensor + (quantized_tensor - tensor).detach()
    return quantized_tensor

# pdb.set_trace()
# text_features = torch.load("data/labels/cifar10_text_features.pt", map_location = device)
# text_features = torch.load("data/labels/text_features.pt", map_location = device).float()
# text_features = torch.load("data/labels/cifar100_text_features.pt", map_location = device)
# text_features = torch.load("stl_text_features.pt", map_location = device)
# text_features = torch.load("data/labels/tiny_imagenet_text_labels.pt", map_location = device)
# text_features = torch.load("caltech101_text_features.pt", map_location = device)
text_features = torch.load("data/labels/mobile_text_features.pt", map_location = device)


total = 0
correct = 0

with torch.no_grad():
    for features, labels in test_loader:
        features = features.to(float)
        features, labels = features.to(device), labels.to(device)
        features = F.normalize(features, p=2, dim=1).float()
        # quantized_features = scalar_quantize(features, levels=2).float()

        text_features = F.normalize(text_features, p=2, dim=1).to(float)

        # logits_images, logits_text = get_logits(quantized_features.float(), text_features[-200:].float(), 1)
        logits_images, logits_text = get_logits(features.float(), text_features.float(), 1)

        probs = logits_images.softmax(dim=1) 
        predicted = torch.argmax(probs, axis=1)

        total += labels.size(0)
        # labels = labels - 800
        correct += (predicted == labels).sum().item()

test_accuracy = 100 * correct / total
print(f'Accuracy on test set: {test_accuracy:.2f}%')

