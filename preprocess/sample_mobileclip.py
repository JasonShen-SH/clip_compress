import json
import pickle
import torch
import numpy as np
import random
import pdb

# text_features_path = "/root/autodl-tmp/nameclip.json"

# with open(text_features_path, 'r') as file:
#     data = json.load(file)

# sorted_keys = sorted(data.keys())

# for i, key in enumerate(sorted_keys):
#     text_feature_per_class = torch.tensor(data[key][0]).view(1,-1)
#     if i==0:
#         text_features = text_feature_per_class
#     else:
#         text_features = torch.cat((text_features, text_feature_per_class), dim=0)

# torch.save(text_features, "data/labels/mobile_text_features.pt")

########################################################################
########################################################################

image_features_path = "/root/autodl-tmp/cutoff.pkl"

with open(image_features_path, 'rb') as file:
    data = pickle.load(file)

sorted_keys = sorted(data.keys())

for i, key in enumerate(sorted_keys):
    image_features_per_class = torch.tensor(data[key]).squeeze()
    class_length = len(data[key])

    train_indices = list(range(0, class_length - 100))
    train_tensor = image_features_per_class[train_indices]
    
    train_labels_per_class = torch.full((len(train_indices),), i, dtype=torch.long)
    
    if i == 0:
        train_features = train_tensor
        train_labels = train_labels_per_class
    else:
        train_features = torch.cat((train_features, train_tensor), dim=0)
        train_labels = torch.cat((train_labels, train_labels_per_class), dim=0)


torch.save( (train_features, train_labels), "data/mobile/mobile_train_full.pt")