import torch
import torch.nn.functional as F
from itertools import combinations
import pdb

features, full_labels = torch.load("train_coco_features_labels.pt", map_location = "cuda:5")

# 计算特征向量对之间的 MSE
def mse(v1, v2):
    return F.mse_loss(v1, v2, reduction='none').mean()

# 计算方向差异（余弦相似度）
def cosine_similarity(v1, v2):
    return F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()

similar_mse_pairs = []
mse_threshold = 1e-5  
cosine_similarity_threshold = 0.5  

all_mse = []

lowest_mse = 10

features = features.squeeze()

i=0
for j in range(1, len(features), 1):
    if mse(features[i], features[j])<lowest_mse:
        lowest_mse = mse(features[0], features[j])
        best_match = j
        print(lowest_mse)
        print(j)

# i=0 和 j=19575

pdb.set_trace()