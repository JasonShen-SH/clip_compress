import torch
import torch.nn.functional as F
import pdb

features, labels = torch.load("test_features_labels.pt")

l2_norms = torch.norm(features, p=2, dim=1)  

specific_sample_l2 = l2_norms[2]
specific_sample_l2_2 = l2_norms[3]

print(torch.abs(specific_sample_l2 - specific_sample_l2_2))

# l2_distances = torch.abs(l2_norms - specific_sample_l2)

# min_value, min_index = torch.min(l2_distances[50:], dim=0)
# min_index += 50

# print(min_value)
# print(min_index)

# pdb.set_trace()

# features_normalized = F.normalize(features, p=2, dim=1)

# first_sample = features_normalized[2].unsqueeze(0) 
# second_sample = features_normalized[3].unsqueeze(0) 

# # (0-947) and (2-346)

# cosine_similarity = torch.mm(first_sample, second_sample.t())
# # cosine_similarity = torch.mm(first_sample, features_normalized.t())
# print(torch.max(cosine_similarity))