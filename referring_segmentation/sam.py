from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import clip
from PIL import Image
import pdb
from torch.nn import functional as F

sam_checkpoint = "/mnt/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(sam)


image = cv2.imread('/mnt/dog_and_cat.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

masks = mask_generator.generate(image)


device = "cuda"
model_clip, preprocess = clip.load("ViT-L/14@336px", device=device)

image_features_list = []

for mask_index in range(len(masks)):
    mask = masks[mask_index]['segmentation']

    masked_image = np.zeros_like(image)
    masked_image[mask] = image[mask]

    pil_image = Image.fromarray(masked_image)
    preprocessed_image = preprocess(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_feature = model_clip.encode_image(preprocessed_image)
        image_features_list.append((image_feature, masked_image))


text = clip.tokenize(["a dog"]).to(device)
with torch.no_grad():
    text_features = model_clip.encode_text(text)


def get_logits(image_features, text_features, logit_scale):
    logits_per_image = (logit_scale * image_features @ text_features.T)
    logits_per_text = (logit_scale * text_features @ image_features.T)
    return logits_per_image, logits_per_text


all_image_features = torch.cat([f[0] for f in image_features_list], dim=0)

all_image_features = F.normalize(all_image_features, p=2, dim=1)
text_features = F.normalize(text_features, p=2, dim=1)


logits_per_image, _ = get_logits(all_image_features, text_features, logit_scale=1)
best_match_index = torch.argmax(logits_per_image)
best_match_image = image_features_list[best_match_index][1]

plt.figure(figsize=(8, 8))
plt.imshow(best_match_image)
plt.axis('off')
plt.title(f"Best Match with Logits {logits_per_image[best_match_index].item():.2f}")

plt.savefig("result.png")
plt.show()
