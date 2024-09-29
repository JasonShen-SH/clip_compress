import os
import io
from PIL import Image
import torch
import pdb

imagenet_val_dir = 'data/imagenet/val'
all_dirs = sorted(os.listdir(imagenet_val_dir))
all_dirs = all_dirs[-201:]
all_dirs = all_dirs[:-1]

def compress_and_calculate_size(image, quality=75):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", quality=quality)
    compressed_size = len(buffer.getvalue())
    return compressed_size

original_sizes = []
compressed_sizes = []
total_bpp = []

for dir_path in sorted(all_dirs):
    subfolder_path = os.path.join(imagenet_val_dir, dir_path)
    for file in sorted(os.listdir(subfolder_path)):
        if file.endswith(('.png', '.jpg', '.jpeg', '.JPEG')):
            img_path = os.path.join(subfolder_path, file)
            img_original = Image.open(img_path)
            width, height = img_original.size
            channels = len(img_original.getbands())
            total_pixels = width * height * channels

            # original_size = len(image.tobytes())
            # original_sizes.append(original_size)

            compressed_size = compress_and_calculate_size(img_original)
            compressed_sizes.append(compressed_size)

            bpp = (compressed_size * 8) / total_pixels
            total_bpp.append(bpp)

    print(dir_path)

original_sizes = torch.tensor(original_sizes)
compressed_sizes = torch.tensor(compressed_sizes).float()

bpp_values = torch.tensor(total_bpp).float()
total_bpp = torch.sum(bpp_values)
average_bpp = torch.mean(bpp_values)
print(average_bpp)
# print(f"Average compressed size: {torch.mean(compressed_sizes):.2f} bytes")
