import os
import io
from PIL import Image
import torch
import pdb
import subprocess

# 设置ImageNet验证集的路径
imagenet_val_dir = 'data/imagenet/val'
all_dirs = sorted(os.listdir(imagenet_val_dir))
all_dirs = all_dirs[-201:]
all_dirs = all_dirs[:-1]


def compress_and_calculate_size(image, image_index, quality=28):
    temp_jpeg_path = f'temp_{image_index}.jpg'
    image.save(temp_jpeg_path, format="JPEG", quality=100) 
    
    temp_bpg_path = f'temp_{image_index}.bpg'
    subprocess.run(['bpgenc', temp_jpeg_path, '-o', temp_bpg_path, '-q', str(quality)])
    
    with open(temp_bpg_path, 'rb') as f:
        compressed_size = len(f.read())
    
    return compressed_size

# 统计原始图像大小和压缩后的图像大小
original_sizes = []
compressed_sizes = []
total_bpp = []

image_index = 0
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

            compressed_size = compress_and_calculate_size(img_original, image_index)
            compressed_sizes.append(compressed_size)

            bpp = (compressed_size * 8) / total_pixels
            total_bpp.append(bpp)

            image_index += 1

    print(dir_path)

original_sizes = torch.tensor(original_sizes)
compressed_sizes = torch.tensor(compressed_sizes).float()

bpp_values = torch.tensor(total_bpp).float()
total_bpp = torch.sum(bpp_values)
average_bpp = torch.mean(bpp_values)
print(average_bpp)
# print(f"Average compressed size: {torch.mean(compressed_sizes):.2f} bytes")
pdb.set_trace()
