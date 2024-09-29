import os
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
import torch
from compressai.zoo import models 
import pdb

print(list(models.keys()))
# 'bmshj2018-factorized', 'bmshj2018-hyperprior', 'mbt2018-mean', 'mbt2018', 'cheng2020-anchor', 'cheng2020-attn', 'ssf2020'

input_dir = 'data/imagenet/val'
output_dir = 'data/compress/compressai_imagenet_mbt2018-mean/val'
os.makedirs(output_dir, exist_ok=True)

all_dirs = sorted(os.listdir(input_dir))
dirs_to_process = all_dirs[-201:]
dirs_to_process = dirs_to_process[:-1]

net = models['mbt2018-mean'](quality=1, pretrained=True).eval()

batch_size = 10
to_tensor = ToTensor()
to_pil = ToPILImage()


for dir_path in sorted(dirs_to_process):
    subfolder_path = os.path.join(input_dir, dir_path)
    output_folder_path = os.path.join(output_dir, dir_path)
    os.makedirs(output_folder_path, exist_ok=True)
    
    image_batch = []
    file_names = []

    for file in sorted(os.listdir(subfolder_path)):
        if file.endswith(('.png', '.jpg', '.jpeg', '.JPEG')):
            img_path = os.path.join(subfolder_path, file)
            
            img = Image.open(img_path).convert('RGB')
            image_batch.append(to_tensor(img).unsqueeze(0))
            file_names.append(file)

            if len(image_batch) == batch_size:
                x_batch = torch.cat(image_batch, dim=0)  
                with torch.no_grad():
                    try:
                        out_enc = net.compress(x_batch)
                        out_dec = net.decompress(out_enc['strings'], out_enc['shape'])
                        for i, img_dec in enumerate(out_dec['x_hat']):
                            img_dec_pil = to_pil(img_dec)
                            save_path = os.path.join(output_folder_path, file_names[i])
                            img_dec_pil.save(save_path)
                            del img_dec_pil
                        # del x, out_enc, out_dec, img_dec
                        del x_batch, out_enc, out_dec
                        torch.cuda.empty_cache()
                    except:
                        pass

    print(f"Processed and saved: {dir_path}")
