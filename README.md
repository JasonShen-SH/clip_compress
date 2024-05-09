# clip_compress
Methods to overcome the Vulnerability of CLIP to Image Compression

来自论文Understanding the Vulnerability of CLIP to Image Compression一文:



CLIP is trained on over 400 million image-text pairs, of which each image is 224*224.

The expeiment 强行将32*32和96*96的图片，通过bicubic interpolation缩放到224*224

这是实验的测试结果：

(两张图）可以看到，96*96的STL效果明显好于32*32的cifar10；

# 我们还用imagenet-tiny进行了测试，发现：224*224的图像



Three main methods:

一. 对于CLIP的image encoder提取出的特征进行优化 （之前先不压缩）

原本CLIP做inference的时候，是用的512维度的image和text features; 这也是要求做zero-shot inference的条件，因此对于特征维度缩减后，我们无法继续直接用clip预训练模型做inference的匹配，
而是做后端的分类任务，构造一个简单的分类器，研究是否可以在极低成本（如10个epoch）下，用现有的缩短了的features，经过简单的训练做分类

1. 特征位数的quantization (统一量化)
1000 images from X_test
  完整数位  小数位  test accuracy（CLIP）  进一步分类器（简单的meta-net作为classifier)
  12      8       92%   95.08%     
   8      4       88.5%   94.48%
   
2. autoencoder (没有结合text信息）
autoencoder的时候
3. autoencoder (结合text信息）

我们考虑到一个事实: 也就是往往传输的是压缩的图像，图像传输很难在信道中完全不受噪声影响被接收，因此考虑image denoising的工作
因为在denoise后，还需要将图像scale up to 224*224，因此我们将" image denoise + scale up"的步骤认为是super-resolution的过程


二. 对于进入image encoder之前的压缩图像进行还原

2.1 尝试用已有的通用模型，做迁移学习
[1] JPEG Artifact Correction using Denoising Diffusion Restoration Models -- Official Code Repository
[2] dcnn-denoise: Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising
[3] SRGAN: Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial
Network


2.2 完全基于当前的cifar10训练 (manaully reconstruction)
自己搭建的模型


