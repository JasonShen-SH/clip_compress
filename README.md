# CLIP Compress

## Literature Review
The paper **"Understanding the Vulnerability of CLIP to Image Compression"** highlights the sensitivity of CLIP regarding jpeg compression of imgs when performing zero-shot recognition task.
<div style="display: flex; justify-content: space-around;">
  <img src="imgs/cifar10.png" width="500">
  <img src="imgs/stl10.png" width="500">
</div>
<p style="text-align: center;">Average precision of CLIP predictions over the test dataset from cifar10/stl10 across different image qualities.</p>

## Main problems and objectives
The performance decline due to JPEG compression is much more significant for the CIFAR10 dataset compared to STL10, where the decrease is not particularly notable. 

The authors did not analyze this aspect in the paper, but we believe it could be attributed to the impact of **image size**. Since a 32x32 image inherently carries **limited information**, compression leads to a greater loss of detail. Additionally, the compression artifacts introduced by JPEG, such as **blocking effects, are exacerbated**, resulting in poorer performance.

Therefore, we focus on **improving JPEG Artifact Correction** in our work.

## Methodologies (I'm improving)
### Operations on image features

#### Image feature quantization
Post-train Quantization (PTQ)
| All digits     | Integer digits  |    Accuracy (CLIP's zero-shot prediction)    |  Classification Accuracy (meta-net for classifier)  |  
|----------------|-----------------|----------------------------------------------|-----------------------------------------------------|
| 12             | 8               |    92%                                       |    95.08%                                           |
| 8              | 4               |    88.5%                                     |    94.48%                                           |
<img src="imgs/quantizer.png" width="500">


#### The denoise of image features
The method is based on the assumption that image features are transmitted, rather than transmitting the images themselves after undergoing JPEG compression.
<img src="imgs/autoencoder_image.png" width="500">
<img src="imgs/autoencoder_image_text.png" width="500">
| All digits     | Integer digits  |    Accuracy (CLIP's zero-shot prediction)    |  Classification Accuracy (meta-net for classifier)  |  
|----------------|-----------------|----------------------------------------------|-----------------------------------------------------|
| 12             | 8               |    92%                                       |    95.08%                                           |
| 8              | 4               |    88.5%                                     |    94.48%                                           |

1. Optimizing the Features Extracted by CLIP's Image Encoder
Initially, CLIP conducts inference using 512-dimensional image and text features necessary for zero-shot inference. However, after feature dimensionality reduction, direct inference using the pre-trained CLIP model is not feasible. Instead, a simple classifier is constructed for backend classification tasks. This involves examining if using the reduced features, after minimal training (like 10 epochs), can perform classification effectively. The process includes:


### Operations on image itself 
1. Based on SRGAN
As besides artifact correction, we also need to scale the image to 224*224 as is required by CLIP's image encoder. We denote this process as the SR(super-resolution process).
<img src="imgs/SRGAN.png" width="500">
SRGAN提供多种倍率的放大倍数，包括*2, *4和*8；
根据cifar10图像是32*32的属性，我们选择*4和*8放大倍数
Our idea is as follows:
<img src="imgs/SRGAN_CLIP.png" width="500">
我们首先将预训练好的SRGAN(G&D）在cifar10数据集上直接做inference:
| CLIP pretrained model  | SRGAN scale  |  Accuracy (CLIP's zero-shot prediction)  |
|----------------|-----------------|----------------------------------------------|
| ViT-B/32       | 4               |    55.474%                                   |
| ViT-B/32       | 8               |    55.474%                                   |
|    \           |   \             |    65.09%                                    |
| RN50           | 4               |    45.968%                                   |  
| RN50           | 8               |    45.968%                                     |
|    \           | \               |    56.782%          |

我们的方法是，将SRGAN的生成器迁移到cifar10数据集上，即基于

Feature Quantization: Quantizing the number of bits used for each feature. An experiment using 1000 images from the X_test set demonstrated that reducing the precision of features slightly affects test accuracy but can be compensated with a simple meta-net classifier. For instance:
12 bits integer and 8 bits decimal portion resulted in 92% accuracy with CLIP and 95.08% with the classifier.
8 bits integer and 4 bits decimal portion resulted in 88.5% accuracy with CLIP and 94.48% with the classifier.
2. Restoring Compressed Images Before Entering the Image Encoder
2.1 Utilizing Existing General Models for Transfer Learning
The study refers to several models and techniques for improving the quality of compressed images:

JPEG Artifact Correction using Denoising Diffusion Restoration Models: This approach focuses on correcting artifacts introduced by JPEG compression.
Beyond a Gaussian Denoiser: This method involves using a deep convolutional neural network (DCNN) for image denoising, which goes beyond traditional Gaussian noise models.
SRGAN (Super-Resolution Generative Adversarial Network): This method employs a GAN for enhancing the resolution of images to a photo-realistic quality.
2.2 Building Custom Models Based on Current Datasets
For datasets like CIFAR-10, creating custom models that manually reconstruct images has been proposed. This could involve designing networks specifically tuned to the characteristics of the dataset and the type of compression used.

3. Addressing Image Noise and Scaling
Acknowledging that most image transmissions undergo compression which introduces noise, the paper proposes a combined approach of image denoising followed by scaling up to the required size (224*224). This process is treated as a super-resolution task, wherein both denoising and upscaling are handled to restore image quality effectively.

These strategies are designed to enhance the robustness of CLIP against the degradation caused by image compression, ensuring more reliable performance in practical applications where image quality may vary.





