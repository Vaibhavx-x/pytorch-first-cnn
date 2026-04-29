# Custom ResNet-18 Image Classifier (CIFAR-10)

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

## 📌 Project Overview
This repository contains a highly optimized, production-ready Computer Vision pipeline trained on the CIFAR-10 dataset. By surgically modifying a pre-trained ResNet-18 architecture and utilizing reactive training techniques, this model achieved a **91.7% test accuracy** (approaching the dataset's architectural limit for ResNet).

In addition to the training pipeline, this project includes a standalone inference script to run the trained model on real-world images downloaded from the web.

## 🧠 Architectural Engineering
Standard ResNet models are designed for large ImageNet images (224x224). Feeding tiny 32x32 CIFAR images into a standard ResNet causes the initial layers to mathematically crush the spatial dimensions, destroying feature visibility. 

To solve this, I modified the ResNet-18 backbone:
* **Surgical Layer Replacement:** Replaced the initial `7x7` stride-2 convolution with a `3x3` stride-1 kernel (`padding=1`) to preserve the 32x32 spatial resolution.
* **Downsampling Removal:** Replaced the initial `MaxPool2d` layer with an `nn.Identity()` layer to prevent premature resolution loss.
* **Classifier Adaptation:** Adjusted the final fully connected (`fc`) layer to output exactly 10 classes.

## 🚀 Advanced Training Techniques
To break through the 90% accuracy plateau and reach **91.7%**, the training loop implements several advanced Deep Learning optimizations:
* **Reactive Learning Rate Scheduling:** Replaced static step schedulers with `ReduceLROnPlateau` using Stochastic Gradient Descent (SGD). By actively monitoring validation metrics, the pipeline detects local minima and dynamically drops the learning rate (e.g., from 0.01 to 0.001) to find deeper, more optimal weights.
* **Automatic Mixed Precision (AMP):** Utilized PyTorch `autocast` and `GradScaler` to train using FP16. This drastically reduced VRAM usage and doubled the training speed without sacrificing accuracy.
* **Advanced Data Augmentation:** Beyond standard flips and crops, the pipeline utilizes `RandomErasing` to force the network to learn holistic object features rather than relying on isolated visual shortcuts.
* **Label Smoothing:** Applied to the `CrossEntropyLoss` criterion to prevent the model from becoming overconfident, reducing overfitting.
* **Ruthless Checkpointing:** The script evaluates the validation set post-epoch and only saves the `.pth` weights if a new high score is achieved.

## 📂 Repository Structure
```text
/pytorch-first-cnn
├── resnet.py          # Modified ResNet-18 architecture class
├── dataset.py         # Data loaders and advanced augmentation pipelines
├── train.py           # The main training loop with AMP and Checkpointing
├── inference.py       # Standalone script to test custom images
└── README.md