import torch
import torchvision
print(torch.__version__)
print(torchvision.__version__)
print(torch.cuda.is_available())
import torch
print(torch.version.cuda)  # PyTorch'un desteklediği CUDA sürümü
print(torch.cuda.is_available())  # GPU kullanılabilir mi kontrol et
