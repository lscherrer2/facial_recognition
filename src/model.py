import torch
from torch import Tensor
from torch import nn
from torchvision import models
from torch.optim import Adam

class Model (nn.Module):
    def __init__ (s):
        super().__init__()
        s.activ = nn.ReLU()
        s.resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        s.fc = nn.Linear(1000, 512)
    
    def forward (s, face: Tensor) -> Tensor:
        y = face
        y = s.resnet(y)
        y = s.activ(y)
        y = s.fc(y)
        return y

class SiameseNet (nn.Module):
    def __init__ (s, model: Model):
        super().__init__()
        s.inner = model 

    def forward (s, face1: Tensor, face2: Tensor) -> Tensor:
        emb1 = s.inner(face1)
        emb2 = s.inner(face2) 
        sim = torch.cosine_similarity(emb1, emb2)
        return sim
