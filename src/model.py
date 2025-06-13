import torch
from torch import Tensor
from torch import nn
from torchvision import models

class Model (nn.Module):
    def __init__ (self):
        super().__init__()

        # ReLU activation
        self.activ = nn.ReLU()

        # Resnet Backbone
        self.resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)

        # One Linear layer
        self.fc = nn.Linear(1000, 512)
    
    def forward (self, face: Tensor) -> Tensor:
        y = face

        # pass through the resnet
        y = self.resnet(y)
        y = self.activ(y)

        # pass through the linear
        y = self.fc(y)

        return y

class SiameseNet (nn.Module):
    def __init__ (self, model: Model):
        super().__init__()
        
        # absorb the passed-in model
        self.inner = model 

    def forward (self, face1: Tensor, face2: Tensor) -> Tensor:

        # embed each face
        emb1 = self.inner(face1)
        emb2 = self.inner(face2) 

        # determine and return the similarity
        sim = torch.cosine_similarity(emb1, emb2)
        return sim
