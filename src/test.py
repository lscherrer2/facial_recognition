from utils import FacesDataset, FacesDataLoader, DATASET_PATH
from model import Model, SiameseNet
import sys
import torch
from torch import Tensor
import os
from pathlib import Path
import json


model_path = Path(__file__).parent/"model.weights".__str__()
model = torch.load("model.weights", weights_only=False)

dataloader = FacesDataLoader(
    dataset = FacesDataset(
        dataset_path= DATASET_PATH,
        max_loaded = 5,
        split = 0.8
    ),
    batch_size=1,
    similar_ratio = 0.5,
)

for i, (f1, f2, s) in enumerate(dataloader):

    f1 = f1.cuda()
    f2 = f2.cuda()
    s = s.cuda()

    s_pred: Tensor = model(f1, f2)
    print(f"predicted s: {s_pred.item()}")
    print(f"true s: {s.item()}")



