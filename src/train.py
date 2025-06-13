from utils import FacesDataset, FacesDataLoader, DATASET_PATH
from model import Model, SiameseNet
import sys
import torch
from torch import Tensor
import os
from pathlib import Path
import json
print("loaded dependencies")

model_path = Path(__file__).parent/"model.weights".__str__()
hp_path = Path(__file__).parent.parent/"hp.json".__str__()
load = True if os.path.exists(model_path) and "--new" not in sys.argv else False
if load:
    with open(hp_path, "r") as f:
        lr = json.load(f)["lr"]
else:
    lr = 0.001

# load or initialize model
model = (
    torch.load("model.weights", weights_only=False)
    if load
    else SiameseNet(Model()).cuda()
)
if load:
    print("model loaded")
else:
    print("model initialized")

# specify hyperparameters
EPOCHS = 1000
BATCH_SIZE = 64
INITIAL_LR = lr
loss_fn = torch.nn.MSELoss()
optim = torch.optim.Adam(model.parameters(), lr=INITIAL_LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=2)

# setup dataset-related utils
dataset = FacesDataset(
    dataset_path= DATASET_PATH,
    max_loaded = 350,
    split = 0.8
)
dataloader = FacesDataLoader(
    dataset = dataset,
    batch_size=BATCH_SIZE,
    similar_ratio = 0.5,
)
relu = torch.nn.ReLU()

print("dataset loaded")
print("beginning training")
# training loop
for epoch in range(EPOCHS):

    # initialize loss metric
    epoch_avg_loss = 0
    batch_idx = 0

    # run the batch loop
    for faces1, faces2, similarities in dataloader.train():

        # add type hints for tensors
        faces1: Tensor
        faces2: Tensor
        similarities: Tensor

        # move tensors to cuda
        faces1 = faces1.cuda()
        faces2 = faces2.cuda()
        similarities = similarities.cuda()

        # 1) forward pass
        y_pred: Tensor = model(faces1, faces2)
        y_true: Tensor = similarities

        # 2) compute loss
        loss: Tensor = relu(loss_fn(y_pred, y_true))
        
        # 3) backward pass
        loss.backward()

        # 4) step optimizer
        optim.step()

        # 5) zero gradients
        optim.zero_grad()
 
        # update loss values for epoch
        epoch_avg_loss = (batch_idx * epoch_avg_loss + loss.item()) / (batch_idx + 1)
        batch_idx += 1

    # save the network
    torch.save(model, model_path)

    # step the scheduler
    scheduler.step(epoch_avg_loss)    

    # save learning rate
    with open(hp_path, "w") as f:
        json.dump({"lr": optim.param_groups[0]["lr"]}, f)

    # set up loss tracker
    eval_avg_loss = 0
    avg_pct_correct = 0
    batch_idx = 0

    # evaluate the model at each epoch
    for faces1, faces2, similarities in dataloader.eval():

        # add type hints for tensors
        faces1: Tensor
        faces2: Tensor
        similarities: Tensor

        # move tensors to cuda
        faces1 = faces1.cuda()
        faces2 = faces2.cuda()
        similarities = similarities.cuda()

        # 1) forward pass
        y_pred: Tensor = model(faces1, faces2)
        y_true: Tensor = similarities

        # 2) compute loss
        loss: Tensor = loss_fn(y_pred, y_true)
 
        # update loss values for epoch
        eval_avg_loss = (batch_idx * eval_avg_loss + loss.item()) / (batch_idx + 1)
        batch_idx += 1

        # snap a smooth guess to either yes or no
        true_values = torch.round(similarities)
        guesses = torch.round(y_pred)

        # determine how many are correct
        correct = 0
        total = true_values.numel()
        for i in range(total):
            if true_values[i] == guesses[i]:
                correct += 1

        # calculate some stats
        pct_correct = (correct / total)
        avg_pct_correct = (avg_pct_correct * batch_idx + pct_correct) / (batch_idx + 1)
        batch_idx += 1

    # print status
    print(f"epoch {epoch}, train_loss (avg): {epoch_avg_loss}, eval_loss (avg) {eval_avg_loss}, pct_correct: {avg_pct_correct}")





