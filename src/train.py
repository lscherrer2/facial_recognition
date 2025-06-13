from utils import FacesDataset, FacesDataLoader, DATASET_PATH
from model import Model, SiameseNet
import sys
import torch
from torch import Tensor
import os
from pathlib import Path
print("loaded dependencies")

# load or initialize model
model = (
    torch.load("model.weights", weights_only=False)
    if os.path.exists((Path(__file__).parent/"model.weights").__str__()) and "--new" not in sys.argv
    else SiameseNet(Model()).cuda()
)
if os.path.exists((Path(__file__).parent/"model.weights").__str__()) and "--new" not in sys.argv:
    print("model loaded")
else:
    print("model initialized")

# specify hyperparameters
EPOCHS = 1000
BATCH_SIZE = 32
INITIAL_LR = 0.001
loss_fn = torch.nn.MSELoss()
optim = torch.optim.Adam(model.parameters(), lr=INITIAL_LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=3)

# setup dataset-related utils
dataset = FacesDataset(
    dataset_path= DATASET_PATH,
    max_loaded = 1000,
    split = 0.8
)
dataloader = FacesDataLoader(
    dataset = dataset,
    batch_size=BATCH_SIZE,
    similar_ratio = 0.5,
)
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
        loss: Tensor = loss_fn(y_pred, y_true)
        
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
    torch.save(model, "./model.weights")

    # step the scheduler
    scheduler.step(epoch_avg_loss)    

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




