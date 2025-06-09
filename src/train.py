from model import SiameseNet, Model
from torch import Tensor, nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import utils
import torch

EPOCHS = 750
BATCH_SIZE = 32
RATE_SIMILAR = 0.4
CUDA = True

dataloader = utils.FacesDataLoader(
    utils.FacesDataset(utils.DATASET_PATH, 1000, 0.8),
    batch_size=BATCH_SIZE,
    similar_ratio=RATE_SIMILAR,
)

network = torch.load("model.weights", weights_only=False)
loss_fn = nn.MSELoss()
optim = Adam(network.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optim, mode="min", factor=0.5, patience=3)

for epoch in range(EPOCHS):
    print(f"\nBeginning epoch {epoch}")
    epoch_loss: float = 0
    index = 0

    for f1s, f2s, s in dataloader:
        f1s: Tensor = f1s.cuda()
        f2s: Tensor = f2s.cuda()
        s = s.cuda()

        y_pred = network(f1s, f2s)
        y_true = s
        loss = loss_fn(y_pred, y_true)
        loss.backward()
        optim.step()
        optim.zero_grad()

        epoch_loss = (loss.item() + epoch_loss * index) / (index + 1)
        index += 1

    torch.save(network, "model.weights")
    print(f"Epoch Loss: {epoch_loss}")
    scheduler.step(epoch_loss)
    epoch_loss = 0.0
    index = 0
