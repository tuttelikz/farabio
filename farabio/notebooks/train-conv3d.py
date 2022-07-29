
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from farabio.data.biodatasets import MosmedDataset
from farabio.models.volume.convnet import ConvNet3D, ConvBlock3D

# CT scan dataset
train_dataset = MosmedDataset(download=False,  train=True)
valid_dataset = MosmedDataset(download=False,  train=False)

train_loader = DataLoader(train_dataset, batch_size=4,
                          shuffle=True, num_workers=0)
valid_loader = DataLoader(valid_dataset, batch_size=4,
                          shuffle=True, num_workers=0)


# Model
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model = ConvNet3D(ConvBlock3D, 1).to(device)

# Training loop
epochs = 15
lr = 0.0001
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
log_step = 1
decayRate = 0.96
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
    optimizer=optimizer, gamma=decayRate)

for epoch in range(epochs):
    model.train()
    train_running_loss = 0.0
    valid_running_loss = 0.0
    batch_count = 0
    lr_scheduler.step()

    print(f"Epoch: {epoch+1}")

    for i, batch in enumerate(train_loader):
        img, label = batch[0], batch[-1]
        img, label = img.to(device), label.to(device)
        output = model(img)

        optimizer.zero_grad()

        loss = criterion(output, label)

        loss.backward()
        optimizer.step()
        train_running_loss += loss.item()

        if i % log_step == 0:
            print(f"Batch: {i}, train loss: {train_running_loss / log_step}")
            train_running_loss = 0.0

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(valid_loader):
            img, label = batch[0], batch[-1]
            img, label = img.to(device), label.to(device)
            output = model(img)
            loss = criterion(output, label)
            valid_running_loss += loss.item()

            if i % log_step == 0:
                print(
                    f"Batch: {i}, train loss: {valid_running_loss / log_step}")
                valid_running_loss = 0.0

print("Training finished")
