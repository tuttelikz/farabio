import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from farabio.data.biodatasets import EpiSeizureDataset
from farabio.models.sequence import VanillaLSTM


csv_file = "/home/data/01_SSD4TB/suzy/datasets/public-datasets/epileptic-seizure-recognition/Epileptic Seizure Recognition.csv"
save_path = "/home/data/01_SSD4TB/suzy/datasets/public-datasets"

train_dataset = EpiSeizureDataset(
    save_path, download=True, mode="train")
train_loader = DataLoader(train_dataset, batch_size=4)

test_dataset = EpiSeizureDataset(
    csv_file, download=False, mode="test")
test_loader = DataLoader(test_dataset, batch_size=4)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = VanillaLSTM().to(device)

epochs = 15
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(epochs):
    print(f"Epoch:{epoch}")
    model.train()
    running_train_loss = 0.0
    for idx, batch in enumerate(train_loader):
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        output = model(x)

        loss = criterion(output, y)

        loss.backward()
        optimizer.step()
        running_train_loss += loss.item()

    print(f"Train loss: {running_train_loss / (idx+1):.3f}")

    running_valid_loss = 0.0
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(train_loader):
            x, y = batch
            x = x.to(device)
            y = y.to(device)

            output = model(x)

            loss = criterion(y, output)

            running_valid_loss += loss.item()

    print(f"Valid loss: {running_valid_loss / (idx+1):.3f}")

print("Training finished!")
