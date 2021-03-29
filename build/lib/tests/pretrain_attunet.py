from farabi.data.tbldata import XenopusData
from farabi.prep.transforms import Normalize, ToTensor
from torchvision import transforms
from farabi.data.datasets import XenopusDataset
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import torch
from farabi.models.segmentation.AttUnet import AttUnet
from farabi.utils.regul import EarlyStopping
import time
from collections import defaultdict
from tqdm import tqdm
from farabi.utils.losses import Losses
import os


# # Load dataset path
xeno_data = XenopusData()
pth_train_img = xeno_data.pth_ftrain_img
pth_train_lbl = xeno_data.pth_ftrain_lbl
pth_test_img = xeno_data.pth_ftest_img
pth_test_lbl = xeno_data.pth_ftest_lbl

# Necessary transforms
composed_train = transforms.Compose([Normalize(), ToTensor()])

transformed_train_data = XenopusDataset(
    img_dir=pth_train_img,
    transform=composed_train,
    lbl_dir=pth_train_lbl,
    mode='train')

random_seed = 42
batch_size = 4  # 16
validation_split = 0.2
shuffle_dataset = True

dataset_size = len(transformed_train_data)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))

if shuffle_dataset:
    np.random.seed(random_seed)
    np.random.shuffle(indices)

train_indices, val_indices = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(
    transformed_train_data, batch_size=batch_size, sampler=train_sampler, num_workers=40)

valid_loader = torch.utils.data.DataLoader(
    transformed_train_data, batch_size=batch_size, sampler=valid_sampler, num_workers=40)
# #########################################################################################

device = torch.device('cuda:0')
model = AttUnet(img_ch=3, out_ch=1)

# beta1 = 0.5
# beta2 = 0.999 [beta1, beta2]
learning_rate = 0.01
# learning_rate = random.random()*0.0005 + 0.0000005

optimizer = torch.optim.Adam(
    list(model.parameters()),
    learning_rate)

model.to(device)  # it was after optimizer
print("hi")

epochs = 100
patience = 10

early_stopping = EarlyStopping(patience=patience, verbose=True)

batch_train_losses = []
batch_valid_losses = []
epoch_train_losses = []
epoch_valid_losses = []

start_time = time.time()

# Training loop
for epoch in range(epochs):
    metrics = defaultdict(float)
    epoch_samples = 0
    val_epoch_samples = 0

    with tqdm(total=len(train_loader.dataset), desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:

        ###################
        # train the model #
        ###################
        model.train()

        for batch in train_loader:

            optimizer.zero_grad()

            imgs = batch['img']
            masks = batch['lbl']

            imgs = imgs.to(device=device, dtype=torch.float32)
            masks = masks.to(device=device, dtype=torch.float32)

            outputs = model(imgs)

            loss = Losses().calc_loss(outputs, masks, metrics)

            # Zero grad was here

            loss.backward()

            optimizer.step()

            epoch_samples += imgs.size(0)

            mean_metrics = {key: metrics[key] /
                            epoch_samples for key in metrics.keys()}

            mean_metrics['val_loss'] = "None"
            mean_metrics['mem_mb'] = torch.cuda.memory_allocated()/1024/1024

            pbar.update(imgs.shape[0])
            pbar.set_postfix(**mean_metrics)

            batch_train_losses.append(loss.item())

        ###################
        # eval the model #
        ###################
        model.eval()

        for batch in valid_loader:
            imgs = batch['img']
            masks = batch['lbl']

            imgs = imgs.to(device=device, dtype=torch.float32)
            masks = masks.to(device=device, dtype=torch.float32)

            outputs = model(imgs)

            loss = Losses().calc_loss(outputs, masks, metrics)

            batch_valid_losses.append(loss.item())

            pbar.update(imgs.shape[0])
            mean_metrics['val_loss'] = np.average(batch_valid_losses)

            pbar.set_postfix(**mean_metrics)

        batch_train_loss = np.average(batch_train_losses)
        batch_valid_loss = np.average(batch_valid_losses)
        epoch_train_losses.append(batch_train_loss)
        epoch_valid_losses.append(batch_valid_loss)

        batch_train_losses = []
        batch_valid_losses = []

        early_stopping(batch_valid_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

time_elapsed = time.time() - start_time
print(f'Training complete in {time_elapsed // 60}m {time_elapsed % 60: .2f}s')

# save model
model_fld = '/home/DATA_Lia/data_02/DATASET_SUZY/MODELS/xeno_att_unet'
model_name = os.path.join(model_fld, "AttUnet_201230_" + f"{batch_valid_loss:.4f}" + ".pt")
torch.save(model.state_dict(), model_name)