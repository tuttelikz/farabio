import os
from torchvision import transforms
from farabi.prep.transforms import Normalize, ToTensor
from farabi.data.datasets import XenopusDataset
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import torch
from farabi.models.segmentation.Unet import Unet
from farabi.models.segmentation.Config import get_config
import time
from farabi.utils.regul import EarlyStopping
from farabi.utils.losses import Losses


def get_loader(config, pth_train_img, pth_train_lbl):
    random_seed = config.random_seed
    batch_size = config.batch_size
    validation_split = config.val_split
    shuffle_dataset = config.shuffle_data
    workers = config.num_workers

    composed_train = transforms.Compose([Normalize(), ToTensor()])

    transformed_train_data = XenopusDataset(img_dir=pth_train_img,
                                            transform=composed_train,
                                            lbl_dir=pth_train_lbl,
                                            mode="train")

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
        transformed_train_data, batch_size=batch_size, sampler=train_sampler, num_workers=workers)

    valid_loader = torch.utils.data.DataLoader(
        transformed_train_data, batch_size=batch_size, sampler=valid_sampler, num_workers=workers)

    return (train_loader, valid_loader)


class Trainer(object):
    def __init__(self, config, data_loader):
        self.device = torch.device(config.device)
        if config.optim == 'adam':
            optim = torch.optim.Adam

        self.early_stopping = EarlyStopping(
            patience=config.patience, verbose=True)
        self.train_losses = []
        self.val_losses = []
        self.train_loader = data_loader[0]
        self.valid_loader = data_loader[-1]
        self.build_model()
        self.optimizer = optim(self.munet.parameters(),
                               lr=config.learning_rate)  # !

    def train(self, epoch):
        batch_tloss = 0
        self.munet.train()
        for iteration, batch in enumerate(self.train_loader):
            self.optimizer.zero_grad()

            imgs = batch['img']
            masks = batch['lbl']

            imgs = imgs.to(device=self.device, dtype=torch.float32)
            masks = masks.to(device=self.device, dtype=torch.float32)

            outputs = self.munet(imgs)

            train_loss = Losses().calc_loss(outputs, masks)
            batch_tloss += train_loss.item()
            train_loss.backward()
            self.optimizer.step()

            if (iteration % 100) == 0:
                print(
                    f"===> Epoch [{epoch}]({iteration}/{len(self.train_loader)}): Loss: {train_loss.item():.4f}")

        epoch_train_loss = round(batch_tloss / len(self.train_loader), 4)
        self.train_losses.append(epoch_train_loss)
        print(
            f"===> Epoch {epoch} Complete: Avg. Train Loss: {epoch_train_loss}")

    def test(self):
        batch_vloss = 0
        self.munet.eval()
        for batch in self.valid_loader:
            imgs = batch['img']
            masks = batch['lbl']

            imgs = imgs.to(device=self.device, dtype=torch.float32)
            masks = masks.to(device=self.device, dtype=torch.float32)

            outputs = self.munet(imgs)

            val_loss = Losses().calc_loss(outputs, masks)
            batch_vloss += val_loss.item()

        epoch_val_loss = round(batch_vloss / len(self.valid_loader), 4)
        self.val_losses.append(epoch_val_loss)
        print(f"===> Epoch {epoch} Valid Loss: {epoch_val_loss}")

        self.early_stopping(epoch_val_loss, self.munet)

        early_stop = False
        if self.early_stopping.early_stop:
            early_stop = True
        return early_stop

    def build_model(self):
        in_ch, out_ch = 3, 1
        self.munet = Unet(in_ch, out_ch)
        self.munet.to(self.device)


if __name__ == "__main__":
    pth_train = '/home/DATA_Lia/data_02/DATASET_SUZY/SM_XENOPUS/DATA/Dataset/Train_200909_all'
    def x(a): return os.path.join(pth_train, a)
    pth_train_img, pth_train_lbl = x('Image'), x('Label')

    config, unparsed = get_config()

    data_loader = get_loader(config, pth_train_img, pth_train_lbl)
    trnr = Trainer(config, data_loader)

    start_time = time.time()
    early_stop = False
    epochs = config.num_epochs
    for epoch in range(epochs):
        trnr.train(epoch)
        early_stop = trnr.test()

        if early_stop:
            print("Early stopping")
            break

    time_elapsed = time.time() - start_time
    print(
        f'Training complete in {time_elapsed // 60}m {time_elapsed % 60: .2f}s')
