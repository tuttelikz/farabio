import argparse
import torch
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import SubsetRandomSampler
import os
from skimage.io import imread
from farabi.utils.regul import EarlyStopping
import torch.nn as nn
import time
from collections import defaultdict
from tqdm import tqdm
from farabi.utils.helpers import makedirs
from attunet import AttUnet

# class conv_block(nn.Module):
#     def __init__(self, ch_in, ch_out):
#         super(conv_block, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(ch_in, ch_out, kernel_size=3,
#                       stride=1, padding=1, bias=True),
#             nn.BatchNorm2d(ch_out),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(ch_out, ch_out, kernel_size=3,
#                       stride=1, padding=1, bias=True),
#             nn.BatchNorm2d(ch_out),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         x = self.conv(x)
#         return x


# class up_conv(nn.Module):
#     def __init__(self, ch_in, ch_out):
#         super(up_conv, self).__init__()
#         self.up = nn.Sequential(
#             nn.Upsample(scale_factor=2),
#             nn.Conv2d(ch_in, ch_out, kernel_size=3,
#                       stride=1, padding=1, bias=True),
#             nn.BatchNorm2d(ch_out),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         x = self.up(x)
#         return x


# class Attention_block(nn.Module):
#     def __init__(self, F_g, F_l, F_int):
#         super(Attention_block, self).__init__()
#         self.W_g = nn.Sequential(
#             nn.Conv2d(F_g, F_int, kernel_size=1,
#                       stride=1, padding=0, bias=True),
#             nn.BatchNorm2d(F_int)
#         )

#         self.W_x = nn.Sequential(
#             nn.Conv2d(F_l, F_int, kernel_size=1,
#                       stride=1, padding=0, bias=True),
#             nn.BatchNorm2d(F_int)
#         )

#         self.psi = nn.Sequential(
#             nn.Conv2d(
#                 F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
#             nn.BatchNorm2d(1),
#             nn.Sigmoid()
#         )

#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, g, x):
#         g1 = self.W_g(g)
#         x1 = self.W_x(x)
#         psi = self.relu(g1+x1)
#         psi = self.psi(psi)

#         return x*psi


# class AttU_Net(nn.Module):
#     def __init__(self, img_ch=3, output_ch=1):
#         super(AttU_Net, self).__init__()

#         self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

#         self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
#         self.Conv2 = conv_block(ch_in=64, ch_out=128)
#         self.Conv3 = conv_block(ch_in=128, ch_out=256)
#         self.Conv4 = conv_block(ch_in=256, ch_out=512)
#         self.Conv5 = conv_block(ch_in=512, ch_out=1024)

#         self.Up5 = up_conv(ch_in=1024, ch_out=512)
#         self.Att5 = Attention_block(F_g=512, F_l=512, F_int=256)
#         self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

#         self.Up4 = up_conv(ch_in=512, ch_out=256)
#         self.Att4 = Attention_block(F_g=256, F_l=256, F_int=128)
#         self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

#         self.Up3 = up_conv(ch_in=256, ch_out=128)
#         self.Att3 = Attention_block(F_g=128, F_l=128, F_int=64)
#         self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

#         self.Up2 = up_conv(ch_in=128, ch_out=64)
#         self.Att2 = Attention_block(F_g=64, F_l=64, F_int=32)
#         self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

#         self.Conv_1x1 = nn.Conv2d(
#             64, output_ch, kernel_size=1, stride=1, padding=0)

#     def forward(self, x):
#         # encoding path
#         x1 = self.Conv1(x)

#         x2 = self.Maxpool(x1)
#         x2 = self.Conv2(x2)

#         x3 = self.Maxpool(x2)
#         x3 = self.Conv3(x3)

#         x4 = self.Maxpool(x3)
#         x4 = self.Conv4(x4)

#         x5 = self.Maxpool(x4)
#         x5 = self.Conv5(x5)

#         # decoding + concat path
#         d5 = self.Up5(x5)
#         x4 = self.Att5(g=d5, x=x4)
#         d5 = torch.cat((x4, d5), dim=1)
#         d5 = self.Up_conv5(d5)

#         d4 = self.Up4(d5)
#         x3 = self.Att4(g=d4, x=x3)
#         d4 = torch.cat((x3, d4), dim=1)
#         d4 = self.Up_conv4(d4)

#         d3 = self.Up3(d4)
#         x2 = self.Att3(g=d3, x=x2)
#         d3 = torch.cat((x2, d3), dim=1)
#         d3 = self.Up_conv3(d3)

#         d2 = self.Up2(d3)
#         x1 = self.Att2(g=d2, x=x1)
#         d2 = torch.cat((x1, d2), dim=1)
#         d2 = self.Up_conv2(d2)

#         d1 = self.Conv_1x1(d2)

#         return d1


class XenopusDataset(Dataset):
    def __init__(self, img_dir, lbl_dir, transform):
        """
        Args:
            img_dir (string): Path to directory of images
            lbl_dir (string): Path to directory of labels
            transform (callable, optional): Optional transform to be applied on a sample
        """

        self.img_dir = img_dir
        self.img_fnames = os.listdir(self.img_dir)  # [:350016]  # 500
        self.lbl_dir = lbl_dir
        self.lbl_fnames = os.listdir(self.lbl_dir)  # [:350016]  # 500

        self.transform = transform

    def __len__(self):
        return len(self.img_fnames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.img_dir, self.img_fnames[idx])
        lbl_name = os.path.join(self.lbl_dir, self.lbl_fnames[idx])

        img = imread(img_name)
        lbl = imread(lbl_name)

        sample = {'img': img, 'lbl': lbl}

        if self.transform:
            sample = self.transform(sample)

        return sample


parser = argparse.ArgumentParser(description="Train Att_Unet")
parser.add_argument("--num_epochs", default=100,
                    type=int, help='train epoch number')
parser.add_argument("--patience", default=10, type=int,
                    help='patience for early stop')
parser.add_argument("--batch_size", default=4, type=int,
                    help='batch size for train')
parser.add_argument("--device", default=0, type=int,
                    help='batch size for train')
parser.add_argument("--learning_rate", default=0.01,
                    type=float, help='learning rate type')


class ToLongTensor(object):
    def __call__(self, sample):
        img, lbl = sample['img'], sample['lbl']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W

        img = img.transpose((2, 0, 1))
        # print(lbl.shape)
        # lbl = lbl.transpose((2, 0, 1))

        return {'img': torch.from_numpy(img),
                'lbl': torch.from_numpy(lbl).type(torch.LongTensor)}


class Normalize(object):
    """
    Normalizes between [-1; 1]
    """

    def __call__(self, sample):
        img, lbl = sample['img'], sample['lbl']

        img = img.astype(np.float32)
        img_norm = img / 255  # b-n [0 and 1]

        return {'img': img_norm, 'lbl': lbl}


if __name__ == '__main__':
    opt = parser.parse_args()
    epochs = opt.num_epochs
    patience = opt.patience
    batch_size = opt.batch_size

    # if opt.device == 0:
    #     device = torch.device('cuda:0')
    # elif opt.device == 1:
    #     device = torch.device('cuda:1')
    # else:
    #     device = torch.device('cuda:2')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    learning_rate = opt.learning_rate

    root_dir = "/home/DATA_Lia/data_02/DATASET_SUZY/SM_XENOPUS/DATA/Dataset/Train_200909_all/"
    pth_train_img = os.path.join(root_dir, "Image-sanmo-bal-aug")
    pth_train_lbl = os.path.join(root_dir, "Mask-sanmo-bal-aug-class")

    #pth_train_img = os.path.join(root_dir, "Image-bal")
    #pth_train_lbl = os.path.join(root_dir, "Maskclass-bal")

    composed_train = transforms.Compose([Normalize(), ToLongTensor()])

    transformed_train_data = XenopusDataset(
        img_dir=pth_train_img,
        transform=composed_train,
        lbl_dir=pth_train_lbl)

    random_seed = 42
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

    model = AttUnet(3, 7)

    print(torch.cuda.device_count())

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # how to get with optimize
    optimizer = torch.optim.Adam(
        list(model.parameters()),
        learning_rate)

    model.to(device)  # it was after optimizer
    print("start")

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
            model.train()

            for batch in train_loader:

                optimizer.zero_grad()

                imgs = batch['img']
                masks = batch['lbl']

                imgs = imgs.to(device=device, dtype=torch.float32)
                masks = masks.to(device=device, dtype=torch.long)
                outputs = model(imgs)

                loss = torch.nn.CrossEntropyLoss()
                my_loss = loss(outputs, masks)

                my_loss.backward()

                optimizer.step()

                epoch_samples += imgs.size(0)

                mean_metrics = {}
                mean_metrics['val_loss'] = "None"

                pbar.update(imgs.shape[0])
                pbar.set_postfix(**mean_metrics)

                batch_train_losses.append(my_loss.item())

            model.eval()

            for batch in valid_loader:
                imgs = batch['img']
                masks = batch['lbl']

                imgs = imgs.to(device=device, dtype=torch.float32)
                masks = masks.to(device=device, dtype=torch.long)

                outputs = model(imgs)

                loss = torch.nn.CrossEntropyLoss()
                my_loss = loss(outputs, masks)

                batch_valid_losses.append(my_loss.item())

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
    print(
        f'Training complete in {time_elapsed // 60}m {time_elapsed % 60: .2f}s')

    date = time.strftime("%y%m%d", time.localtime())
    model_dir = '/home/DATA_Lia/data_02/DATASET_SUZY/MODELS/segmentation'
    model_save_dir = os.path.join(model_dir, "semattunet_" + date)
    makedirs(model_save_dir)

    model_name = 'attunet_ce_' + f"{batch_valid_loss:.4f}" + '.pt'

    # if torch.cuda.device_count() > 1:
    #     torch.save(model.module.state_dict(),
    #                os.path.join(model_save_dir, model_name))
    # else:
    torch.save(model.state_dict(),
               os.path.join(model_save_dir, model_name))

    with open(os.path.join(model_save_dir, 'epoch_train_losses.txt'), 'w') as f:
        for item in epoch_train_losses:
            item = round(item, 4)
            f.write("%s\n" % item)

    with open(os.path.join(model_save_dir, 'epoch_valid_losses.txt'), 'w') as f:
        for item in epoch_valid_losses:
            item = round(item, 4)
            f.write("%s\n" % item)
