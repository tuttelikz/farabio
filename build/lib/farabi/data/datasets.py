
import os
import glob
import skimage
import torch
from torch.utils.data import Dataset
import torchvision.transforms as tt
from skimage import io
from PIL import Image
from farabi.prep.transforms import imresize, pad_to_square, horizontal_flip
from farabi.utils.helpers import _sdir, _rjoin
import random
import numpy as np
from farabi.utils.helpers import is_image_file, calculate_valid_crop_size
from farabi.prep.transforms import train_hr_transform, train_lr_transform


class XenopusDataset(Dataset):
    def __init__(self, img_dir, lbl_dir=None, transform=None, mode="train", filtered=None):
        """Contructor for Xenopus Dataset

        Parameters
        ----------
        img_dir : str
            path to directory of images
        lbl_dir : str
            path to directory of labels (in training mode)
        transform : torchvision.transforms
            opt. transform to be applied on a sample
        mode : str
            train or test
        """

        self.mode = mode
        self.img_dir = img_dir
        self.img_fnames = sorted(os.listdir(self.img_dir))

        if filtered:
            self.img_fnames = sorted(os.listdir(self.img_dir))

        if mode == 'train':
            self.lbl_dir = lbl_dir
            self.lbl_fnames = sorted(os.listdir(self.lbl_dir))  # [:350016]  # 500

        self.transform = transform

    def __len__(self):
        return len(self.img_fnames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.img_dir, self.img_fnames[idx])
        img = skimage.io.imread(img_name)

        sample = {'img': img, 'fname': self.img_fnames[idx]}

        if self.mode == 'train':
            lbl_name = os.path.join(self.lbl_dir, self.lbl_fnames[idx])
            lbl = skimage.io.imread(lbl_name)
            sample['lbl'] = lbl

        if self.transform:
            sample = self.transform(sample)

        return sample


# class XenopusTestDataset(Dataset):
#     def __init__(self, img_dir, transform):
#         """
#         Args:
#             img_dir (string): Path to directory of images
#             lbl_dir (string): Path to directory of labels
#             transform (callable, optional): Optional transform to be applied on a sample
#         """

#         self.img_dir = img_dir
#         self.img_fnames = os.listdir(self.img_dir)  # 500

#         self.transform = transform

#     def __len__(self):
#         return len(self.img_fnames)

#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()

#         img_name = os.path.join(self.img_dir, self.img_fnames[idx])

#         img = skimage.io.imread(img_name)

#         sample = {'img': img, 'fname': self.img_fnames[idx]}

#         if self.transform:
#             sample = self.transform(sample)

#         return sample


class SkinDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, transform):
        """Contructor for Skin Dataset

        Parameters
        ----------
        lr_dir : str
            path to directory of lowres images
        hr_dir : str
            path to directory of highres images
        transform : torchvision.transforms
            opt. transform to be applied on a sample
        """

        self.lr_dir = lr_dir
        self.lr_fnames = os.listdir(self.lr_dir)  # [:350016]  # 500
        self.hr_dir = hr_dir
        self.hr_fnames = os.listdir(self.hr_dir)  # [:350016]  # 500

        self.transform = transform

    def __len__(self):
        return len(self.lr_fnames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        lr_name = os.path.join(self.lr_dir, self.lr_fnames[idx])
        hr_name = os.path.join(self.hr_dir, self.hr_fnames[idx])

        img_lr = skimage.io.imread(lr_name)
        img_hr = skimage.io.imread(hr_name)

        sample = {'img': img_lr, 'lbl': img_hr}

        if self.transform:
            sample = self.transform(sample)

        return sample


class ImgLabelDataset(Dataset):
    def __init__(self, img_dir, lbl_dir, transform=None, augs=None):
        """Custom Image-Label Dataset from directory
        """
        self.img_dir = img_dir
        self.lbl_dir = lbl_dir
        self.transform = transform
        self.augs = augs

    def __len__(self):
        return len(os.listdir(self.img_dir))

    def __getitem__(self, idx):
        img_name = _sdir(self.img_dir)[idx]
        lbl_name = _sdir(self.lbl_dir)[idx]

        img = Image.open(fp=_rjoin(self.img_dir, img_name), mode='r')
        lbl = Image.open(fp=_rjoin(self.lbl_dir, lbl_name))

        if self.augs is not None:
            img, lbl = self.augs(img, lbl)

        sample = {'img': img, 'lbl': lbl, 'fname': img_name}

        return sample


class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=416):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.img_size = img_size

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image as PyTorch tensor
        img = tt.ToTensor()(Image.open(img_path))
        # Pad to square resolution
        img, _ = pad_to_square(img, 0)
        # Resize
        img = imresize(img, self.img_size)

        return img_path, img

    def __len__(self):
        return len(self.files)


class ListDataset(Dataset):
    def __init__(self, list_path, img_size=416, augment=True, multiscale=True, normalized_labels=True):
        with open(list_path, "r") as file:
            self.img_files = file.readlines()

        self.label_files = [
            path.replace("images", "labels").replace(".png", ".txt").replace(
                ".Bmp", ".txt").replace(".bmp", ".txt").replace(".jpg", ".txt")
            for path in self.img_files
        ]
        self.img_size = img_size
        self.max_objects = 100
        self.augment = augment
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0

    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------

        img_path = self.img_files[index % len(self.img_files)].rstrip()

        # Extract image as PyTorch tensor
        img = tt.ToTensor()(Image.open(img_path).convert('RGB'))

        # Handle images with less than three channels
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))

        _, h, w = img.shape
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
        # Pad to square resolution
        img, pad = pad_to_square(img, 0)
        _, padded_h, padded_w = img.shape

        # ---------
        #  Label
        # ---------

        label_path = self.label_files[index % len(self.img_files)].rstrip()

        targets = None
        if os.path.exists(label_path):
            boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))
            # Extract coordinates for unpadded + unscaled image
            x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)
            y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)
            x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
            y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)
            # Adjust for added padding
            x1 += pad[0]
            y1 += pad[2]
            x2 += pad[1]
            y2 += pad[3]
            # Returns (x, y, w, h)
            boxes[:, 1] = ((x1 + x2) / 2) / padded_w
            boxes[:, 2] = ((y1 + y2) / 2) / padded_h
            boxes[:, 3] *= w_factor / padded_w
            boxes[:, 4] *= h_factor / padded_h

            targets = torch.zeros((len(boxes), 6))
            targets[:, 1:] = boxes

        # Apply augmentations
        if self.augment:
            if np.random.random() < 0.5:
                img, targets = horizontal_flip(img, targets)

        return img_path, img, targets

    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i

        try:
            targets = torch.cat(targets, 0)
        except RuntimeError as e_inst:
            targets = None  # No boxes for an image

        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(
                range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        imgs = torch.stack([imresize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        return paths, imgs, targets

    def __len__(self):
        return len(self.img_files)


class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        self.transform = tt.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = sorted(
            glob.glob(os.path.join(root, f'{mode}A') + '/*.*'))
        self.files_B = sorted(
            glob.glob(os.path.join(root, f'{mode}B') + '/*.*'))
            #glob.glob(os.path.join(root, '%s/B' % mode) + '/*.*'))

    def __getitem__(self, index):
        item_A = self.transform(Image.open(
            self.files_A[index % len(self.files_A)]))

        if self.unaligned:
            item_B = self.transform(Image.open(
                self.files_B[random.randint(0, len(self.files_B) - 1)]))
        else:
            item_B = self.transform(Image.open(
                self.files_B[index % len(self.files_B)]))

        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))


class TrainDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, crop_size, upscale_factor):
        super(TrainDatasetFromFolder, self).__init__()
        self.image_filenames = [os.path.join(dataset_dir, x)
                                for x in os.listdir(dataset_dir) if is_image_file(x)]
        crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        self.hr_transform = train_hr_transform(crop_size)
        self.lr_transform = train_lr_transform(crop_size, upscale_factor)

    def __getitem__(self, index):
        hr_image = self.hr_transform(Image.open(self.image_filenames[index]))
        lr_image = self.lr_transform(hr_image)
        return lr_image, hr_image

    def __len__(self):
        return len(self.image_filenames)


class ValDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(ValDatasetFromFolder, self).__init__()
        self.upscale_factor = upscale_factor
        self.image_filenames = [os.path.join(dataset_dir, x)
                                for x in os.listdir(dataset_dir) if is_image_file(x)]

    def __getitem__(self, index):
        hr_image = Image.open(self.image_filenames[index])
        w, h = hr_image.size
        crop_size = calculate_valid_crop_size(min(w, h), self.upscale_factor)
        lr_scale = tt.Resize(crop_size // self.upscale_factor,
                             interpolation=Image.BICUBIC)
        hr_scale = tt.Resize(crop_size, interpolation=Image.BICUBIC)
        hr_image = tt.CenterCrop(crop_size)(hr_image)
        lr_image = lr_scale(hr_image)
        hr_restore_img = hr_scale(lr_image)
        return tt.ToTensor()(lr_image), tt.ToTensor()(hr_restore_img), tt.ToTensor()(hr_image)

    def __len__(self):
        return len(self.image_filenames)


class TestDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(TestDatasetFromFolder, self).__init__()
        self.lr_path = dataset_dir + '/data/'
        self.hr_path = dataset_dir + '/target/'
        self.upscale_factor = upscale_factor
        self.lr_filenames = [os.path.join(self.lr_path, x)
                             for x in sorted(os.listdir(self.lr_path)) if is_image_file(x)]
        self.hr_filenames = [os.path.join(self.hr_path, x)
                             for x in sorted(os.listdir(self.hr_path)) if is_image_file(x)]

    def __getitem__(self, index):
        image_name = self.lr_filenames[index].split('/')[-1]
        lr_image = Image.open(self.lr_filenames[index])
        w, h = lr_image.size
        hr_image = Image.open(self.hr_filenames[index])
        hr_scale = tt.Resize(
            (self.upscale_factor * h, self.upscale_factor * w), interpolation=Image.BICUBIC)
        hr_restore_img = hr_scale(lr_image)
        return image_name, tt.ToTensor()(lr_image), tt.ToTensor()(hr_restore_img), tt.ToTensor()(hr_image)

    def __len__(self):
        return len(self.lr_filenames)
