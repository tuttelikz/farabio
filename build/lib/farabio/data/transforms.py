from skimage.transform import resize
import numpy as np
import torch
import torchvision.transforms as tt
import torchvision.transforms.functional as tf
import torch.nn.functional as F
import random
from PIL import Image


_err_size = "img and mask size does not match"


class Normalize(object):
    """Normalizes between [0; 1]
    """

    def __call__(self, sample):
        fname = sample['fname']

        img = sample['img']
        img = img.astype(np.float32)
        img_norm = img / 255

        norm_sample = {'img': img_norm, 'fname': fname}

        if 'lbl' in sample:
            lbl = sample['lbl']
            lbl = lbl.astype(np.float32)
            lbl_norm = lbl / 255
            norm_sample['lbl'] = lbl_norm

        return norm_sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors
    """

    def __call__(self, sample):
        fname = sample['fname']

        img = sample['img']

        if len(img.shape) == 3:
            img = img.transpose((2, 0, 1))
        else:
            img = np.expand_dims(img, axis=0)  # lbl

        tensor_sample = {
            'img': torch.from_numpy(img).float(),
            'fname': fname
        }

        if 'lbl' in sample:
            lbl = sample['lbl']

            if len(lbl.shape) == 3:
                lbl = lbl.transpose((2, 0, 1))
            else:
                lbl = np.expand_dims(lbl, axis=0)

            tensor_sample['lbl'] = torch.from_numpy(lbl).float()

        return tensor_sample


class Rescale(object):
    """Rescale the image in a sample to a given size
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        img, lbl = sample['img'], sample['lbl']

        h, w = img.shape[:2]

        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = resize(img, (new_h, new_w, 3))
        lbl = resize(lbl, (new_h, new_w))

        return {'img': img, 'lbl': lbl}


class RandomCrop(object):
    """Crop randomly the image in a sample

    Parameters
    ----------
    output_size : (tuple or int)
        desired output size. If int, square crop is made
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        img, lbl = sample['img'], sample['lbl']

        h, w = img.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        img = img[
            top: top + new_h,
            left: left + new_w,
            :]

        return {'img': img, 'lbl': lbl}


class Datajit(object):
    """Applies color jitter to dataset
    """

    def __init__(self, br=0, cnt=0, sat=0, hue=0):
        self.br = br
        self.cnt = cnt
        self.sat = sat
        self.hue = hue

    def __call__(self, img, mask):
        assert img.size == mask.size, _err_size
        return tt.ColorJitter(brightness=self.br, contrast=self.cnt, saturation=self.sat, hue=self.hue)(img), mask


class Dataraf(object):
    """Applies random affine transformation to dataset
    """

    def __init__(self, deg=0, trn=(0, 0), sc=(1, 1), shr=0, flc=None):
        self.deg = deg
        self.trn = trn
        self.sc = sc
        self.shr = shr
        self.flc = flc

    def __call__(self, img, mask):
        assert img.size == mask.size, _err_size

        raf = tt.RandomAffine(degrees=self.deg, translate=self.trn,
                              scale=self.sc, shear=self.shr, fillcolor=self.flc)

        affine_params = raf.get_params(degrees=(-self.deg, self.deg),
                                       translate=(self.trn[0], self.trn[-1]),
                                       scale_ranges=(self.sc[0], self.sc[-1]),
                                       shears=(-self.shr, self.shr),
                                       img_size=list(img.size))

        return tf.affine(img, *affine_params), tf.affine(mask, *affine_params)

# from segment and classify


class ImgToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, img, mask):
        return tt.ToTensor()(img), tt.ToTensor()(mask)


class ImgNormalize(object):
    """
    Normalizes between [-1; 1]
    """

    def __call__(self, sample):
        img, lbl = sample['img'], sample['lbl']

        img = img.astype(np.float32)
        img_norm = img / 255  # b-n [0 and 1]

        return {'img': img_norm, 'lbl': lbl}


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


class Compose(object):
    def __init__(self, augs):
        self.augs = augs

    def __call__(self, img, mask):
        assert img.size == mask.size, _err_size
        for a in self.augs:
            img, mask = a(img, mask)
        return img, mask


def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


def imresize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size,
                          mode="nearest").squeeze(0)
    return image


def random_resize(images, min_size=288, max_size=448):
    new_size = random.sample(list(range(min_size, max_size + 1, 32)), 1)[0]
    images = F.interpolate(images, size=new_size, mode="nearest")
    return images


def horizontal_flip(images, targets):
    images = torch.flip(images, [-1])
    targets[:, 2] = 1 - targets[:, 2]
    return images, targets


def tensor2image(tensor):
    image = 127.5*(tensor[0].cpu().float().numpy() + 1.0)
    if image.shape[0] == 1:
        image = np.tile(image, (3, 1, 1))
    return image.astype(np.uint8)


def train_hr_transform(crop_size):
    return tt.Compose([
        tt.RandomCrop(crop_size),
        tt.ToTensor(),
    ])


def train_lr_transform(crop_size, upscale_factor):
    return tt.Compose([
        tt.ToPILImage(),
        tt.Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC),
        tt.ToTensor()
    ])


def display_transform():
    return tt.Compose([
        tt.ToPILImage(),
        tt.Resize(400),
        tt.CenterCrop(400),
        tt.ToTensor()
    ])
