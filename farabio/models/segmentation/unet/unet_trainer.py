import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import albumentations as A
from albumentations.pytorch import ToTensor
from farabio.core.convnettrainer import ConvnetTrainer
from farabio.models.segmentation.unet.unet import Unet
from farabio.utils.regul import EarlyStopping
from farabio.utils.losses import Losses
from farabio.utils.tensorboard import TensorBoard
from farabio.utils.helpers import makedirs, parallel_state_dict
import skimage
from skimage import io, transform, img_as_ubyte
from skimage.io import imsave
from torchsummary import summary


class UnetTrainer(ConvnetTrainer):
    """U-Net trainer class. Override with custom methods here.

    Parameters
    ----------
    ConvnetTrainer : BaseTrainer
        Inherits ConvnetTrainer class
    """

    def define_data_attr(self, *args):
        self._in_ch = self.config.in_ch
        self._out_ch = self.config.out_ch
        self._data_path = self.config.data_path
        self._in_shape = self.config.shape

    def define_model_attr(self, *args):
        self._semantic = self.config.semantic
        self._model_save_name = self.config.model_save_name
        self._model_save_dir = self.config.model_save_dir

    def define_train_attr(self, *args):
        self._epoch = self.config.start_epoch
        self._patience = self.config.patience
        self._early_stop = self.config.early_stop
        self._save_epoch = self.config.save_epoch
        self._has_eval = self.config.has_eval

        if self.config.optim == 'adam':
            self.optim = torch.optim.Adam

        if self._early_stop:
            self.early_stopping = EarlyStopping(
                patience=self._patience, verbose=True)

        makedirs(self._model_save_dir)

    def define_test_attr(self, *args):
        self._output_img_dir = self.config.output_img_dir
        self._output_mask_dir = self.config.output_mask_dir
        self._output_overlay_dir = self.config.output_overlay_dir
        self._model_load_dir = self.config.model_load_dir

    def define_log_attr(self, *args):
        self._use_visdom = self.config.use_visdom
        self._use_tensorboard = self.config.use_tensorboard

        if self._use_tensorboard:
            self.tb = TensorBoard(os.path.join(self._model_save_dir, "logs"))
        elif not self._use_tensorboard:
            self.tb = None

    def define_compute_attr(self, *args):
        self._cuda = self.config.cuda
        self._device = torch.device(self.config.device)
        self._num_gpu = self.config.num_gpu
        self._num_workers = self.config.num_workers
        self._data_parallel = self.config.data_parallel

    def define_misc_attr(self, *args):
        self._train_losses = []
        self._val_losses = []

    def get_trainloader(self):
        train_dataset = LoadDataSet(
            self._data_path,
            shape=self._in_shape,
            transform=get_train_transform(self._in_shape))

        split_ratio = 0.25
        train_size = int(
            np.round(train_dataset.__len__()*(1 - split_ratio), 0))
        valid_size = int(np.round(train_dataset.__len__()*split_ratio, 0))

        train_data, valid_data = random_split(
            train_dataset, [train_size, valid_size])

        self.train_loader = DataLoader(
            dataset=train_data, batch_size=10, shuffle=True, num_workers=self._num_workers)

        self.valid_loader = DataLoader(
            dataset=valid_data, batch_size=10, num_workers=self._num_workers)

    def get_testloader(self):
        self.test_loader = self.valid_loader

    def build_model(self):
        self.model = Unet(self._in_ch, self._out_ch)

        if self._cuda:
            self.model.to(self._device)

        self.optimizer = self.optim(self.model.parameters(),
                                    lr=self.config.learning_rate)

    def build_parallel_model(self):
        self.model = Unet(self._in_ch, self._out_ch)
        self.model = nn.DataParallel(self.model)
        self.model.to(self._device)
        self.optimizer = self.optim(list(self.model.parameters()),
                                    lr=self.config.learning_rate)

    def show_model_summary(self, *args):
        print(summary(self.model, [(self._in_ch,
                                    self._in_shape,
                                    self._in_shape)]))

    def load_model(self):
        self.model.load_state_dict(torch.load(self._model_load_dir))

    def load_parallel_model(self):
        state_dict = torch.load(self._model_load_dir)
        _par_state_dict = parallel_state_dict(state_dict)
        self.model.load_state_dict(_par_state_dict)

    def start_logger(self):
        if self._use_visdom:
            self.logger = None

    def on_train_epoch_start(self):
        self.batch_tloss = 0
        self.model.train()

        self.train_epoch_iter = enumerate(self.train_loader)

    def on_start_training_batch(self, args):
        self.iteration = args[0]
        self.batch = args[-1]

    def optimizer_zero_grad(self):
        self.optimizer.zero_grad()

    def loss_backward(self):
        self.train_loss.backward()

    def optimizer_step(self):
        self.optimizer.step()

    def training_step(self):
        self.optimizer_zero_grad()

        self.imgs = self.batch[0]
        self.masks = self.batch[1]

        if self._cuda:
            self.imgs = self.imgs.to(self._device, dtype=torch.float32)
            self.masks = self.masks.to(self._device, dtype=torch.float32)

        self.outputs = self.model(self.imgs)

        if self._semantic:
            self.train_loss = Losses().extract_loss(self.outputs, self.masks)
        elif not self._semantic:
            self.train_loss = Losses().calc_loss(self.outputs, self.masks)

        self.batch_tloss += self.train_loss.item()

        self.loss_backward()
        self.optimizer_step()

    def on_end_training_batch(self):
        if self._use_visdom:
            self.logger.log(
                images={
                    'imgs': self.imgs,
                    'masks': self.masks,
                    'outputs': self.outputs
                }
            )

        print(
            f"===> Epoch [{self._epoch}]({self.iteration}/{len(self.train_loader)}): Loss: {self.train_loss.item():.4f}")

    def on_train_epoch_end(self):
        epoch_train_loss = round(self.batch_tloss / len(self.train_loader), 4)
        self._train_losses.append(epoch_train_loss)
        print(
            f"===> Epoch {self._epoch} Complete: Avg. Train Loss: {epoch_train_loss}")

    def on_evaluate_epoch_start(self):
        self.batch_vloss = 0
        self.model.eval()
        self.valid_epoch_iter = self.valid_loader

    def evaluate_batch(self, args):
        self.batch = args

        imgs = self.batch[0]
        masks = self.batch[1]

        imgs = imgs.to(device=self._device, dtype=torch.float32)
        outputs = self.model(imgs)

        if self._semantic:
            masks = masks.to(device=self._device, dtype=torch.long)
            self.val_loss = Losses().extract_loss(outputs, masks, self._device)
        elif not self._semantic:
            masks = masks.to(device=self._device, dtype=torch.float32)
            self.val_loss = Losses().calc_loss(outputs, masks)

    def on_evaluate_batch_end(self):
        self.batch_vloss += self.val_loss.item()

    def on_evaluate_epoch_end(self):
        epoch_val_loss = round(self.batch_vloss / len(self.valid_loader), 4)
        self._val_losses.append(epoch_val_loss)

        print(f"===> Epoch {self._epoch} Valid Loss: {epoch_val_loss}")

        if self._use_tensorboard:
            self.tb.scalar_summary('val_loss', epoch_val_loss, self._epoch)

        if self._early_stop:
            self.early_stopping(epoch_val_loss, self.model,
                                self._model_save_dir)
            self.early_stop = self.early_stopping.early_stop

    def on_epoch_end(self):
        if self._epoch % self._save_epoch == 0:
            if self._data_parallel:
                self.save_parallel_model()
            elif not self._data_parallel:
                self.save_model()

        if self.early_stop:
            print("Early stopping")
            self._model_save_name = "unet_es.pt"

            if self._data_parallel:
                self.save_parallel_model()
            elif not self._data_parallel:
                self.save_model()

            self.stop_train()

    def save_model(self):
        torch.save(self.model.state_dict(), os.path.join(
            self._model_save_dir, self._model_save_name))

    def save_parallel_model(self):
        torch.save(self.model.module.state_dict(), os.path.join(
            self._model_save_dir, self._model_save_name))

    def on_test_start(self):
        self.model.eval()
        self.test_loop_iter = enumerate(self.test_loader)

    def test_step(self, args):
        self.cur_batch = args[0]
        self.imgs = args[-1][0]
        self.fname = args[-1][-1]

        if self._cuda:
            self.imgs = self.imgs.to(device=self._device, dtype=torch.float32)

        outputs = self.model(self.imgs)
        self.pred = torch.sigmoid(outputs)
        self.pred = (self.pred > 0.5).bool()

        self.generate_result_img()

    def generate_result_img(self, *args):
        """Generate image from batch: one by one
        """
        for i in range(self.test_loader.batch_size):
            img_fname = self.fname[i]

            in_img = format_image(self.imgs[i].cpu().numpy())
            out_img = format_mask(self.pred[i].cpu().numpy())

            imsave(os.path.join(self._output_img_dir, img_fname),
                   img_as_ubyte(in_img), check_contrast=False)
            imsave(os.path.join(self._output_mask_dir, img_fname),
                   img_as_ubyte(out_img), check_contrast=False)

    def on_end_test_batch(self):
        print(f"{self.cur_batch} / {len(self.test_loader)}")


def drug_color(self):
    drug_color = {
        1: (255, 0, 0),
        2: (255, 0, 255),
        3: (0, 0, 255),
        4: (0, 255, 255),
        5: (255, 255, 0),
        6: (0, 255, 0),  # 7th for white color artifact
    }
    return drug_color


def format_image(img):
    img = np.array(np.transpose(img, (1, 2, 0)))
    mean = np.array((0.485, 0.456, 0.406))
    std = np.array((0.229, 0.224, 0.225))
    img = std * img + mean
    img = img*255
    img = img.astype(np.uint8)
    return img


def format_mask(mask):
    mask = np.squeeze(np.transpose(mask, (1, 2, 0)))
    return mask


def get_train_transform(img_shape):
    """Albumentations transform
    """
    return A.Compose(
        [
            A.Resize(img_shape, img_shape),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            A.HorizontalFlip(p=0.25),
            A.VerticalFlip(p=0.25),
            ToTensor()
        ])


class LoadDataSet(Dataset):
    """Dataset loader
    """

    def __init__(self, path, shape, transform=None):
        self.path = path
        self.folders = os.listdir(path)
        self.transforms = get_train_transform(shape)
        self.shape = shape

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, idx):
        image_folder = os.path.join(self.path, self.folders[idx], 'images/')
        mask_folder = os.path.join(self.path, self.folders[idx], 'masks/')
        fname = os.listdir(image_folder)[0]
        image_path = os.path.join(image_folder, fname)

        img = io.imread(image_path)[:, :, :3].astype('float32')
        img = transform.resize(img, (self.shape, self.shape))

        mask = self.get_mask(mask_folder, self.shape,
                             self.shape).astype('float32')

        augmented = self.transforms(image=img, mask=mask)
        img = augmented['image']
        mask = augmented['mask']

        mask = mask[0].permute(2, 0, 1)
        return (img, mask, fname)

    def get_mask(self, mask_folder, IMG_HEIGHT, IMG_WIDTH):
        mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
        for mask_ in os.listdir(mask_folder):
            mask_ = io.imread(os.path.join(mask_folder, mask_))
            mask_ = transform.resize(mask_, (IMG_HEIGHT, IMG_WIDTH))
            mask_ = np.expand_dims(mask_, axis=-1)
            mask = np.maximum(mask, mask_)

        return mask
