import os
import torch
import torch.nn as nn
from farabio.utils.regul import EarlyStopping
from farabio.utils.loggers import TensorBoard, vi
from farabio.utils.losses import Losses
from farabio.models.segmentation.unet.unet import Unet
from farabio.models.segmentation.attunet.attunet import AttUnet
from farabio.prep.imgops import Imgops
from farabio.utils.helpers import makedirs
import torchvision.utils as vutils
import numpy as np
import skimage
from skimage.measure import *
from skimage.io import imsave
from skimage import img_as_ubyte
from farabio.utils.helpers import equal_colors


class Trainer(object):
    """[summary]

    Methods
    -------
    __init__(self, config, data_loader, model)
        Constructor for Unet class
    forward(self, X):
        Forward propagation
    """

    def __init__(self, config, data_loader=None, mode='train'):
        """Constructor for Unet class

        Parameters
        ------------
        config : object
            configurations
        data_loader : torch.utils.data.dataloader
            dataloader
        model : nn.Module
            segmentation model
        """
        if config.optim == 'adam':
            optim = torch.optim.Adam

        self.early_stopping = EarlyStopping(
            patience=config.patience, verbose=True)
        self.train_losses = []
        self.val_losses = []

        self.early_stop = False
        self.save_epoch = config.save_epoch
        self.start_epoch = 0
        self.num_epochs = config.num_epochs
        self.semantic = config.semantic
        self.in_ch = config.in_ch
        self.out_ch = config.out_ch
        self.model_save_dir = os.path.join(config.model_save_dir)
        self.data_parallel = config.data_parallel
        self.output_img_dir = config.output_img_dir
        self.output_mask_dir = config.output_mask_dir
        self.output_overlay_dir = config.output_overlay_dir
        self.build_model()
        self.use_visdom = config.use_visdom

        if mode == 'train':
            self.train_loader = data_loader[0]
            self.valid_loader = data_loader[-1]

        makedirs(self.model_save_dir)

        if config.use_tensorboard:
            self.tb = TensorBoard(os.path.join(self.model_save_dir, "logs"))
        else:
            self.tb = None
        self.epoch = 0
        self.num_gpu = config.num_gpu

        self.device = torch.device(config.device)

        if self.data_parallel is False:
            self.model.to(self.device)
        elif self.data_parallel is True:
            # self.model = nn.DataParallel(self.model.cuda())
            self.model = nn.DataParallel(self.model)
            self.model.to(self.device)

        if mode == 'test':
            self.test_loader = data_loader
            self.load_model(config.model_load_dir)

        if self.data_parallel is False:
            self.optimizer = optim(self.model.parameters(),
                                   lr=config.learning_rate)
        elif self.data_parallel is True:
            self.optimizer = optim(list(self.model.parameters()),
                                   lr=config.learning_rate)

    def train(self):
        """Training function

        Parameters
        ----------
        epoch : int
            current epoch
        """
        unet_visdom = UnetViz(self.num_epochs, len(self.train_loader))

        for self.epoch in range(self.start_epoch, self.num_epochs):
            batch_tloss = 0
            self.model.train()

            for iteration, batch in enumerate(self.train_loader):
                self.optimizer.zero_grad()

                imgs = batch['img']
                masks = batch['lbl']

                imgs = imgs.to(device=self.device, dtype=torch.float32)

                # if self.num_gpu == 0:
                #     imgs = imgs.to('cuda', dtype=torch.float32)
                #     masks = masks.to('cuda', dtype=torch.float32)
                # elif self.num_gpu == 1:
                #     imgs = imgs.to(device=self.device, dtype=torch.float32)
                #     masks = masks.to(device=self.device, dtype=torch.float32)

                if self.semantic is True:
                    #masks = masks.to('cuda', dtype=torch.long)
                    masks = masks.to(device=self.device, dtype=torch.long)
                else:
                    masks = masks.to(device=self.device, dtype=torch.float32)

                outputs = self.model(imgs)

                # if self.semantic is True:
                #     loss = torch.nn.CrossEntropyLoss()
                #     train_loss = loss(outputs, masks)
                # else:
                #     train_loss = Losses().calc_loss(outputs, masks)

                if self.semantic is True:
                    train_loss = Losses().extract_loss(outputs, masks, self.device)
                else:
                    train_loss = Losses().calc_loss(outputs, masks)

                batch_tloss += train_loss.item()
                train_loss.backward()
                self.optimizer.step()

                if self.use_visdom is True:
                    unet_visdom.log(
                        images={'imgs': imgs, 'masks': masks, 'outputs': outputs})

                if (iteration % 100) == 0:
                    print(
                        f"===> Epoch [{self.epoch}]({iteration}/{len(self.train_loader)}): Loss: {train_loss.item():.4f}")

                    # if self.use_visdom is True:
                    #     unet_visdom.log({'train_loss': train_loss.item()}, images={
                    #                     'imgs': imgs, 'masks': masks, 'outputs': outputs})

            epoch_train_loss = round(batch_tloss / len(self.train_loader), 4)
            self.train_losses.append(epoch_train_loss)
            print(
                f"===> Epoch {self.epoch} Complete: Avg. Train Loss: {epoch_train_loss}")

            self.evaluate()
            if self.early_stop is True:
                print("Early stopping")
                self.save_model()
                break

            if self.epoch % self.save_epoch == 0:
                self.save_model()

    def evaluate(self):
        """Test function

        Parameters
        ----------
        epoch : int
            current epoch
        """
        batch_vloss = 0
        self.model.eval()

        for batch in self.valid_loader:
            imgs = batch['img']
            masks = batch['lbl']

            # if self.num_gpu == 0:
            #     imgs = imgs.to('cuda', dtype=torch.float32)
            #     masks = masks.to('cuda', dtype=torch.float32)
            # elif self.num_gpu == 1:
            #     imgs = imgs.to(device=self.device, dtype=torch.float32)
            #     masks = masks.to(device=self.device, dtype=torch.float32)

            imgs = imgs.to(device=self.device, dtype=torch.float32)

            if self.semantic is True:
                masks = masks.to(device=self.device, dtype=torch.long)
                # masks = masks.to('cuda', dtype=torch.long)
            else:
                masks = masks.to(device=self.device, dtype=torch.float32)

            outputs = self.model(imgs)  # calc loss

            # if self.semantic is True:
            #     loss = torch.nn.CrossEntropyLoss()
            #     val_loss = loss(outputs, masks)
            # else:
            #     val_loss = Losses().calc_loss(outputs, masks)

            if self.semantic is True:
                val_loss = Losses().extract_loss(outputs, masks, self.device)
            else:
                val_loss = Losses().calc_loss(outputs, masks)

            batch_vloss += val_loss.item()

        epoch_val_loss = round(batch_vloss / len(self.valid_loader), 4)
        self.val_losses.append(epoch_val_loss)
        print(f"===> Epoch {self.epoch} Valid Loss: {epoch_val_loss}")

        self.tb.scalar_summary('val_loss', epoch_val_loss, self.epoch)
        self.early_stopping(epoch_val_loss, self.model, self.model_save_dir)

        self.early_stop = self.early_stopping.early_stop

    def build_model(self):
        """Build model

        Parameters
        ----------
        epoch : int
            current epoch
        """

        self.model = Unet(self.in_ch, self.out_ch)

    def save_model(self):
        """Save model

        Parameters
        ----------
        epoch : int
            current epoch
        """

        model_name = os.path.join(self.model_save_dir, "semunet.pt")

        if self.data_parallel is False:
            torch.save(self.model.state_dict(), model_name)
        elif self.data_parallel is True:
            torch.save(self.model.module.state_dict(), model_name)

        # if self.num_gpu == 1:
        #     torch.save(self.model.state_dict(), model_name)
        # elif self.num_gpu > 1:
        #     torch.save(self.model.module.state_dict(), model_name)

    def test(self):
        # clrs = equal_colors(self.out_ch - 1, int_flag=True)

        # drug_color = {}
        # # color coding
        # for ijk, clr in enumerate(clrs):
        #     drug_color[ijk+1] = clr

        drug_color = {
            1: (255, 0, 0),
            2: (255, 0, 255),
            3: (0, 0, 255),
            4: (0, 255, 255),
            5: (255, 255, 0),
            6: (0, 255, 0),  # 7th for white color artifact
        }

        # same as trained in categorical
        # 1: red = #bio
        # 2: purple = #iver
        # 3: blue = #iwr
        # 4: cyan = #ag1
        # 5: yellow = #c59
        # 6: green = #control

        self.model.eval()

        cur_batch = 0
        in_shape = (512, 512, 3)

        for batch in self.test_loader:
            cur_batch += 1

            imgs = batch['img']

            imgs = imgs.to(device=self.device, dtype=torch.float32)

            outputs = self.model(imgs)
            pred = torch.sigmoid(outputs)
            pred = (pred > 0.5).bool()

            for i in range(self.test_loader.batch_size):
                img_fname = batch['fname'][i]

                bw_img = np.squeeze(pred[i].cpu().numpy())

                if self.semantic is True:
                    label_max_drug = np.argmax(
                        np.count_nonzero(bw_img, axis=(1, 2))[1:])
                    bw_img = np.squeeze(bw_img[label_max_drug+1, :, :])

                labels = skimage.measure.label(bw_img, return_num=False)
                largestCC = labels == np.argmax(
                    np.bincount(labels.flat, weights=bw_img.flat))

                img_hey = img_as_ubyte(
                    imgs[i].permute(1, 2, 0).cpu().numpy())

                if self.semantic is False:
                    img_o = img_as_ubyte(largestCC)
                    img_over = np.zeros(in_shape, dtype='uint8')

                    img_over[:, :, 0] = img_hey[:, :, 0] * \
                        largestCC  # (img_mask/255)
                    img_over[:, :, 1] = img_hey[:, :, 1] * \
                        largestCC  # (img_mask/255)
                    img_over[:, :, 2] = img_hey[:, :, 2] * \
                        largestCC  # (img_mask/255)

                elif self.semantic is True:
                    img_o = np.zeros(in_shape, dtype='uint8')
                    img_o[:, :, 0] = drug_color[label_max_drug+1][0]*largestCC
                    img_o[:, :, 1] = drug_color[label_max_drug+1][1]*largestCC
                    img_o[:, :, 2] = drug_color[label_max_drug+1][2]*largestCC

                    img_over = Imgops.blend_imgs(
                        img_hey, img_o, overlap=1, ratio=0.7)

                imsave(os.path.join(self.output_mask_dir, img_fname),
                       img_as_ubyte(img_o), check_contrast=False)

                if not os.path.isfile(os.path.join(self.output_overlay_dir, img_fname)):
                    imsave(os.path.join(self.output_overlay_dir, img_fname),
                           img_as_ubyte(img_over), check_contrast=False)

                # plt.imshow(img_over)

            if cur_batch % 100 == 0:
                print(f"{cur_batch} / {len(self.test_loader)}")

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path))

    def test_batch(self, model_name):
        test_dir = os.path.join(self.model_save_dir, 'test')
        makedirs(test_dir)

        # load model here

        self.model.load_state_dict(torch.load(model_name))

        a_test_loader = iter(self.test_loader)

        step = 0

        csv_name = os.path.join(test_dir, 'batch_fnames.csv')

        with open(csv_name, 'a') as f:
            while True:
                try:
                    imgs = a_test_loader.next()['img']
                    img_fnames = a_test_loader.next()['fname']

                    imgs = imgs.to(device=self.device, dtype=torch.float32)
                    outputs = self.model(imgs)

                    pred = torch.sigmoid(outputs)

                except StopIteration:
                    print(
                        f"[!] Test sample generation finished. Samples are in {test_dir}")
                    break

                img_name = os.path.join(
                    test_dir, str(step).zfill(4) + '_img.png')
                save_name = os.path.join(
                    test_dir, str(step).zfill(4) + '_mask.png')

                vutils.save_image(imgs.data, img_name, padding=0)
                vutils.save_image(pred.data, save_name, padding=0)

                for fname in img_fnames:
                    f.write(str(step).zfill(6) + ":" + fname)
                    f.write("\n")

                step += 1

                print(f"{step}/{len(self.test_loader)}")
