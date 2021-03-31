import os
import time
from farabio.data.dataloader import get_trainloader, get_testloader
from config import get_config
from farabio.utils.helpers import x, makedirs
import pandas as pd
from farabio.core.trainermod import BaseTrainer
from farabio.models.segmentation.unet.unet import Unet
import torch
from farabio.utils.regul import EarlyStopping
from farabio.utils.losses import Losses
from farabio.utils.loggers import TensorBoard, UnetViz
import numpy as np
import skimage
from skimage.io import imsave
from skimage import img_as_ubyte
from farabio.prep.imgops import ImgOps
from torchsummary import summary


class UnetTrainer(BaseTrainer):
    def init_attributes(self):
        self.early_stopping = EarlyStopping(
            patience=config.patience, verbose=True)
        if config.use_tensorboard:
            self.tb = TensorBoard(os.path.join(self.model_save_dir, "logs"))
        else:
            self.tb = None

        self.output_mask_dir = config.output_mask_dir
        self.output_overlay_dir = config.output_overlay_dir
        self.semantic = config.semantic
        self.in_ch = config.in_ch
        self.out_ch = config.out_ch

    def get_dataloader(self):
        if self.mode == 'train':
            data_loader = get_trainloader(
                config, pth_train_img, pth_train_lbl, augment=False)

            self.train_loader = data_loader[0]
            self.valid_loader = data_loader[-1]

        elif self.mode == 'test':
            data_loader = get_testloader(
                config, pth_test_img)
            self.test_loader = data_loader

    def build_model(self):
        self.model = Unet(self.in_ch, self.out_ch)
        summary(self.model, [(3, 512, 512)])

    def init_optimizers(self, learning_rate):
        if self.data_parallel is True:
            self.optimizer = torch.optim.Adam(list(self.model.parameters()),
                                              lr=learning_rate)

        elif self.data_parallel is False:
            self.optimizer = torch.optim.Adam(self.model.parameters(),
                                              lr=learning_rate)
    # def training_step(self, batch, batch_idx):
    #     print(self.device)
    #     imgs = batch['img']
    #     masks = batch['lbl']
    #     imgs = imgs.to(device=self.device, dtype=torch.float32)

    #     if self.semantic is True:
    #         # masks = masks.to('cuda', dtype=torch.long)
    #         masks = masks.to(device=self.device, dtype=torch.long)
    #     else:
    #         masks = masks.to(device=self.device, dtype=torch.float32)

    #     outputs = self.model(imgs)

    #     if self.semantic is True:
    #         loss = Losses().extract_loss(outputs, masks, self.device)
    #     else:
    #         loss = Losses().calc_loss(outputs, masks)

    #     self.log_batch(imgs, masks, outputs)
    #     return loss

    # def on_train_epoch_end(self, epoch, batch_train_loss, train_loader_len):
    #     epoch_training_loss = round(
    #         batch_train_loss / train_loader_len, 4)

    #     self.epoch_train_losses.append(epoch_training_loss)

    # def on_end_training_batch(self, epoch, batch_idx, train_loss, train_loader_len):
    #     if (batch_idx % 100) == 0:
    #         print(
    #             f"===> Epoch [{epoch}]({batch_idx}/{train_loader_len}): Loss: {train_loss:.4f}")

    # def evaluate_step(self, batch, batch_idx):
    #     imgs = batch['img']
    #     masks = batch['lbl']

    #     imgs = imgs.to(device=self.device, dtype=torch.float32)

    #     if self.semantic is True:
    #         masks = masks.to(device=self.device, dtype=torch.long)
    #         # masks = masks.to('cuda', dtype=torch.long)
    #     else:
    #         masks = masks.to(device=self.device, dtype=torch.float32)

    #     outputs = self.model(imgs)  # calc loss

    #     if self.semantic is True:
    #         val_loss = Losses().extract_loss(outputs, masks, self.device)
    #     else:
    #         val_loss = Losses().calc_loss(outputs, masks)

    #     return val_loss.item()

    # def on_evaluate_end(self, batch_validation_loss, valid_loader_len, epoch):
    #     epoch_valid_loss = round(
    #         batch_validation_loss / valid_loader_len, 4)

    #     self.epoch_valid_losses.append(epoch_valid_loss)
    #     print(f"===> Epoch {epoch} Valid Loss: {epoch_valid_loss}")

    #     self.tb.scalar_summary('val_loss', epoch_valid_loss, epoch)
    #     self.early_stopping(epoch_valid_loss, self.model, self.model_save_dir)

    #     self.early_stop = self.early_stopping.early_stop

    #     if self.early_stop is True:
    #         print("Early stopping")
    #         self.save_model()
    #         self.stop_train()

    #     if self.epoch % self.save_epoch == 0:
    #         self.save_model()

    # def start_logger(self):
    #     self.logger = UnetViz(self.num_epochs, len(self.train_loader))

    # def log_batch(self, imgs, masks, outputs):
    #     self.logger.log(
    #         images={'imgs': imgs, 'masks': masks, 'outputs': outputs})

    # def save_model(self, *args):
    #     model_name = os.path.join(self.model_save_dir, "semunet.pt")

    #     if self.data_parallel is False:
    #         torch.save(self.model.state_dict(), model_name)
    #     elif self.data_parallel is True:
    #         torch.save(self.model.module.state_dict(), model_name)

    def evaluate_step(self):
        pass

    def training_step(self):
        pass

    def test_step(self, batch, batch_idx):
        drug_color = {
            1: (255, 0, 0),
            2: (255, 0, 255),
            3: (0, 0, 255),
            4: (0, 255, 255),
            5: (255, 255, 0),
            6: (0, 255, 0),  # 7th for white color artifact
            7: (255, 255, 255)
        }

        in_shape = (512, 512, 3)

        imgs = batch['img']

        imgs = imgs.to(device=self.device, dtype=torch.float32)

        outputs = self.model(imgs)
        pred = torch.sigmoid(outputs)
        pred = (pred > 0.5).bool()

        # saving images with filenames
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

        if batch_idx % 100 == 0:
            print(f"{batch_idx} / {len(self.test_loader)}")


if __name__ == "__main__":

    start_time = time.time()
    config, unparsed = get_config()

    ####################################################################
    # Train: 12 cycles of xenopus
    # ___________________________________________________________________

    # base_dir = '/home/DATA_Lia/data_02/DATASET_SUZY/SM_XENOPUS/DATA/Dataset/'
    # pth_train, pth_test = x(base_dir, 'Train_200909_all'), x(
    #     base_dir, 'Test_200909_all')
    # pth_train_img, pth_train_lbl = x(pth_train, 'Image'), x(pth_train, 'Label')
    # pth_test_img, pth_test_lbl = x(pth_test, 'Image'), x(pth_test, 'Label')

    # models_pth = "/home/DATA_Lia/data_02/DATASET_SUZY/MODELS/segmentation"

    ####################################################################
    # Test: blind cycle of xenopus
    # ___________________________________________________________________
    # base_dir = "/home/DATA_Lia/data_02/DATASET_SUZY/SM_XENOPUS/DATA/Roi_512_Blind14"
    # pth_test_omask, pth_test_ovgen = x(base_dir, 'Mask-att'), x(base_dir, 'Overlay-att')
    # model_pth = '/home/DATA_Lia/data_02/DATASET_SUZY/MODELS/attunet16_0.350_210112_ch2.pt'

    ####################################################################
    # Train: semantic segmentation with artifacts
    # ___________________________________________________________________
    # base_dir = "/home/DATA_Lia/data_02/DATASET_SUZY/SM_XENOPUS/DATA/Dataset/Train_200909_all/"
    # pth_train_img, pth_train_lbl = x(
    #     base_dir, 'Image-sanmo-bal-wbadm-aug'), x(base_dir, 'Mask-sanmo-bal-wbadm-aug-class')

    # date = time.strftime("%y%m%d", time.localtime())
    # model_dir = '/home/DATA_Lia/data_02/DATASET_SUZY/MODELS/segmentation'
    # model_save_dir = os.path.join(model_dir, "semunet_" + date)
    # # makedirs(model_save_dir)
    # config.model_save_dir = model_save_dir

    ######################################################################
    # Test: semantic segmentation

    base_dir = "/home/DATA_Lia/data_02/DATASET_SUZY/SM_XENOPUS/DATA/Dataset/Test_200909_all_attunet/"
    config.output_mask_dir, config.output_overlay_dir = x(
        base_dir, 'Mask-semantic-unet2'), x(base_dir, "Overlay-semantic-unet2")

    pth_test_img = x(base_dir, "Image")
    csv_file = x(base_dir, "DATA_GOODBAD_v210118.csv")  # 25.csv

    df = pd.read_csv(csv_file)
    good_tag_df = df.loc[df['tag'] == "good"]
    good_fnames = good_tag_df['fname_src'].values.tolist()

    config.model_load_dir = "/home/DATA_Lia/data_02/DATASET_SUZY/MODELS/segmentation/semunet_210128_noise_nowall/semunet.pt"

    ######################################################################
    # Train: initiate trainer and start training
    trnr = UnetTrainer(config)
    trnr.test()

    time_elapsed = time.time() - start_time
    print(
        f'Complete in {time_elapsed // 60}m {time_elapsed % 60: .2f}s')
