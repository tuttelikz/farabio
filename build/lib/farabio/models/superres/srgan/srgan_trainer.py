import os
import time
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.utils as vutils
from farabio.data.transforms import display_transform
from farabio.core.gantrainer import GanTrainer
from farabio.utils.losses import GeneratorLoss
from farabio.utils.metrics import ssim
from farabio.utils.helpers import x, makedirs
from farabio.models.superres.srgan.srgan import Generator, Discriminator
from farabio.data.datasets import TrainDatasetFromFolder, ValDatasetFromFolder, TestDatasetFromFolder


class SrganTrainer(GanTrainer):
    """SrganTrainer trainer class. Override with custom methods here.

    Parameters
    ----------
    GanTrainer : parent object
        Parent object of SrganTrainer
    """

    def define_data_attr(self):
        self._trainset_dir = self.config.train_set
        self._validset_dir = self.config.valid_set
        self._testset_dir = self.config.test_set
        self._batch_size_train = self.config.batch_size_train
        self._batch_size_valid = self.config.batch_size_valid
        self._batch_size_test = self.config.batch_size_test
        self._upscale_factor = self.config.upscale_factor
        self._crop_size = self.config.crop_size

    def define_model_attr(self):
        self._model_path = self.config.model_path
        self._model_save_dir = self.config.model_save_dir

    def define_train_attr(self):
        self._num_epochs = self.config.num_epochs
        self._start_epoch = self.config.start_epoch
        self._has_eval = True
        if self.config.optim == 'adam':
            self.optim = torch.optim.Adam

    def define_log_attr(self):
        self._save_epoch = self.config.save_epoch
        self._save_csv_epoch = self.config.save_csv_epoch
    
    def define_compute_attr(self):
        self._cuda = self.config.cuda
        self._num_workers = self.config.num_workers

    def define_misc_attr(self):
        self._mode = self.config.mode

    def get_trainloader(self):
        if self._mode == 'train':
            train_set = TrainDatasetFromFolder(
                self._trainset_dir, crop_size=self._crop_size, upscale_factor=self._upscale_factor)
            valid_set = ValDatasetFromFolder(
                self._validset_dir, upscale_factor=self._upscale_factor)
            self.train_loader = DataLoader(
                dataset=train_set, num_workers=self._num_workers, batch_size=self._batch_size_train, shuffle=True)
            self.valid_loader = DataLoader(dataset=valid_set, num_workers=4,
                                           batch_size=self._batch_size_valid, shuffle=False)

    def get_testloader(self):
        if self._mode == 'test':
            test_set = TestDatasetFromFolder(
                self._testset_dir, upscale_factor=self._upscale_factor)
            self.test_loader = DataLoader(
                dataset=test_set, num_workers=self._num_workers, batch_size=self._batch_size_test, shuffle=False)
            # self.load_model(config.model_name)

    def build_model(self):
        """Build model

        Parameters
        ----------
        epoch : int
            current epoch
        """

        self.netG = Generator(self._upscale_factor)
        self.netD = Discriminator()
        self.generator_criterion = GeneratorLoss()

        if self._cuda:
            self.netG.cuda()
            self.netD.cuda()
            self.generator_criterion.cuda()

        self.optimizerG = self.optim(self.netG.parameters())
        self.optimizerD = self.optim(self.netD.parameters())

        # print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
        # print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))

    def start_logger(self):
        self.results = {'d_loss': [], 'g_loss': [],
                        'd_score': [], 'g_score': [], 'psnr': [], 'ssim': []}

    def on_train_epoch_start(self):
        self.train_epoch_iter = tqdm(self.train_loader)
        self.running_results = {'batch_sizes': 0, 'd_loss': 0,
                                'g_loss': 0, 'd_score': 0, 'g_score': 0}

        self.netG.train()
        self.netD.train()

    def train_batch(self, args):
        self.on_start_training_batch(args)

        ###### Discriminator ######
        self.discriminator_zero_grad()
        self.discriminator_loss()
        self.discriminator_backward()
        self.discriminator_optim_step()

        ###### Generator ######
        self.generator_zero_grad()
        self.generator_loss()
        self.generator_backward()
        self.generator_optim_step()

        self.on_end_training_batch()

    def on_start_training_batch(self, args):
        self.data = args[0]
        self.target = args[1]
        self.batch_size = self.data.size(0)
        self.running_results['batch_sizes'] += self.batch_size

    def discriminator_zero_grad(self):
        self.netD.zero_grad()

    def discriminator_loss(self):
        ############################
        # (1) Update D network: maximize D(x)-1-D(G(z))
        ###########################
        self.real_img = Variable(self.target)
        self.z = Variable(self.data)

        if self._cuda:
            self.real_img = self.real_img.cuda()
            self.z = self.z.cuda()

        fake_img = self.netG(self.z)
        self.real_out = self.netD(self.real_img).mean()
        fake_out = self.netD(fake_img).mean()
        self._discriminator_loss = 1 - self.real_out + fake_out

    def discriminator_optim_step(self):
        """Discriminator optimizer step
        """
        self.optimizerD.step()

    def generator_zero_grad(self):
        """Zero grad
        """
        self.netG.zero_grad()

    def generator_loss(self):
        ############################
        # (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
        ###########################

        fake_img = self.netG(self.z)
        self.fake_out = self.netD(fake_img).mean()

        self._generator_loss = self.generator_criterion(
            self.fake_out, fake_img, self.real_img)

    def generator_backward(self):
        """Hook: sends backward
        """
        self._generator_loss.backward()

    def generator_optim_step(self):
        """Discriminator optimizer step
        """
        self.optimizerG.step()

    def optimizer_zero_grad(self):
        """Zero grad
        """
        self.netG.zero_grad()
        self.netD.zero_grad()

    def discriminator_backward(self):
        self._discriminator_loss.backward(retain_graph=True)

    def on_end_training_batch(self):
        # loss for current batch before optimization
        self.running_results['g_loss'] += self._generator_loss.item() * \
            self.batch_size
        self.running_results['d_loss'] += self._discriminator_loss.item() * \
            self.batch_size
        self.running_results['d_score'] += self.real_out.item() * \
            self.batch_size
        self.running_results['g_score'] += self.fake_out.item() * \
            self.batch_size

        self.train_epoch_iter.set_description(desc='[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f' % (
            self.epoch, self._num_epochs, self.running_results['d_loss'] /
            self.running_results['batch_sizes'],
            self.running_results['g_loss'] /
            self.running_results['batch_sizes'],
            self.running_results['d_score'] /
            self.running_results['batch_sizes'],
            self.running_results['g_score'] / self.running_results['batch_sizes']))

    def on_epoch_end(self):
        if self.epoch % self._save_csv_epoch == 0 and self.epoch != 0:
            self.save_model()
            self.save_csv()

    def on_evaluate_epoch_start(self):
        self.netG.eval()
        self.out_img_path = os.path.join(self._model_save_dir,
                                         "SRF_" + str(self._upscale_factor), "output")
        makedirs(self.out_img_path)

        self.valing_results = {'mse': 0, 'ssims': 0,
                               'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
        self.val_images = []

        self.valid_epoch_iter = tqdm(self.valid_loader)

    def evaluate_batch(self, args):
        val_lr = args[0]
        val_hr_restore = args[1]
        val_hr = args[2]

        self.valing_results['batch_sizes'] += self._batch_size_valid
        lr = val_lr
        self.hr = val_hr
        if self._cuda:
            lr = lr.cuda()
            self.hr = self.hr.cuda()

        self.sr = self.netG(lr)
        self.val_hr_restore = val_hr_restore
        self.batch_mse = ((self.sr - self.hr) ** 2).data.mean()
        self.valing_results['mse'] += self.batch_mse * self._batch_size_valid
        self.batch_ssim = ssim(self.sr, self.hr).item()

    def on_evaluate_batch_end(self):
        self.valing_results['ssims'] += self.batch_ssim * \
            self._batch_size_valid
        self.valing_results['psnr'] = 10 * math.log10((self.hr.max()**2) / (
            self.valing_results['mse'] / self.valing_results['batch_sizes']))
        self.valing_results['ssim'] = self.valing_results['ssims'] / \
            self.valing_results['batch_sizes']
        self.valid_epoch_iter.set_description(
            desc='[converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f' % (
                self.valing_results['psnr'], self.valing_results['ssim']))

        self.val_images.extend(
            [display_transform()(self.val_hr_restore.squeeze(0)), display_transform()(self.hr.data.cpu().squeeze(0)),
                display_transform()(self.sr.data.cpu().squeeze(0))])

    def on_evaluate_epoch_end(self):
        self.val_images = torch.stack(self.val_images)
        self.val_images = torch.chunk(
            self.val_images, self.val_images.size(0) // 15)
        val_save_bar = tqdm(self.val_images, desc='[saving training results]')

        index = 1
        for image in val_save_bar:
            image = vutils.make_grid(image, nrow=3, padding=5)
            vutils.save_image(
                image, self.out_img_path + 'epoch_%d_index_%d.png' % (self.epoch, index), padding=5)
            index += 1

    def save_model(self):
        """Save model

        Parameters
        ----------
        epoch : int
            current epoch
        """
        out_model_path = os.path.join(self._model_save_dir, "epochs")
        makedirs(out_model_path)

        g_name = f'netG_epoch_{self._upscale_factor}_{self.epoch}.pth'
        d_name = f'netD_epoch_{self._upscale_factor}_{self.epoch}.pth'

        torch.save(self.netG.state_dict(),
                   os.path.join(out_model_path, g_name))
        torch.save(self.netG.state_dict(),
                   os.path.join(out_model_path, d_name))

        self.results['d_loss'].append(
            self.running_results['d_loss'] / self.running_results['batch_sizes'])
        self.results['g_loss'].append(
            self.running_results['g_loss'] / self.running_results['batch_sizes'])
        self.results['d_score'].append(
            self.running_results['d_score'] / self.running_results['batch_sizes'])
        self.results['g_score'].append(
            self.running_results['g_score'] / self.running_results['batch_sizes'])
        self.results['psnr'].append(self.valing_results['psnr'])
        self.results['ssim'].append(self.valing_results['ssim'])

    def save_csv(self):
        # save loss\scores\psnr\ssim
        out_stat_path = os.path.join(self._model_save_dir,
                                     "SRF_" + str(self._upscale_factor), "statistics")

        makedirs(out_stat_path)
        print("saving .csv file")
        print(self.results)
        data_frame = pd.DataFrame(
            data={'Loss_D': self.results['d_loss'], 'Loss_G': self.results['g_loss'], 'Score_D': self.results['d_score'],
                  'Score_G': self.results['g_score'], 'PSNR': self.results['psnr'], 'SSIM': self.results['ssim']},
            index=range(1, self.epoch+1))

        csv_name = 'srf_' + str(self._upscale_factor) + '_train_results.csv'
        data_frame.to_csv(os.path.join(
            out_stat_path, csv_name), index_label='Epoch')

    def load_model(self):
        self.netG.eval()
        self.netG.load_state_dict(torch.load(self._model_path))

    def test_batch(self, model_name):
        pass

    def on_test_start(self):
        self.results = {'Set5': {'psnr': [], 'ssim': []}, 'Set14': {'psnr': [], 'ssim': []}, 'BSD100': {'psnr': [], 'ssim': []},
                        'Urban100': {'psnr': [], 'ssim': []}, 'SunHays80': {'psnr': [], 'ssim': []}}

        self.out_bench_path = os.path.join(self._model_save_dir,
                                           "SRF_" + str(self._upscale_factor), "benchmark_results")

        makedirs(self.out_bench_path)

        test_bar = tqdm(self.test_loader, desc='[testing benchmark datasets]')
        self.test_loop_iter = test_bar

    def test_step(self, test_arg):
        image_name = test_arg[0]
        lr_image = test_arg[1]
        hr_restore_img = test_arg[2]
        hr_image = test_arg[3]

        image_name = image_name[0]
        lr_image = Variable(lr_image)
        hr_image = Variable(hr_image)

        if self._cuda:
            lr_image = lr_image.cuda()
            hr_image = hr_image.cuda()

        sr_image = self.netG(lr_image)

        mse = ((hr_image - sr_image) ** 2).data.mean()
        psnr = 10 * math.log10(1 / mse)
        _ssim = ssim(sr_image, hr_image).item()  # data[0]

        test_images = torch.stack(
            [display_transform()(hr_restore_img.squeeze(0)), display_transform()(hr_image.data.cpu().squeeze(0)),
             display_transform()(sr_image.data.cpu().squeeze(0))])

        image = vutils.make_grid(test_images, nrow=3, padding=5)
        vutils.save_image(image, self.out_bench_path + image_name.split('.')[0] + '_psnr_%.4f_ssim_%.4f.' % (psnr, _ssim) +
                          image_name.split('.')[-1], padding=5)

        # save psnr\ssim
        # print(image_name)
        # print(image_name.split('_')[0])
        self.results['SunHays80']['psnr'].append(psnr)
        self.results['SunHays80']['ssim'].append(_ssim)

    def on_test_end(self):
        out_stat_path = os.path.join(self._model_save_dir,
                                     "SRF_" + str(self._upscale_factor), "statistics")

        makedirs(out_stat_path)

        saved_results = {'psnr': [], 'ssim': []}
        for item in self.results.values():
            psnr = np.array(item['psnr'])
            _ssim = np.array(item['ssim'])
            if (len(psnr) == 0) or (len(_ssim) == 0):
                psnr = 'No data'
                _ssim = 'No data'
            else:
                psnr = psnr.mean()
                _ssim = _ssim.mean()
            saved_results['psnr'].append(psnr)
            saved_results['ssim'].append(_ssim)

        data_frame = pd.DataFrame(saved_results, self.results.keys())
        data_frame.to_csv(os.path.join(out_stat_path, 'srf_' +
                                       str(self._upscale_factor) + '_test_results.csv'), index_label='DataSet')
