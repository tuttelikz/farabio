import os
import torch
from farabi.utils.helpers import makedirs
import torchvision.utils as vutils
from srgan import Generator, Discriminator
from farabi.utils.losses import GeneratorLoss
from tqdm import tqdm
from torch.autograd import Variable
from farabi.utils.metrics import ssim
import math
from farabi.prep.transforms import display_transform
import pandas as pd
import numpy as np


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
        self.upscale_factor = config.upscale_factor

        self.cuda = config.cuda

        self.num_epochs = config.num_epochs
        self.model_save_dir = config.model_save_dir
        
        if config.optim == 'adam':
            self.optim = torch.optim.Adam

        self.build_model()

        if mode == 'train':
            self.train_loader = data_loader[0]
            self.valid_loader = data_loader[-1]
        elif mode == 'test':
            self.test_loader = data_loader
            self.load_model(config.model_name)

    def train(self):
        self.results = {'d_loss': [], 'g_loss': [],
                        'd_score': [], 'g_score': [], 'psnr': [], 'ssim': []}
        """Training function

        Parameters
        ----------
        epoch : int
            current epoch
        """
        for self.epoch in range(1, self.num_epochs + 1):
            train_bar = tqdm(self.train_loader)
            self.running_results = {'batch_sizes': 0, 'd_loss': 0,
                                    'g_loss': 0, 'd_score': 0, 'g_score': 0}

            self.netG.train()
            self.netD.train()

            for data, target in train_bar:
                batch_size = data.size(0)
                self.running_results['batch_sizes'] += batch_size

                ############################
                # (1) Update D network: maximize D(x)-1-D(G(z))
                ###########################
                real_img = Variable(target)
                if self.cuda:
                    real_img = real_img.cuda()
                z = Variable(data)
                if self.cuda:
                    z = z.cuda()
                fake_img = self.netG(z)

                self.netD.zero_grad()
                real_out = self.netD(real_img).mean()
                fake_out = self.netD(fake_img).mean()
                d_loss = 1 - real_out + fake_out
                d_loss.backward(retain_graph=True)
                self.optimizerD.step()

                ############################
                # (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
                ###########################

                self.netG.zero_grad()
                fake_img = self.netG(z)
                fake_out = self.netD(fake_img).mean()

                g_loss = self.generator_criterion(fake_out, fake_img, real_img)
                g_loss.backward()

                self.optimizerG.step()

                # loss for current batch before optimization
                self.running_results['g_loss'] += g_loss.item() * batch_size
                self.running_results['d_loss'] += d_loss.item() * batch_size
                self.running_results['d_score'] += real_out.item() * batch_size
                self.running_results['g_score'] += fake_out.item() * batch_size

                train_bar.set_description(desc='[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f' % (
                    self.epoch, self.num_epochs, self.running_results['d_loss'] /
                    self.running_results['batch_sizes'],
                    self.running_results['g_loss'] /
                    self.running_results['batch_sizes'],
                    self.running_results['d_score'] /
                    self.running_results['batch_sizes'],
                    self.running_results['g_score'] / self.running_results['batch_sizes']))

            self.evaluate()
            self.save_model()

            if self.epoch % 10 == 0 and self.epoch != 0:
                self.save_csv()

    def evaluate(self):
        """Test function

        Parameters
        ----------
        epoch : int
            current epoch
        """
        self.netG.eval()

        out_img_path = os.path.join(self.model_save_dir,
                                    "SRF_" + str(self.upscale_factor), "output")
        makedirs(out_img_path)

        with torch.no_grad():
            val_bar = tqdm(self.valid_loader)
            self.valing_results = {'mse': 0, 'ssims': 0,
                                   'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
            val_images = []

            for val_lr, val_hr_restore, val_hr in val_bar:
                batch_size = val_lr.size(0)
                self.valing_results['batch_sizes'] += batch_size
                lr = val_lr
                hr = val_hr
                if self.cuda:
                    lr = lr.cuda()
                    hr = hr.cuda()
                sr = self.netG(lr)

                batch_mse = ((sr - hr) ** 2).data.mean()
                self.valing_results['mse'] += batch_mse * batch_size
                batch_ssim = ssim(sr, hr).item()
                self.valing_results['ssims'] += batch_ssim * batch_size
                self.valing_results['psnr'] = 10 * math.log10((hr.max()**2) / (
                    self.valing_results['mse'] / self.valing_results['batch_sizes']))
                self.valing_results['ssim'] = self.valing_results['ssims'] / \
                    self.valing_results['batch_sizes']
                val_bar.set_description(
                    desc='[converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f' % (
                        self.valing_results['psnr'], self.valing_results['ssim']))

                val_images.extend(
                    [display_transform()(val_hr_restore.squeeze(0)), display_transform()(hr.data.cpu().squeeze(0)),
                     display_transform()(sr.data.cpu().squeeze(0))])

            val_images = torch.stack(val_images)
            val_images = torch.chunk(val_images, val_images.size(0) // 15)
            val_save_bar = tqdm(val_images, desc='[saving training results]')

            index = 1
            for image in val_save_bar:
                image = vutils.make_grid(image, nrow=3, padding=5)
                vutils.save_image(
                    image, out_img_path + 'epoch_%d_index_%d.png' % (self.epoch, index), padding=5)
                index += 1

    def build_model(self):
        """Build model

        Parameters
        ----------
        epoch : int
            current epoch
        """

        self.netG = Generator(self.upscale_factor)
        self.netD = Discriminator()
        self.generator_criterion = GeneratorLoss()

        if self.cuda:
            self.netG.cuda()
            self.netD.cuda()
            self.generator_criterion.cuda()

        self.optimizerG = self.optim(self.netG.parameters())
        self.optimizerD = self.optim(self.netD.parameters())

        # print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
        # print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))

    def save_model(self):
        """Save model

        Parameters
        ----------
        epoch : int
            current epoch
        """
        out_model_path = os.path.join(self.model_save_dir, "epochs")
        makedirs(out_model_path)

        g_name = f'netG_epoch_{self.upscale_factor}_{self.epoch}.pth'
        d_name = f'netD_epoch_{self.upscale_factor}_{self.epoch}.pth'

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
        out_stat_path = os.path.join(self.model_save_dir,
                                     "SRF_" + str(self.upscale_factor), "statistics")

        makedirs(out_stat_path)

        data_frame = pd.DataFrame(
            data={'Loss_D': self.results['d_loss'], 'Loss_G': self.results['g_loss'], 'Score_D': self.results['d_score'],
                  'Score_G': self.results['g_score'], 'PSNR': self.results['psnr'], 'SSIM': self.results['ssim']},
            index=range(1, self.epoch + 1))

        csv_name = 'srf_' + str(self.upscale_factor) + '_train_results.csv'
        data_frame.to_csv(os.path.join(
            out_stat_path, csv_name), index_label='Epoch')

    def test(self):
        self.results = {'Set5': {'psnr': [], 'ssim': []}, 'Set14': {'psnr': [], 'ssim': []}, 'BSD100': {'psnr': [], 'ssim': []},
                        'Urban100': {'psnr': [], 'ssim': []}, 'SunHays80': {'psnr': [], 'ssim': []}}

        test_bar = tqdm(self.test_loader, desc='[testing benchmark datasets]')

        out_bench_path = os.path.join(self.model_save_dir,
                                      "SRF_" + str(self.upscale_factor), "benchmark_results")

        makedirs(out_bench_path)

        for image_name, lr_image, hr_restore_img, hr_image in test_bar:
            image_name = image_name[0]
            lr_image = Variable(lr_image)
            hr_image = Variable(hr_image)

            if self.cuda:
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
            vutils.save_image(image, out_bench_path + image_name.split('.')[0] + '_psnr_%.4f_ssim_%.4f.' % (psnr, _ssim) +
                              image_name.split('.')[-1], padding=5)

            # save psnr\ssim
            # print(image_name)
            # print(image_name.split('_')[0])
            self.results['SunHays80']['psnr'].append(psnr)
            self.results['SunHays80']['ssim'].append(_ssim)

        out_stat_path = os.path.join(self.model_save_dir,
                                     "SRF_" + str(self.upscale_factor), "statistics")

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
                                       str(self.upscale_factor) + '_test_results.csv'), index_label='DataSet')

    def load_model(self, model_path):
        self.netG.eval()

        self.netG.load_state_dict(torch.load(model_path))

    def test_batch(self, model_name):
        pass
