import os
import sys
import itertools
from PIL import Image
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torchvision.transforms as transforms
from farabio.core.gantrainer import GanTrainer
from farabio.data.datasets import ImageDataset
from farabio.utils.helpers import ReplayBuffer
from farabio.utils.visdom import CycleganViz
from farabio.utils.regul import weights_init_normal, LambdaLR
from farabio.models.translation.cyclegan.cyclegan import Generator, Discriminator


class CycleganTrainer(GanTrainer):
    """CycleganTrainer trainer class. Override with custom methods here.

    Parameters
    ----------
    GanTrainer : farabio.core.basetrainer.BaseTrainer
        Inherits GanTrainer class
    """

    def define_data_attr(self):
        self._dataroot = self.config.dataroot
        self._size = self.config.size
        self._in_ch = self.config.in_ch
        self._out_ch = self.config.out_ch
        self._batch_size = self.config.batch_size

    def define_model_attr(self):
        self._model_save_dir = self.config.save_dir
        self._genA2B_path = self.config.generator_A2B
        self._genB2A_path = self.config.generator_A2B

    def define_train_attr(self):
        self._num_epochs = self.config.num_epochs
        self._start_epoch = self.config.start_epoch
        self._learning_rate = self.config.learning_rate
        self._decay_epoch = self.config.decay_epoch
        self._has_eval = self.config.has_eval
        if self.config.optim == 'adam':
            self._optim = torch.optim.Adam
        self._scheduler = torch.optim.lr_scheduler.LambdaLR

    def define_test_attr(self):
        self._output_dir = self.config.output_dir

    def define_log_attr(self):
        self._save_epoch = self.config.save_epoch

    def define_compute_attr(self):
        self._cuda = self.config.cuda
        self._num_workers = self.config.num_workers

    def define_misc_attr(self):
        self._mode = self.config.mode

    def build_model(self):
        self.netG_A2B = Generator(self._in_ch, self._out_ch)
        self.netG_B2A = Generator(self._out_ch, self._in_ch)
        self.netD_A = Discriminator(self._in_ch)
        self.netD_B = Discriminator(self._out_ch)

        if self._cuda:
            self.netG_A2B.cuda()
            self.netG_B2A.cuda()
            self.netD_A.cuda()
            self.netD_B.cuda()

        self.netG_A2B.apply(weights_init_normal)
        self.netG_B2A.apply(weights_init_normal)
        self.netD_A.apply(weights_init_normal)
        self.netD_B.apply(weights_init_normal)

        # Losses
        self.criterion_GAN = torch.nn.MSELoss()
        self.criterion_cycle = torch.nn.L1Loss()
        self.criterion_identity = torch.nn.L1Loss()

        # Optimizers & LR schedulers
        self.optimizer_G = self._optim(itertools.chain(self.netG_A2B.parameters(),
                                                       self.netG_B2A.parameters()),
                                       lr=self._learning_rate, betas=(0.5, 0.999))
        self.optimizer_D_A = self._optim(self.netD_A.parameters(),
                                         lr=self._learning_rate,
                                         betas=(0.5, 0.999))
        self.optimizer_D_B = self._optim(self.netD_B.parameters(),
                                         lr=self._learning_rate,
                                         betas=(0.5, 0.999))

        self.lr_scheduler_G = self._scheduler(self.optimizer_G, lr_lambda=LambdaLR(
            self._num_epochs, self._start_epoch, self._decay_epoch).step)
        self.lr_scheduler_D_A = self._scheduler(self.optimizer_D_A, lr_lambda=LambdaLR(
            self._num_epochs, self._start_epoch, self._decay_epoch).step)
        self.lr_scheduler_D_B = self._scheduler(self.optimizer_D_B, lr_lambda=LambdaLR(
            self._num_epochs, self._start_epoch, self._decay_epoch).step)

    def on_train_start(self):
        Tensor = torch.cuda.FloatTensor if self._cuda else torch.Tensor
        self.input_A = Tensor(
            self._batch_size, self._in_ch, self._size, self._size)
        self.input_B = Tensor(
            self._batch_size, self._out_ch, self._size, self._size)
        self.target_real = Variable(
            Tensor(self._batch_size).fill_(1.0), requires_grad=False)
        self.target_fake = Variable(
            Tensor(self._batch_size).fill_(0.0), requires_grad=False)

        self.fake_A_buffer = ReplayBuffer()
        self.fake_B_buffer = ReplayBuffer()

    def start_logger(self):
        self.logger = CycleganViz(self._num_epochs, len(self.train_loader))

    def on_train_epoch_start(self):
        self.train_epoch_iter = enumerate(self.train_loader)

    def train_batch(self, args):
        self.on_start_training_batch(args)

        ###### Generators A2B and B2A ######
        self.generator_zero_grad()
        self.generator_loss()
        self.generator_backward()
        self.generator_optim_step()

        ###### Discriminator A ######
        self.discriminatorA_zero_grad()
        self.discriminatorA_loss()
        self.discriminatorA_backward()
        self.discriminatorA_optim_step()

        ###### Discriminator B ######
        self.discriminatorB_zero_grad()
        self.discriminatorB_loss()
        self.discriminatorB_backward()
        self.discriminatorB_optim_step()
        self.on_end_training_batch()

    def on_start_training_batch(self, args):
        # Set model input
        self.i = args[0]
        self.batch = args[-1]
        self.real_A = Variable(self.input_A.copy_(self.batch['A']))
        self.real_B = Variable(self.input_B.copy_(self.batch['B']))

    def generator_zero_grad(self):
        self.optimizer_G.zero_grad()

    def generator_loss(self):
        """Total generator loss
        """
        self.loss_identity_A, self.loss_identity_B = self.identity_g_loss()
        self.loss_GAN_A2B, self.loss_GAN_B2A = self.gan_g_loss()
        self.loss_cycle_ABA, self.loss_cycle_BAB = self.cycle_g_loss()

        self.loss_G = self.loss_identity_A + self.loss_identity_B + \
            self.loss_GAN_A2B + self.loss_GAN_B2A + \
            self.loss_cycle_ABA + self.loss_cycle_BAB

    def generator_backward(self):
        self.loss_G.backward()

    def generator_optim_step(self):
        self.optimizer_G.step()

    def identity_g_loss(self):
        """Identity loss

        Returns
        -------
        scalar, scalar
            torch.nn.L1Loss, torch.nn.L1Loss
        """
        # G_A2B(B) should equal B if real B is fed
        same_B = self.netG_A2B(self.real_B)
        loss_identity_B = self.criterion_identity(same_B, self.real_B)*5.0
        # G_B2A(A) should equal A if real A is fed
        same_A = self.netG_B2A(self.real_A)
        loss_identity_A = self.criterion_identity(same_A, self.real_A)*5.0

        return loss_identity_A, loss_identity_B

    def gan_g_loss(self):
        """GAN loss

        Returns
        -------
        scalar, scalar
            torch.nn.MSELoss, torch.nn.MSELoss
        """
        self.fake_B = self.netG_A2B(self.real_A)
        pred_fake = self.netD_B(self.fake_B)
        loss_GAN_A2B = self.criterion_GAN(pred_fake, self.target_real)

        self.fake_A = self.netG_B2A(self.real_B)
        pred_fake = self.netD_A(self.fake_A)
        loss_GAN_B2A = self.criterion_GAN(pred_fake, self.target_real)

        return loss_GAN_A2B, loss_GAN_B2A

    def cycle_g_loss(self):
        """Cycle loss

        Returns
        -------
        scalar, scalar
            torch.nn.L1Loss, torch.nn.L1Loss
        """
        recovered_A = self.netG_B2A(self.fake_B)
        loss_cycle_ABA = self.criterion_cycle(recovered_A, self.real_A)*10.0

        recovered_B = self.netG_A2B(self.fake_A)
        loss_cycle_BAB = self.criterion_cycle(recovered_B, self.real_B)*10.0

        return loss_cycle_ABA, loss_cycle_BAB

    def discriminatorA_zero_grad(self):
        self.optimizer_D_A.zero_grad()

    def discriminatorA_loss(self):
        """Loss for discriminator A: fake and real.
        """
        # Real loss
        loss_D_real = self.real_dA_loss()
        # Fake loss
        loss_D_fake = self.fake_dA_loss()
        # Total loss
        self.loss_D_A = (loss_D_real + loss_D_fake)*0.5

    def real_dA_loss(self):
        """Loss for discriminator A: real

        Returns
        -------
        scalar
            torch.nn.MSELoss
        """
        pred_real = self.netD_A(self.real_A)
        loss_D_real = self.criterion_GAN(pred_real, self.target_real)
        return loss_D_real

    def fake_dA_loss(self):
        """Loss for discriminator A: fake

        Returns
        -------
        scalar
            torch.nn.MSELoss
        """
        self.fake_A = self.fake_A_buffer.push_and_pop(self.fake_A)
        pred_fake = self.netD_A(self.fake_A.detach())
        loss_D_fake = self.criterion_GAN(pred_fake, self.target_fake)
        return loss_D_fake

    def discriminatorA_backward(self):
        self.loss_D_A.backward()

    def discriminatorA_optim_step(self):
        self.optimizer_D_A.step()

    def discriminatorB_zero_grad(self):
        self.optimizer_D_B.zero_grad()

    def discriminatorB_loss(self):
        """Loss for discriminator B: fake and real.
        """
        # Real loss
        loss_D_real = self.real_dB_loss()
        # Fake loss
        loss_D_fake = self.fake_dB_loss()
        # Total loss
        self.loss_D_B = (loss_D_real + loss_D_fake)*0.5

    def real_dB_loss(self):
        """Loss for discriminator B: real

        Returns
        -------
        scalar
            torch.nn.MSELoss
        """
        pred_real = self.netD_B(self.real_B)
        loss_D_real = self.criterion_GAN(pred_real, self.target_real)

        return loss_D_real

    def fake_dB_loss(self):
        """Loss for discriminator B: fake

        Returns
        -------
        scalar
            torch.nn.MSELoss
        """
        self.fake_B = self.fake_B_buffer.push_and_pop(self.fake_B)
        pred_fake = self.netD_B(self.fake_B.detach())
        loss_D_fake = self.criterion_GAN(pred_fake, self.target_fake)

        return loss_D_fake

    def discriminatorB_backward(self):
        self.loss_D_B.backward()

    def discriminatorB_optim_step(self):
        self.optimizer_D_B.step()

    def on_end_training_batch(self):
        self.logger.log({'loss_G': self.loss_G,
                         'loss_G_identity': (self.loss_identity_A + self.loss_identity_B),
                         'loss_G_GAN': (self.loss_GAN_A2B + self.loss_GAN_B2A),
                         'loss_G_cycle': (self.loss_cycle_ABA + self.loss_cycle_BAB),
                         'loss_D': (self.loss_D_A + self.loss_D_B)},
                        images={'real_A': self.real_A,
                                'real_B': self.real_B,
                                'fake_A': self.fake_A,
                                'fake_B': self.fake_B})

    def on_train_epoch_end(self):
        # Update learning rates
        self.lr_scheduler_G.step()
        self.lr_scheduler_D_A.step()
        self.lr_scheduler_D_B.step()

    def on_epoch_end(self):
        if self.epoch % self._save_epoch == 0:
            self.save_model()

    def get_trainloader(self):
        transforms_ = [transforms.Resize(int(self._size*1.12), Image.BICUBIC),
                       transforms.RandomCrop(self._size),
                       transforms.RandomHorizontalFlip(),
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

        self.train_loader = DataLoader(ImageDataset(self._dataroot, transforms_=transforms_, unaligned=True, mode='train'),
                                       batch_size=self._batch_size, shuffle=True, num_workers=self._num_workers)

    def get_testloader(self):
        transforms_ = [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

        self.test_loader = DataLoader(ImageDataset(self._dataroot, transforms_=transforms_, unaligned=False, mode='test'),
                                      batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers)

    def save_model(self):
        """Save model

        Parameters
        ----------
        epoch : int
            current epoch
        """
        netG_A2B_name = os.path.join(self._model_save_dir, "netG_A2B.pth")
        netG_B2A_name = os.path.join(self._model_save_dir, "netG_B2A.pth")
        netD_A_name = os.path.join(self._model_save_dir, "netD_A.pth")
        netD_B_name = os.path.join(self._model_save_dir, "netD_B.pth")

        # Save models checkpoints
        torch.save(self.netG_A2B.state_dict(), netG_A2B_name)
        torch.save(self.netG_B2A.state_dict(), netG_B2A_name)
        torch.save(self.netD_A.state_dict(), netD_A_name)
        torch.save(self.netD_B.state_dict(), netD_B_name)

    def load_model(self):

        self.netG_A2B = Generator(self._in_ch, self._out_ch)
        self.netG_B2A = Generator(self._out_ch, self._in_ch)

        if self._cuda:
            self.netG_A2B.cuda()
            self.netG_B2A.cuda()

        # Load state dicts
        self.netG_A2B.load_state_dict(torch.load(self._genA2B_path))
        self.netG_B2A.load_state_dict(torch.load(self._genB2A_path))

        # Set model's test mode
        self.netG_A2B.eval()
        self.netG_B2A.eval()

        # # Create output dirs if they don't exist
        if not os.path.exists(os.path.join(self._output_dir, "A")):
            os.makedirs(os.path.join(self._output_dir, "A"))
        if not os.path.exists(os.path.join(self._output_dir, "B")):
            os.makedirs(os.path.join(self._output_dir, "B"))

    def on_test_start(self):
        # Inputs & targets memory allocation
        Tensor = torch.cuda.FloatTensor if self._cuda else torch.Tensor
        self.input_A = Tensor(
            self._batch_size, self._in_ch, self._size, self._size)
        self.input_B = Tensor(
            self._batch_size, self._out_ch, self._size, self._size)

        self.test_loop_iter = enumerate(self.test_loader)

    def test_step(self, args):
        self.i = args[0]
        batch = args[-1]

        # Set model input
        real_A = Variable(self.input_A.copy_(batch['A']))
        real_B = Variable(self.input_B.copy_(batch['B']))

        # Generate output
        self.fake_B = 0.5*(self.netG_A2B(real_A).data + 1.0)
        self.fake_A = 0.5*(self.netG_B2A(real_B).data + 1.0)

    def on_end_test_batch(self):
        idx = str(self.i+1).zfill(4)

        # Save image files
        save_image(self.fake_A, os.path.join(
            self._output_dir, "A", idx+".png"))
        save_image(self.fake_B, os.path.join(
            self._output_dir, "B", idx+".png"))
        sys.stdout.write('\rGenerated images %04d of %04d' %
                         (self.i+1, len(self.test_loader)))
