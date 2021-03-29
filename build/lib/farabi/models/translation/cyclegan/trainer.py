import torch
import torch.nn as nn
import os
import sys
from cyclegan import Generator, Discriminator
from torch.autograd import Variable
from farabi.utils.regul import weights_init_normal, LambdaLR
from farabi.utils.helpers import ReplayBuffer
import itertools
from farabi.utils.loggers import LoggerViz
from torchvision.utils import save_image


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
        data_loader : torch.utils.data.DataLoader
            dataloader
        model : nn.Module
            segmentation model
        """
        self.dataloader = data_loader
        self.model_save_dir = os.path.join(config.save_dir, config.model)
        self.start_epoch = config.start_epoch
        self.num_epochs = config.num_epochs
        self.dataroot = config.dataroot
        self.learning_rate = config.learning_rate
        self.decay_epoch = config.decay_epoch
        self.size = config.size
        self.in_ch = config.in_ch
        self.out_ch = config.out_ch
        self.batch_size = config.batch_size
        self.n_cpu = config.n_cpu
        self.cuda = config.cuda
        self.output_dir = config.output_dir
        self.genA2B_path = config.generator_A2B
        self.genB2A_path = config.generator_A2B

        if mode == 'train':
            self.build_model()
        elif mode == 'test':
            self.load_model()

    def train(self):
        """Training function

        Parameters
        ----------
        epoch : int
            current epoch
        """
        # Inputs & targets memory allocation
        Tensor = torch.cuda.FloatTensor if self.cuda else torch.Tensor
        input_A = Tensor(self.batch_size, self.in_ch, self.size, self.size)
        input_B = Tensor(self.batch_size, self.out_ch, self.size, self.size)
        target_real = Variable(
            Tensor(self.batch_size).fill_(1.0), requires_grad=False)
        target_fake = Variable(
            Tensor(self.batch_size).fill_(0.0), requires_grad=False)

        fake_A_buffer = ReplayBuffer()
        fake_B_buffer = ReplayBuffer()

        logger = LoggerViz(self.num_epochs, len(self.dataloader))

        for epoch in range(self.start_epoch, self.num_epochs+1):
            for i, batch in enumerate(self.dataloader):
                # Set model input
                real_A = Variable(input_A.copy_(batch['A']))
                real_B = Variable(input_B.copy_(batch['B']))

                ###### Generators A2B and B2A ######
                self.optimizer_G.zero_grad()

                # Identity loss
                # G_A2B(B) should equal B if real B is fed
                same_B = self.netG_A2B(real_B)
                loss_identity_B = self.criterion_identity(same_B, real_B)*5.0
                # G_B2A(A) should equal A if real A is fed
                same_A = self.netG_B2A(real_A)
                loss_identity_A = self.criterion_identity(same_A, real_A)*5.0

                # GAN loss
                fake_B = self.netG_A2B(real_A)
                pred_fake = self.netD_B(fake_B)
                loss_GAN_A2B = self.criterion_GAN(pred_fake, target_real)

                fake_A = self.netG_B2A(real_B)
                pred_fake = self.netD_A(fake_A)
                loss_GAN_B2A = self.criterion_GAN(pred_fake, target_real)

                # Cycle loss
                recovered_A = self.netG_B2A(fake_B)
                loss_cycle_ABA = self.criterion_cycle(recovered_A, real_A)*10.0

                recovered_B = self.netG_A2B(fake_A)
                loss_cycle_BAB = self.criterion_cycle(recovered_B, real_B)*10.0

                # Total loss
                loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + \
                    loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
                loss_G.backward()

                self.optimizer_G.step()
                ###################################

                ###### Discriminator A ######
                self.optimizer_D_A.zero_grad()

                # Real loss
                pred_real = self.netD_A(real_A)
                loss_D_real = self.criterion_GAN(pred_real, target_real)

                # Fake loss
                fake_A = fake_A_buffer.push_and_pop(fake_A)
                pred_fake = self.netD_A(fake_A.detach())
                loss_D_fake = self.criterion_GAN(pred_fake, target_fake)

                # Total loss
                loss_D_A = (loss_D_real + loss_D_fake)*0.5
                loss_D_A.backward()

                self.optimizer_D_A.step()
                ###################################

                ###### Discriminator B ######
                self.optimizer_D_B.zero_grad()

                # Real loss
                pred_real = self.netD_B(real_B)
                loss_D_real = self.criterion_GAN(pred_real, target_real)

                # Fake loss
                fake_B = fake_B_buffer.push_and_pop(fake_B)
                pred_fake = self.netD_B(fake_B.detach())
                loss_D_fake = self.criterion_GAN(pred_fake, target_fake)

                # Total loss
                loss_D_B = (loss_D_real + loss_D_fake)*0.5
                loss_D_B.backward()

                self.optimizer_D_B.step()
                ###################################

                # Progress report (http://localhost:8097)
                logger.log({'loss_G': loss_G, 'loss_G_identity': (loss_identity_A + loss_identity_B), 'loss_G_GAN': (loss_GAN_A2B + loss_GAN_B2A),
                            'loss_G_cycle': (loss_cycle_ABA + loss_cycle_BAB), 'loss_D': (loss_D_A + loss_D_B)},
                           images={'real_A': real_A, 'real_B': real_B, 'fake_A': fake_A, 'fake_B': fake_B})

            # Update learning rates
            self.lr_scheduler_G.step()
            self.lr_scheduler_D_A.step()
            self.lr_scheduler_D_B.step()

            self.save_model()

    def build_model(self):
        """Build model

        Parameters
        ----------
        epoch : int
            current epoch
        """

        self.netG_A2B = Generator(self.in_ch, self.out_ch)
        self.netG_B2A = Generator(self.out_ch, self.in_ch)
        self.netD_A = Discriminator(self.in_ch)
        self.netD_B = Discriminator(self.out_ch)

        if self.cuda:
            self.netG_A2B.cuda()
            self.netG_B2A.cuda()
            self.netD_A.cuda()
            self.netD_B.cuda()

        self.netG_A2B.apply(weights_init_normal)
        self.netG_B2A.apply(weights_init_normal)
        self.netD_A.apply(weights_init_normal)
        self.netD_B.apply(weights_init_normal)

        # Lossess
        self.criterion_GAN = torch.nn.MSELoss()
        self.criterion_cycle = torch.nn.L1Loss()
        self.criterion_identity = torch.nn.L1Loss()

        # Optimizers & LR schedulers
        optim = torch.optim.Adam

        self.optimizer_G = optim(itertools.chain(self.netG_A2B.parameters(), self.netG_B2A.parameters()),
                                 lr=self.learning_rate, betas=(0.5, 0.999))
        self.optimizer_D_A = optim(
            self.netD_A.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))
        self.optimizer_D_B = optim(
            self.netD_B.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))

        scheduler = torch.optim.lr_scheduler.LambdaLR
        self.lr_scheduler_G = scheduler(self.optimizer_G, lr_lambda=LambdaLR(
            self.num_epochs, self.start_epoch, self.decay_epoch).step)
        self.lr_scheduler_D_A = scheduler(self.optimizer_D_A, lr_lambda=LambdaLR(
            self.num_epochs, self.start_epoch, self.decay_epoch).step)
        self.lr_scheduler_D_B = scheduler(self.optimizer_D_B, lr_lambda=LambdaLR(
            self.num_epochs, self.start_epoch, self.decay_epoch).step)

    def save_model(self):
        """Save model

        Parameters
        ----------
        epoch : int
            current epoch
        """
        netG_A2B_name = os.path.join(self.model_save_dir, "netG_A2B.pth")
        netG_B2A_name = os.path.join(self.model_save_dir, "netG_B2A.pth")
        netD_A_name = os.path.join(self.model_save_dir, "netD_A.pth")
        netD_B_name = os.path.join(self.model_save_dir, "netD_B.pth")

        # Save models checkpoints
        torch.save(self.netG_A2B.state_dict(), netG_A2B_name)
        torch.save(self.netG_B2A.state_dict(), netG_B2A_name)
        torch.save(self.netD_A.state_dict(), netD_A_name)
        torch.save(self.netD_B.state_dict(), netD_B_name)

    def load_model(self):

        self.netG_A2B = Generator(self.in_ch, self.out_ch)
        self.netG_B2A = Generator(self.out_ch, self.in_ch)

        if self.cuda:
            self.netG_A2B.cuda()
            self.netG_B2A.cuda()

        # Load state dicts
        self.netG_A2B.load_state_dict(torch.load(self.genA2B_path))
        self.netG_B2A.load_state_dict(torch.load(self.genB2A_path))

        # Set model's test mode
        self.netG_A2B.eval()
        self.netG_B2A.eval()

        # Create output dirs if they don't exist
        if not os.path.exists(os.path.join(self.output_dir, "A")):
            os.makedirs(os.path.join(self.output_dir, "A"))
        if not os.path.exists(os.path.join(self.output_dir, "B")):
            os.makedirs(os.path.join(self.output_dir, "B"))

    def test(self):
        # Inputs & targets memory allocation
        Tensor = torch.cuda.FloatTensor if self.cuda else torch.Tensor
        input_A = Tensor(self.batch_size, self.in_ch, self.size, self.size)
        input_B = Tensor(self.batch_size, self.out_ch, self.size, self.size)

        for i, batch in enumerate(self.dataloader):
            # Set model input
            real_A = Variable(input_A.copy_(batch['A']))
            real_B = Variable(input_B.copy_(batch['B']))

            # Generate output
            fake_B = 0.5*(self.netG_A2B(real_A).data + 1.0)
            fake_A = 0.5*(self.netG_B2A(real_B).data + 1.0)

            idx = str(i+1).zfill(4)

            # Save image files
            save_image(fake_A, os.path.join(self.output_dir, "A", idx+".png"))
            save_image(fake_B, os.path.join(self.output_dir, "B", idx+".png"))

            sys.stdout.write('\rGenerated images %04d of %04d' %
                             (i+1, len(self.dataloader)))
