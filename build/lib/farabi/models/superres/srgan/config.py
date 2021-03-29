import argparse
import time
from farabi.utils.helpers import makedirs
import os

arg_lists = []
parser = argparse.ArgumentParser(
    description="Train super resolution SRGAN model")


date = time.strftime("%y%m%d", time.localtime())
#model_dir = '/data/02_SSD4TB/suzy/models/srgan'
model_dir = '/home/data/02_SSD4TB/suzy/models/srgan'
model_save_dir = os.path.join(model_dir, "srgan_" + date)
makedirs(model_save_dir)


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

# Argument groups:
# ----------------------------------
# data | model | train | test | misc

# Arguments order:
# --------------------------------------
# name | default | type | choices | help


# Data
data_arg = add_argument_group('Data')
data_arg.add_argument("--crop_size", default=88,
                      type=int, help='training images crop size')
data_arg.add_argument("--upscale_factor", default=4,
                      type=int, choices=[2, 4, 8], help='super resolution upscale factor')

# Train
train_arg = add_argument_group('Train')
train_arg.add_argument("--num_epochs", default=100,
                       type=int, help='train epoch number')
train_arg.add_argument("--optim", default='adam',
                       type=str, help='optimizer')
train_arg.add_argument("--mode", default='test', type=str, choices=['train', 'test'], help='mode of learning')

# Test
test_arg = add_argument_group('Test')
test_arg.add_argument("--model_name", default='/home/data/02_SSD4TB/suzy/models/srgan/epochs/netG_epoch_4_1.pth',
                      type=str, help='pretrained model path')
# /home/DATA_Lia/data_02/DATASET_SUZY/MODELS/superres/srgan_210121/epochs/netG_epoch_4_100.pth

# Model
model_arg = add_argument_group('Model')
model_arg.add_argument("--model_save_dir", default=model_save_dir,
                       type=str, help='model save path directory')

# Misc
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--cuda', action='store_true',
                      help='use GPU')


def get_config(model='srgan'):
    config, unparsed = parser.parse_known_args()
    return config, unparsed
