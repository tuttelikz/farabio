import argparse
import time
from farabi.utils.helpers import makedirs
import os


arg_lists = []
parser = argparse.ArgumentParser(description="Train cyclegan")

date = time.strftime("%y%m%d", time.localtime())
model_dir = '/home/data/02_SSD4TB/suzy/models/cyclegan'
save_dir = os.path.join(model_dir, date)
makedirs(save_dir)
output_dir = os.path.join(model_dir, date, "output")
makedirs(output_dir)

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
data_arg.add_argument("--batch_size", default=1,
                      type=int, help='batch size for trasin')
data_arg.add_argument('--dataroot',  default='/home/data/02_SSD4TB/suzy/datasets/public/monet2photo/',
                      type=str, help='root directory of the dataset')
data_arg.add_argument('--size', default=256,
                      type=int, help='size of the data crop (squared assumed)')
data_arg.add_argument("--in_ch", default=3,
                      type=int, help='number of channels of input data')
data_arg.add_argument("--out_ch", default=3,
                      type=int, help='number of channels of output data')
data_arg.add_argument("--output_dir", default=output_dir,
                      type=str, help='directory to save output images')
data_arg.add_argument("--val_split", default=0.2,
                      type=float, help='split ratio')
data_arg.add_argument("--shuffle_data", default=True,
                      type=bool, help='flag to shuffle data')
data_arg.add_argument("--num_workers", default=40,
                      type=int, help='number of processes to load data')
data_arg.add_argument("--balanced", default=True,
                      type=bool, help='Whether data is balanced')


# Model
model_arg = add_argument_group('Model')
model_arg.add_argument("--model", default="cyclegan",
                       type=str, help='translation model')
model_arg.add_argument("--save_dir", default=save_dir,
                       type=str, help='model save path directory')
model_arg.add_argument("--save_epoch", default=4,
                       type=int, help='frequency to save model after #epochs')


# Train
train_arg = add_argument_group('Train')
train_arg.add_argument("--start_epoch", default=0,
                       type=int, help='train starting epoch')
train_arg.add_argument("--num_epochs", default=200,
                       type=int, help='number of epochs of training')
train_arg.add_argument("--learning_rate", default=0.0002,
                       type=float, help='learning rate type')
train_arg.add_argument('--decay_epoch',  default=100,
                       type=int, help='epoch to start linearly decaying the learning rate to 0')
train_arg.add_argument("--mode", default='train',
                       type=str, choices=['train', 'test'], help='mode of learning')
train_arg.add_argument("--device", default='cuda:2',
                       type=str, help='GPU device')

train_arg.add_argument("--random_seed", default=42,
                       type=int, help='random seed')

train_arg.add_argument("--patience", default=10,
                       type=int, help='patience for early stop')
train_arg.add_argument("--optim", default='adam',
                       type=str, help='learning optimizer')

# Test
test_arg = add_argument_group('Test')
test_arg.add_argument("--generator_A2B", default="/home/DATA_Lia/data_02/DATASET_SUZY/MODELS/translation/cyclegan/210118/netG_A2B.pth",
                      type=str, help='directory of pretrained A2B generator')
test_arg.add_argument("--generator_B2A", default="/home/DATA_Lia/data_02/DATASET_SUZY/MODELS/translation/cyclegan/210118/netG_B2A.pth",
                      type=str, help='directory of pretrained B2A generator')


# Misc
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--use_tensorboard', default=True,
                      type=bool, help='flag to log into tensorboard')
misc_arg.add_argument('--num_gpu', default=1,
                      type=int, help='use 0 to use all except one')
misc_arg.add_argument('--cuda', action='store_true',
                      help='use GPU computation')
misc_arg.add_argument('--n_cpu', type=int, default=8,
                      help='number of cpu threads to use during batch generation')


def get_config(model='cyclegan'):
    config, unparsed = parser.parse_known_args()
    return config, unparsed
