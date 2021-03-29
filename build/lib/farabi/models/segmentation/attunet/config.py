import argparse
import time
from farabi.utils.helpers import makedirs
import os

date = time.strftime("%y%m%d", time.localtime())
model_dir = '/home/DATA_Lia/data_02/DATASET_SUZY/MODELS/segmentation'
model_save_dir = os.path.join(model_dir, "attunet_" + date)
makedirs(model_save_dir)

model_load = "/home/DATA_Lia/data_02/DATASET_SUZY/MODELS/segmentation/date_attunet_loss/"
model_load_dir = os.path.join(model_load, "attunet16_0.350_210112_ch2.pt")
output_img_dir = os.path.join(model_load, "output")
makedirs(output_img_dir)

arg_lists = []
parser = argparse.ArgumentParser(description="Train attunet model")


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
data_arg.add_argument("--batch_size", default=4,
                      type=int, help='batch size for trasin')
data_arg.add_argument("--val_split", default=0.2,
                      type=float, help='split ratio')
data_arg.add_argument("--shuffle_data", default=True,
                      type=bool, help='flag to shuffle data')
data_arg.add_argument("--num_workers", default=40,
                      type=int, help='number of processes to load data')
data_arg.add_argument("--balanced", default=True,
                      type=bool, help='Whether data is balanced')
data_arg.add_argument("--in_ch", default=3,
                      type=int, help='channels in')
data_arg.add_argument("--out_ch", default=7,
                      type=int, help='channels out')  # 1

# Model
model_arg = add_argument_group('Model')
model_arg.add_argument("--model", default="attunet",
                       type=str, choices=['unet', 'attunet'], help='segmentation model')
model_arg.add_argument("--model_save_dir", default=model_save_dir,
                       type=str, help='model save path directory')
model_arg.add_argument("--save_epoch", default=1,
                       type=int, help='frequency to save model after #epochs')
model_arg.add_argument("--semantic", default=True,
                       type=bool, help='flag for semantic segmentation')


# Train
train_arg = add_argument_group('Train')
train_arg.add_argument("--mode", default='train',
                       type=str, choices=['train', 'test'], help='mode of learning')
train_arg.add_argument("--device", default='cuda:0',
                       type=str, help='GPU device')
train_arg.add_argument("--learning_rate", default=0.01,
                       type=float, help='learning rate type')
train_arg.add_argument("--random_seed", default=42,
                       type=int, help='random seed')
train_arg.add_argument("--num_epochs", default=100,
                       type=int, help='train epoch number')
train_arg.add_argument("--patience", default=10,
                       type=int, help='patience for early stop')
train_arg.add_argument("--optim", default='adam',
                       type=str, help='learning optimizer')
train_arg.add_argument("--data_parallel", default=True,
                       type=bool, help='flag for multigpu')

# Test
test_arg = add_argument_group('Test')
test_arg.add_argument("--model_load_dir", default=model_load_dir,
                      type=str, help='test model load directory')
test_arg.add_argument("--output_img_dir", default=output_img_dir,
                      type=str, help='output images load directory')

# Misc
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--use_tensorboard', default=True,
                      type=bool, help='flag to log into tensorboard')
misc_arg.add_argument('--num_gpu', default=1,
                      type=int, help='use 0 to use all except one')


def get_config(model='unet'):
    config, unparsed = parser.parse_known_args()
    return config, unparsed
