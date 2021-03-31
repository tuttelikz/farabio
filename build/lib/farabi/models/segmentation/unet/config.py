import argparse
import time
import os
from farabio.utils.helpers import makedirs


arg_lists = []
parser = argparse.ArgumentParser(description="Train unet model")

date = time.strftime("%y%m%d", time.localtime())
model_dir = '/home/DATA_Lia/data_02/DATASET_SUZY/MODELS/segmentation'
model_save_dir = os.path.join(model_dir, "unet_" + date)
# makedirs(model_save_dir)

# model_load = "/home/DATA_Lia/data_02/DATASET_SUZY/MODELS/segmentation/date_unet_0.0259/"
# model_load_dir = os.path.join(model_load, "date_vunet_0.0259.pt")
# output_img_dir = os.path.join(model_load, "output")
model_load = "."
model_load_dir = os.path.join(model_load, "date_vunet_0.0259.pt")
output_img_dir = os.path.join(model_load, "output")
# makedirs(output_img_dir)


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
                      type=int, help='# channels of input image')
data_arg.add_argument("--out_ch", default=8,
                      type=int, help='# channels of output image')

# Model
model_arg = add_argument_group('Model')
model_arg.add_argument("--model", default="unet",
                       type=str, choices=['unet', 'attunet'], help='segmentation model')
model_arg.add_argument("--model_save_dir", default=model_save_dir,
                       type=str, help='model save path directory')
model_arg.add_argument("--save_epoch", default=1,
                       type=int, help='frequency to save model after #epochs')
model_arg.add_argument("--semantic", default=True,
                       type=bool, help='flag for semantic segmentation')


# Train
train_arg = add_argument_group('Train')
train_arg.add_argument("--mode", default='test',
                       type=str, choices=['train', 'test'], help='mode of learning')
train_arg.add_argument("--device", default='cuda:1',
                       type=str, help='GPU device')
train_arg.add_argument("--learning_rate", default=0.01,
                       type=float, help='learning rate type')
train_arg.add_argument("--random_seed", default=42,
                       type=int, help='random seed')
train_arg.add_argument("--start_epoch", default=0,
                       type=int, help='training epoch start')
train_arg.add_argument("--num_epochs", default=100,
                       type=int, help='train epoch number')
train_arg.add_argument("--patience", default=13,
                       type=int, help='patience for early stop')
train_arg.add_argument("--optim", default='adam',
                       type=str, help='learning optimizer')
train_arg.add_argument("--data_parallel", default=False,
                       type=bool, help='flag for multigpu')
# train_arg.add_argument("--data_parallel", default=True,
#                        type=bool, help='flag for multigpu')

# Test
test_arg = add_argument_group('Test')
test_arg.add_argument("--model_load_dir", default=model_load_dir,
                      type=str, help='test model load directory')
test_arg.add_argument("--output_img_dir", default=output_img_dir,
                      type=str, help='output images load directory')
test_arg.add_argument("--output_mask_dir", default=".",
                      type=str, help='output masks directory')
test_arg.add_argument("--output_overlay_dir", default=".",
                      type=str, help='output overlay directory')

# Misc
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--use_tensorboard', default=False,
                      type=bool, help='flag to log into tensorboard')
misc_arg.add_argument('--use_visdom', default=False,
                      type=bool, help='flag to log into Visdom')
misc_arg.add_argument('--num_gpu', default=1,
                      type=int, help='use 0 to use all except one')


def get_config(model='attunet'):
    config, unparsed = parser.parse_known_args()
    return config, unparsed
