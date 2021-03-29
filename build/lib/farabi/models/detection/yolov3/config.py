import os
import argparse

_base_dir = "/home/DATA_Lia/data_02/DATASET_SUZY/MODELS/detection/yolov3"
def _rjoin(fname): return os.path.join(_base_dir, fname)


_img_dir = "/home/DATA_Lia/data_02/DATASET_SUZY/MO_EAR/detection/samples/ear"
_weights_dir = _rjoin("weights")
_weights_darknet = os.path.join(_weights_dir, "darknet53.conv.74")
_weights_pretr = os.path.join(_weights_dir, "yolov3_ckpt_99_ear.pth")
_logdir = _rjoin("logs")
_output = _rjoin("output")
_chckpt = _rjoin("checkpoints")
arg_lists = []
parser = argparse.ArgumentParser(description="Train detection models")


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
data_arg.add_argument("--batch_size", default=8,
                      type=int, help='batch size for training')
data_arg.add_argument("--data_config", default="config/custom.data",
                      type=str, help="path to data config file")
data_arg.add_argument("--n_cpu", default=8,
                      type=int, help="number of cpu threads to use during batch generation")
data_arg.add_argument("--img_size", default=416,
                      type=int, help="size of each image dimension")

# Model
model_arg = add_argument_group('Model')
model_arg.add_argument("--model_def", default="config/yolov3.cfg",
                       type=str, help="path to model definition file")
model_arg.add_argument("--pretrained_weights", default=_weights_darknet,
                       type=str, help="if specified starts from checkpoint model")

# Train
train_arg = add_argument_group('Train')
train_arg.add_argument("--num_epochs", default=100,
                       type=int, help='train epoch number')
train_arg.add_argument("--gradient_accumulations", default=2,
                       type=int, help="number of gradient accums before step")
train_arg.add_argument("--checkpoint_interval", default=1,
                       type=int, help="interval between saving model weights")
train_arg.add_argument("--evaluation_interval", default=1,
                       type=int, help="interval evaluations on validation set")
train_arg.add_argument("--compute_map", default=False,
                       type=bool, help="if True computes mAP every tenth batch")
train_arg.add_argument("--multiscale_training", default=True,
                       type=bool, help="allow for multi-scale training")
train_arg.add_argument("--mode", default='train',
                       type=str, choices=['train', 'test', 'detect'], help='mode of learning')
train_arg.add_argument("--optim", default='adam',
                       type=str, help='learning optimizer')

# Test
test_arg = add_argument_group('Test')
test_arg.add_argument("--weights_path", default=_weights_pretr,
                      type=str, help="path to weights file")
test_arg.add_argument("--iou_thres", default=0.5,
                      type=float, help="iou threshold required to qualify as detected")
test_arg.add_argument("--conf_thres", default=0.001,
                      type=float, help="object confidence threshold")
test_arg.add_argument("--nms_thres", default=0.5,
                      type=float, help="iou thresshold for non-maximum suppression")
test_arg.add_argument("--detect", default=False,
                      type=bool, help="flag whether bbox detections needed")

# Detect
detect_arg = add_argument_group('Detect')
detect_arg.add_argument("--image_folder", default=_img_dir,
                        type=str, help="choose sample images directory")
detect_arg.add_argument("--dconf_thres", default=0.8,
                        type=float, help="object confidence threshold")
detect_arg.add_argument("--dnms_thres", default=0.4,
                        type=float, help="iou thresshold for non-maximum suppression")
detect_arg.add_argument("--dbatch_size",  default=1,
                        type=int, help="size of the batches")

# Misc
misc_arg = add_argument_group('Misc')
misc_arg.add_argument("--device", default=2,
                      type=int, help="choose GPU device")
misc_arg.add_argument("--logdir", default=_logdir,
                      type=str, help="choose log directory")
misc_arg.add_argument("--chckpt_dir", default=_chckpt,
                      type=str, help="choose checkpoint directory")
misc_arg.add_argument("--output_dir", default=_output,
                      type=str, help="choose output directory")


def get_config(model='yolov3'):
    config, unparsed = parser.parse_known_args()
    return config, unparsed
