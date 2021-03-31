import os
import argparse


parser = argparse.ArgumentParser()
_arg_ls = []


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    _arg_ls.append(arg)
    return arg

##########################
# Add model config below
##########################


def define_unet():
    """Defines U-Net configuration
    """
    # Data
    data_arg = add_argument_group('Data')
    data_arg.add_argument("--batch_size", default=4,
                          type=int, help='batch size for training')
    data_arg.add_argument("--val_split", default=0.2,
                          type=float, help='split ratio')
    data_arg.add_argument("--shuffle_data", default=True,
                          type=bool, help='flag to shuffle data')
    data_arg.add_argument("--balanced", default=True,
                          type=bool, help='whether data is balanced')
    data_arg.add_argument("--in_ch", default=3,
                          type=int, help='# channels of input image')
    data_arg.add_argument("--out_ch", default=1,
                          type=int, help='# channels of output image')
    data_arg.add_argument("--shape", default=512,
                          type=int, help='dimension of image (square)')
    data_arg.add_argument("--data_path", default='/home/suzy/notebooks/one-time/stage1_train/',
                          type=str, help='path to data directory')

    # Model
    model_arg = add_argument_group('Model')
    model_arg.add_argument("--save_epoch", default=1,
                           type=int, help='frequency to save model after #epochs')
    model_arg.add_argument("--semantic", default=False,
                           type=bool, help='flag for semantic segmentation')
    model_arg.add_argument("--model_save_name", default="unet.pt",
                           type=str, help='segmentation model')
    model_arg.add_argument("--model_save_dir", default='/home/data/02_SSD4TB/suzy/models/unet',
                           type=str, help='model save path directory')

    # Train
    train_arg = add_argument_group('Train')
    train_arg.add_argument("--learning_rate", default=0.01,
                           type=float, help='learning rate')
    train_arg.add_argument("--start_epoch", default=1,
                           type=int, help='training epoch start')
    train_arg.add_argument("--num_epochs", default=100,
                           type=int, help='training epochs number')
    train_arg.add_argument("--has_eval", default=True,
                           type=bool, help='has evaluation loop')
    train_arg.add_argument("--early_stop", default=True,
                           type=bool, help='early stop')
    train_arg.add_argument("--patience", default=13,
                           type=int, help='patience for early stop')
    train_arg.add_argument("--optim", default='adam',
                           type=str, help='type of optimizer')

    # Test
    test_arg = add_argument_group('Test')
    test_arg.add_argument("--model_load_dir", default='/home/data/02_SSD4TB/suzy/models/unet/unet.pt',
                          type=str, help='path for pretrained model')
    test_arg.add_argument("--output_img_dir", default='/home/data/02_SSD4TB/suzy/models/unet/output/img',
                          type=str, help='path to save output images')
    test_arg.add_argument("--output_mask_dir", default='/home/data/02_SSD4TB/suzy/models/unet/output/mask',
                          type=str, help='path to save mask images')
    test_arg.add_argument("--output_overlay_dir", default="/home/data/02_SSD4TB/suzy/models/unet/output/overlay",
                          type=str, help='path to save image overlay with masks')

    # Log
    log_arg = add_argument_group('Log')
    log_arg.add_argument('--use_tensorboard', default=True,
                         type=bool, help='flag to use tensorboard')
    log_arg.add_argument('--use_visdom', default=False,
                         type=bool, help='flag to use visdom')

    # Compute
    compute_arg = add_argument_group('Compute')
    compute_arg.add_argument("--num_workers", default=32,
                             type=int, help='number of cpu threads to load batches')
    compute_arg.add_argument("--device", default='cuda',
                             type=str, help='GPU device')
    compute_arg.add_argument('--num_gpu', default=1,
                             type=int, help='use 0 to use all except one')
    compute_arg.add_argument('--cuda', action='store_true',
                             help='use GPU computation')
    compute_arg.add_argument('--data_parallel', action='store_true',
                             help='flag to use parallel GPU models (nn.DataParallel)')

    # Misc
    misc_arg = add_argument_group('Miscellaneous')
    misc_arg.add_argument("--random_seed", default=42,
                          type=int, help='random seed')
    misc_arg.add_argument("--mode", default='train',
                          type=str, choices=['train', 'test'], help='mode of learning')


def define_attunet():
    """Define Attention U-Net configurations here
    """
    # Data
    data_arg = add_argument_group('Data')
    data_arg.add_argument("--batch_size", default=4,
                          type=int, help='batch size for trasin')
    data_arg.add_argument("--val_split", default=0.2,
                          type=float, help='split ratio')
    data_arg.add_argument("--shuffle_data", default=True,
                          type=bool, help='flag to shuffle data')
    data_arg.add_argument("--balanced", default=True,
                          type=bool, help='whether data is balanced')
    data_arg.add_argument("--in_ch", default=3,
                          type=int, help='# channels of input image')
    data_arg.add_argument("--out_ch", default=1,
                          type=int, help='# channels of output image')
    data_arg.add_argument("--shape", default=512,
                          type=int, help='dimension of image (square)')
    data_arg.add_argument("--data_path", default='/home/suzy/notebooks/one-time/stage1_train/',
                          type=str, help='path to data directory')

    # Model
    model_arg = add_argument_group('Model')
    model_arg.add_argument("--save_epoch", default=1,
                           type=int, help='frequency to save model after # epochs')
    model_arg.add_argument("--semantic", default=False,
                           type=bool, help='flag for semantic segmentation')
    model_arg.add_argument("--model_save_name", default="attunet.pt",
                           type=str, help='segmentation model')
    model_arg.add_argument("--model_save_dir", default='/home/data/02_SSD4TB/suzy/models/attunet',
                           type=str, help='model save path directory')

    # Train
    train_arg = add_argument_group('Train')
    train_arg.add_argument("--learning_rate", default=0.01,
                           type=float, help='learning rate')
    train_arg.add_argument("--start_epoch", default=1,
                           type=int, help='training epoch start')
    train_arg.add_argument("--num_epochs", default=100,
                           type=int, help='training epochs number')
    train_arg.add_argument("--has_eval", default=True,
                           type=bool, help='has evaluation loop')
    train_arg.add_argument("--early_stop", default=True,
                           type=bool, help='early stop')
    train_arg.add_argument("--patience", default=13,
                           type=int, help='patience for early stop')
    train_arg.add_argument("--optim", default='adam',
                           type=str, help='type of optimizer')

    # Test
    test_arg = add_argument_group('Test')
    test_arg.add_argument("--model_load_dir", default='/home/data/02_SSD4TB/suzy/models/attunet/attunet.pt',
                          type=str, help='path for pretrained model')
    test_arg.add_argument("--output_img_dir", default='/home/data/02_SSD4TB/suzy/models/attunet/output/img',
                          type=str, help='path to save output images')
    test_arg.add_argument("--output_mask_dir", default='/home/data/02_SSD4TB/suzy/models/attunet/output/mask',
                          type=str, help='path to save mask images')
    test_arg.add_argument("--output_overlay_dir", default="/home/data/02_SSD4TB/suzy/models/attunet/output/overlay",
                          type=str, help='path to save image overlay with masks')

    # Log
    log_arg = add_argument_group('Log')
    log_arg.add_argument('--use_tensorboard', default=True,
                         type=bool, help='flag to use tensorboard')
    log_arg.add_argument('--use_visdom', default=False,
                         type=bool, help='flag to use visdom')

    # Compute
    compute_arg = add_argument_group('Compute')
    compute_arg.add_argument("--num_workers", default=32,
                             type=int, help='number of cpu threads to load batches')
    compute_arg.add_argument("--device", default='cuda',
                             type=str, help='GPU device')
    compute_arg.add_argument('--num_gpu', default=1,
                             type=int, help='use 0 to use all except one')
    compute_arg.add_argument('--cuda', action='store_true',
                             help='use GPU computation')
    compute_arg.add_argument('--data_parallel', action='store_true',
                             help='flag to use parallel GPU models (nn.DataParallel)')

    # Misc
    misc_arg = add_argument_group('Miscellaneous')
    misc_arg.add_argument("--random_seed", default=42,
                          type=int, help='random seed')
    misc_arg.add_argument("--mode", default='train',
                          type=str, choices=['train', 'test'], help='mode of learning')


def define_srgan():
    """Defines SRGAN configuration
    """
    # Data
    data_arg = add_argument_group('Data')
    data_arg.add_argument("--crop_size", default=88,
                          type=int, help='training images crop size')
    data_arg.add_argument("--upscale_factor", default=4,
                          type=int, choices=[2, 4, 8], help='super resolution upscale factor')
    data_arg.add_argument("--train_set", default='/home/data/02_SSD4TB/suzy/datasets/public/div2k/DIV2K_train_HR',
                          type=str, help='train set directory')
    data_arg.add_argument("--valid_set", default='/home/data/02_SSD4TB/suzy/datasets/public/div2k/DIV2K_valid_HR',
                          type=str, help='valid set directory')
    data_arg.add_argument("--test_set", default='/home/suzy/images',
                          type=str, help='test set directory')
    data_arg.add_argument("--batch_size_train", default=64,
                          type=int, help='training batch size')
    data_arg.add_argument("--batch_size_valid", default=1,
                          type=int, help='validation batch size')
    data_arg.add_argument("--batch_size_test", default=1,
                          type=int, help='test batch size')

    # Model
    model_arg = add_argument_group('Model')
    model_arg.add_argument("--model_path", default='/home/data/02_SSD4TB/suzy/models/srgan/epochs/netG_epoch_4_1.pth',
                           type=str, help='pretrained model path')
    model_arg.add_argument("--model_save_dir", default='/home/data/02_SSD4TB/suzy/models/srgan',
                           type=str, help='model save path directory')
    model_arg.add_argument("--save_csv_epoch", default=1,
                           type=int, help='frequency to save csv after #epochs')

    # Train
    train_arg = add_argument_group('Train')
    train_arg.add_argument("--start_epoch", default=1,
                           type=int, help='starting epoch number')
    train_arg.add_argument("--num_epochs", default=100,
                           type=int, help='train epoch number')
    train_arg.add_argument("--has_eval", default=True,
                           type=bool, help='has evaluation loop')
    train_arg.add_argument("--optim", default='adam',
                           type=str, help='optimizer')

    # Test
    test_arg = add_argument_group('Test')

    # Log
    log_arg = add_argument_group('Log')
    log_arg.add_argument("--save_epoch", default=1,
                         type=int, help='frequency to save model after #epochs')

    # Compute
    compute_arg = add_argument_group('Misc')
    compute_arg.add_argument("--num_workers", default=4,
                             type=int, help='number of workers for dataloader')
    compute_arg.add_argument('--cuda', action='store_true',
                             help='use GPU')
    # Misc
    misc_arg = add_argument_group('Misc')
    misc_arg.add_argument("--mode", default='train', type=str,
                          choices=['train', 'test'], help='mode of learning')


def define_cyclegan():
    """Defines CycleGAN configuration
    """
    # Data
    data_arg = add_argument_group('Data')
    data_arg.add_argument('--dataroot',  default='/home/data/02_SSD4TB/suzy/datasets/public/monet2photo/',
                          type=str, help='root directory of the dataset')
    data_arg.add_argument("--batch_size", default=1,
                          type=int, help='batch size for trasin')
    data_arg.add_argument('--size', default=256,
                          type=int, help='size of the data crop (squared assumed)')
    data_arg.add_argument("--in_ch", default=3,
                          type=int, help='number of channels of input data')
    data_arg.add_argument("--out_ch", default=3,
                          type=int, help='number of channels of output data')
    data_arg.add_argument("--val_split", default=0.2,
                          type=float, help='split ratio')
    data_arg.add_argument("--shuffle_data", default=True,
                          type=bool, help='flag to shuffle data')
    data_arg.add_argument("--balanced", default=True,
                          type=bool, help='Whether data is balanced')

    # Model
    model_arg = add_argument_group('Model')
    model_arg.add_argument("--model", default="cyclegan",
                           type=str, help='translation model')
    model_arg.add_argument("--save_dir", default='/home/data/02_SSD4TB/suzy/models/cyclegan',
                           type=str, help='model save path directory')
    model_arg.add_argument("--generator_A2B", default="/home/data/02_SSD4TB/suzy/models/cyclegan/netG_A2B.pth",
                           type=str, help='directory of pretrained A2B generator')
    model_arg.add_argument("--generator_B2A", default="/home/data/02_SSD4TB/suzy/models/cyclegan/netG_B2A.pth",
                           type=str, help='directory of pretrained B2A generator')

    # Train
    train_arg = add_argument_group('Train')
    train_arg.add_argument("--start_epoch", default=1,
                           type=int, help='train starting epoch')
    train_arg.add_argument("--num_epochs", default=200,
                           type=int, help='number of epochs of training')
    train_arg.add_argument("--has_eval", default=False,
                           type=bool, help='has evaluation loop')
    train_arg.add_argument("--learning_rate", default=0.0002,
                           type=float, help='learning rate type')
    train_arg.add_argument('--decay_epoch',  default=100,
                           type=int, help='epoch to start linearly decaying the learning rate to 0')
    train_arg.add_argument("--patience", default=10,
                           type=int, help='patience for early stop')
    train_arg.add_argument("--optim", default='adam',
                           type=str, help='learning optimizer')

    # Test
    test_arg = add_argument_group('Test')
    test_arg.add_argument("--output_dir", default='/home/data/02_SSD4TB/suzy/models/cyclegan/output',
                          type=str, help='directory to save output images')

    # Log
    log_arg = add_argument_group('Log')
    log_arg.add_argument('--use_tensorboard', default=True,
                         type=bool, help='flag to log into tensorboard')
    log_arg.add_argument("--save_epoch", default=1,
                         type=int, help='frequency to save model after #epochs')

    # Compute
    compute_arg = add_argument_group('Compute')
    compute_arg.add_argument('--num_workers', type=int, default=8,
                             help='number of cpu threads to use during batch generation')
    compute_arg.add_argument('--cuda', action='store_true',
                             help='use GPU computation')
    compute_arg.add_argument('--num_gpu', default=1,
                             type=int, help='use 0 to use all except one')
    compute_arg.add_argument("--device", default='cuda:2',
                             type=str, help='GPU device')

    # Misc
    misc_arg = add_argument_group('Misc')
    misc_arg.add_argument("--mode", default='test',
                          type=str, choices=['train', 'test'], help='mode of learning')
    train_arg.add_argument("--random_seed", default=42,
                           type=int, help='random seed')


def define_yolo():
    # Data
    data_arg = add_argument_group('Data')
    data_arg.add_argument("--batch_size", default=8,
                          type=int, help='batch size for training')
    data_arg.add_argument("--data_config", default="/home/suzy/gitrepos/TBL-UNIST/tbl-ai/farabio/models/detection/yolov3/config/coco.data",
                          type=str, help="path to data config file")
    data_arg.add_argument("--img_size", default=416,
                          type=int, help="size of each image dimension")

    # Model
    model_arg = add_argument_group('Model')
    model_arg.add_argument("--model_def", default="/home/suzy/gitrepos/TBL-UNIST/tbl-ai/farabio/models/detection/yolov3/config/yolov3.cfg",
                           type=str, help="path to model definition file")
    model_arg.add_argument("--pretrained_weights", default="/home/data/02_SSD4TB/suzy/models/yolov3/weights/darknet53.conv.74",
                           type=str, help="if specified starts from checkpoint model")
    model_arg.add_argument("--weights_path", default="/home/data/02_SSD4TB/suzy/models/yolov3/weights/yolov3.weights",
                           type=str, help="path to weights file")

    # Train
    train_arg = add_argument_group('Train')
    train_arg.add_argument("--num_epochs", default=100,
                           type=int, help='train epoch number')
    train_arg.add_argument("--gradient_accumulations", default=2,
                           type=int, help="number of gradient accums before step")
    train_arg.add_argument("--compute_map", default=False,
                           type=bool, help="if True computes mAP every tenth batch")
    train_arg.add_argument("--multiscale_training", default=True,
                           type=bool, help="allow for multi-scale training")
    train_arg.add_argument("--optim", default='adam',
                           type=str, help='learning optimizer')
    train_arg.add_argument("--econf_thres", default=0.5,
                           type=float, help="object confidence threshold during evaluation")

    # Test
    test_arg = add_argument_group('Test')
    test_arg.add_argument("--iou_thres", default=0.5,
                          type=float, help="iou threshold required to qualify as detected")
    test_arg.add_argument("--conf_thres", default=0.001,
                          type=float, help="object confidence threshold during test")
    test_arg.add_argument("--nms_thres", default=0.5,
                          type=float, help="iou thresshold for non-maximum suppression")
    test_arg.add_argument("--detect", default=False,
                          type=bool, help="flag whether bbox detections needed")

    # Detect
    detect_arg = add_argument_group('Detect')
    detect_arg.add_argument("--image_folder", default="/home/data/02_SSD4TB/suzy/datasets/public/coco/images/check",
                            type=str, help="choose sample images directory")
    detect_arg.add_argument("--dconf_thres", default=0.8,
                            type=float, help="object confidence threshold")
    detect_arg.add_argument("--dnms_thres", default=0.4,
                            type=float, help="iou thresshold for non-maximum suppression")
    detect_arg.add_argument("--dbatch_size",  default=1,
                            type=int, help="size of the batches")

    # Log
    log_arg = add_argument_group('Log')
    log_arg.add_argument("--checkpoint_interval", default=1,
                         type=int, help="interval between saving model weights")
    log_arg.add_argument("--evaluation_interval", default=1,
                         type=int, help="interval evaluations on validation set")
    log_arg.add_argument("--logdir", default="/home/data/02_SSD4TB/suzy/models/yolov3/logs",
                         type=str, help="choose log directory")
    log_arg.add_argument("--chckpt_dir", default="/home/data/02_SSD4TB/suzy/models/yolov3/checkpoints",
                         type=str, help="choose checkpoint directory")
    log_arg.add_argument("--output_dir", default="/home/data/02_SSD4TB/suzy/models/yolov3/output",
                         type=str, help="choose output directory")

    # Compute
    compute_arg = add_argument_group('Compute')
    compute_arg.add_argument("--n_cpu", default=32,
                             type=int, help="number of cpu threads to use during batch generation")
    compute_arg.add_argument("--device", default=2,
                             type=int, help="choose GPU device")
    compute_arg.add_argument('--data_parallel', action='store_true',
                             help='flag to use parallel GPU models (nn.DataParallel)')

    # Misc
    misc_arg = add_argument_group('Misc')
    misc_arg.add_argument("--mode", default='train',
                          type=str, choices=['train', 'test', 'detect'], help='mode of learning')


def define_faster_rcnn():
    # Data
    data_arg = add_argument_group('Data')
    data_arg.add_argument(
        "--voc_data_dir", default='/home/data/02_SSD4TB/suzy/datasets/public/pascalvoc/VOC2007')
    data_arg.add_argument("--min_size", default=600,
                          type=int, help='min image resize')
    data_arg.add_argument("--max_size", default=1000,
                          type=int, help='max image resize')

    # Model
    model_arg = add_argument_group('Model')
    model_arg.add_argument("--load_path", default=None)
    model_arg.add_argument("--save_path", default="/home/data/02_SSD4TB/suzy/models/faster-rcnn/checkpoints/",
                           type=str, help="path to save model")
    model_arg.add_argument("--load_optimizer", default=True,
                           type=bool, help="flag to load optimizer")
    model_arg.add_argument("--pretrained_model", default="vgg16",
                           type=str, help="pretrained model for backbone")

    # Train
    train_arg = add_argument_group('Train')
    train_arg.add_argument("--data", default="voc",
                           type=str, help="dataset type")
    train_arg.add_argument("--start_epoch", default=1,
                           type=int, help='train starting epoch')
    train_arg.add_argument("--num_epochs", default=14,
                           type=int, help='number of epochs of training')
    train_arg.add_argument("--rpn_sigma", default=3.,
                           type=float, help="l1_smooth_loss rpn")
    train_arg.add_argument("--roi_sigma", default=1.,
                           type=float, help="l1_smooth_loss roi")
    train_arg.add_argument("--weight_decay", default=0.0005,
                           type=float, help="weight_decay")
    train_arg.add_argument("--scale_epoch", default=9,
                           type=int, help='epoch to start decay')
    train_arg.add_argument("--lr", default=1e-3,
                           type=float, help="lr")
    train_arg.add_argument("--lr_decay", default=0.1,
                           type=float, help="lr_decay")
    train_arg.add_argument("--use_adam", default=False,
                           type=bool, help="flag for adam optimizer")
    train_arg.add_argument("--use_chainer", default=False,
                           type=bool, help="try match everything as chainer")
    train_arg.add_argument("--use_drop", default=False,
                           type=bool, help="use dropout in RoIHead")
    train_arg.add_argument("--has_eval", default=True,
                           type=bool, help="Has evaluation loop")
    train_arg.add_argument("--eval_interval", default=1,
                           type=int, help='number of epochs between evaluations')
    train_arg.add_argument("--debug_file", default="/tmp/debugf",
                           type=str, help="debug_file")

    # Test
    test_arg = add_argument_group('Test')
    train_arg.add_argument("--test_num", default=10000,
                           type=int, help='test number')

    # Detect
    detect_arg = add_argument_group('Detect')

    # Log
    log_arg = add_argument_group('Log')
    log_arg.add_argument("--use_visdom", default=True,
                         type=bool, help="use visdom")
    log_arg.add_argument("--env", default="faster-rcnn",
                         type=str, help="visdom env")
    log_arg.add_argument("--port", default=8097,
                         type=int, help="visdom port")
    log_arg.add_argument("--plot_every", default=40,
                         type=int, help="plot interval batch examples")
    log_arg.add_argument("--save_optimizer", default=False,
                         type=bool, help="save optimizer")

    # Compute
    compute_arg = add_argument_group('Compute')
    compute_arg.add_argument("--num_workers", default=8,
                             type=int, help="number of cpu threads to use during batch generation")
    compute_arg.add_argument("--test_num_workers", default=8,
                             type=int, help="number of cpu threads to use during test batch generation")
    compute_arg.add_argument('--cuda', action='store_true',
                             help='use GPU computation')

    # Misc
    misc_arg = add_argument_group('Misc')
    misc_arg.add_argument("--mode", default='train',
                          type=str, choices=['train', 'test'], help='mode of learning')


model_config = {
    "srgan": define_srgan,
    "cyclegan": define_cyclegan,
    "unet": define_unet,
    "attunet": define_attunet,
    "yolov3": define_yolo,
    "faster_rcnn": define_faster_rcnn
}

COUNT = 0


def get_config(model=None):
    global COUNT
    COUNT += 1
    if COUNT == 1:
        model_config[model]()

    config, unparsed = parser.parse_known_args()
    return config, unparsed
