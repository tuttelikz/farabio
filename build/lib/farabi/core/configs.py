import argparse

parser = argparse.ArgumentParser()

_arglists = []


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    _arglists.append(arg)
    return arg


def get_config(model=None):
    if model == 'srgan':
        define_srgan()
    elif model == 'unet':
        define_unet()
    elif model == 'yolo':
        define_yolo()
    elif model == 'cyclegan':
        define_cyclegan()

    config, unparsed = parser.parse_known_args()
    return config, unparsed


def define_unet():
    """Defines UNet configuration
    """
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
    data_arg.add_argument("--out_ch", default=1,
                          type=int, help='# channels of output image')

    # Model
    model_arg = add_argument_group('Model')
    model_arg.add_argument("--model", default="unet",
                           type=str, choices=['unet', 'attunet'], help='segmentation model')
    model_arg.add_argument("--model_save_dir", default='/home/data/02_SSD4TB/suzy/models/unet',
                           type=str, help='model save path directory')
    model_arg.add_argument("--save_epoch", default=1,
                           type=int, help='frequency to save model after #epochs')
    model_arg.add_argument("--semantic", default=False,
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
    test_arg.add_argument("--model_load_dir", default='/home/data/02_SSD4TB/suzy/models/unet/unet.pt',
                          type=str, help='test model load directory')
    test_arg.add_argument("--output_img_dir", default='/home/data/02_SSD4TB/suzy/models/unet/output/img',
                          type=str, help='output images load directory')
    test_arg.add_argument("--output_mask_dir", default='/home/data/02_SSD4TB/suzy/models/unet/output/mask',
                          type=str, help='output masks directory')
    test_arg.add_argument("--output_overlay_dir", default="/home/data/02_SSD4TB/suzy/models/unet/output/overlay",
                          type=str, help='output overlay directory')

    # Misc
    misc_arg = add_argument_group('Misc')
    misc_arg.add_argument('--use_tensorboard', default=False,
                          type=bool, help='flag to log into tensorboard')
    misc_arg.add_argument('--use_visdom', default=False,
                          type=bool, help='flag to log into Visdom')
    misc_arg.add_argument('--num_gpu', default=1,
                          type=int, help='use 0 to use all except one')
    misc_arg.add_argument('--cuda', action='store_true',
                          help='use GPU computation')
    misc_arg.add_argument("--nw", default=32,
                          type=int, help='number of cpu threads to use during batch generation')


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

    # Train
    train_arg = add_argument_group('Train')
    train_arg.add_argument("--start_epoch", default=1,
                           type=int, help='starting epoch number')
    train_arg.add_argument("--num_epochs", default=100,
                           type=int, help='train epoch number')
    train_arg.add_argument("--optim", default='adam',
                           type=str, help='optimizer')
    train_arg.add_argument("--mode", default='test', type=str,
                           choices=['train', 'test'], help='mode of learning')
    train_arg.add_argument("--batch_size_train", default=64,
                           type=int, help='training batch size')
    train_arg.add_argument("--batch_size_valid", default=1,
                           type=int, help='validation batch size')

    # Test
    test_arg = add_argument_group('Test')
    test_arg.add_argument("--model_path", default='/home/data/02_SSD4TB/suzy/models/srgan/epochs/netG_epoch_4_1.pth',
                          type=str, help='pretrained model path')
    test_arg.add_argument("--batch_size_test", default=1,
                          type=int, help='test batch size')

    # Model
    model_arg = add_argument_group('Model')
    model_arg.add_argument("--model_save_dir", default='/home/data/02_SSD4TB/suzy/models/srgan',
                           type=str, help='model save path directory')
    model_arg.add_argument("--save_epoch", default=1,
                           type=int, help='frequency to save model after #epochs')
    model_arg.add_argument("--save_csv_epoch", default=1,
                           type=int, help='frequency to save csv after #epochs')

    # Misc
    misc_arg = add_argument_group('Misc')
    misc_arg.add_argument('--cuda', action='store_true',
                          help='use GPU')
    misc_arg.add_argument("--num_workers", default=4,
                          type=int, help='number of workers for dataloader')


def define_cyclegan():
    """Defines CycleGAN configuration
    """
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
    data_arg.add_argument("--output_dir", default='/home/data/02_SSD4TB/suzy/models/cyclegan/output',
                          type=str, help='directory to save output images')
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
    model_arg.add_argument("--save_epoch", default=4,
                           type=int, help='frequency to save model after #epochs')

    # Train
    train_arg = add_argument_group('Train')
    train_arg.add_argument("--start_epoch", default=1,
                           type=int, help='train starting epoch')
    train_arg.add_argument("--num_epochs", default=200,
                           type=int, help='number of epochs of training')
    train_arg.add_argument("--learning_rate", default=0.0002,
                           type=float, help='learning rate type')
    train_arg.add_argument('--decay_epoch',  default=100,
                           type=int, help='epoch to start linearly decaying the learning rate to 0')
    train_arg.add_argument("--mode", default='test',
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
    test_arg.add_argument("--generator_A2B", default="/home/data/02_SSD4TB/suzy/models/cyclegan/netG_A2B.pth",
                          type=str, help='directory of pretrained A2B generator')
    test_arg.add_argument("--generator_B2A", default="/home/data/02_SSD4TB/suzy/models/cyclegan/netG_B2A.pth",
                          type=str, help='directory of pretrained B2A generator')

    # Misc
    misc_arg = add_argument_group('Misc')
    misc_arg.add_argument('--use_tensorboard', default=True,
                          type=bool, help='flag to log into tensorboard')
    misc_arg.add_argument('--num_gpu', default=1,
                          type=int, help='use 0 to use all except one')
    misc_arg.add_argument('--cuda', action='store_true',
                          help='use GPU computation')
    misc_arg.add_argument('--num_workers', type=int, default=8,
                          help='number of cpu threads to use during batch generation')


def define_yolo():
    """Defines YOLO configuration
    """
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
