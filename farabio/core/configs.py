def _cfg_unet():
    """Configurations for U-Net
    """
    return {
        # Data
        'batch_size': 4,
        'val_split': 0.2,
        'shuffle_data': True,
        'balanced': True,
        'in_ch': 3,
        'out_ch': 1,
        'shape': 512,
        'data_path': '/home/suzy/notebooks/one-time/stage1_train/',
        # Model
        'save_epoch': 1,
        'semantic': False,
        'model_save_name': "unet.pt",
        'model_save_dir': '/home/data/02_SSD4TB/suzy/models/unet',
        # Train
        'learning_rate': 0.01,
        'start_epoch': 1,
        'num_epochs': 100,
        'has_eval': True,
        'early_stop': True,
        'patience': 13,
        'optim': 'adam',
        # Test
        'model_load_dir': '/home/data/02_SSD4TB/suzy/models/unet/unet.pt',
        'output_img_dir': '/home/data/02_SSD4TB/suzy/models/unet/output/img',
        'output_mask_dir': '/home/data/02_SSD4TB/suzy/models/unet/output/mask',
        'output_overlay_dir': '/home/data/02_SSD4TB/suzy/models/unet/output/overlay',
        # Log
        'use_tensorboard': True,
        'use_visdom': False,
        # Compute
        'num_workers': 32,
        'device': 'cuda',
        'num_gpu': 1,
        'cuda': True,
        'data_parallel': True,
        # Misc
        'random_seed': 42,
        'mode': 'train'
    }


def _cfg_attunet():
    """Configurations for Attention U-Net
    """
    return {
        # Data
        'batch_size': 4,
        'val_split': 0.2,
        'shuffle_data': True,
        'balanced': True,
        'in_ch': 3,
        'out_ch': 1,
        'shape': 512,
        'data_path': '/home/suzy/notebooks/one-time/stage1_train/',
        # Model
        'save_epoch': 1,
        'semantic': False,
        'model_save_name': "attunet.pt",
        'model_save_dir': '/home/data/02_SSD4TB/suzy/models/attunet',
        # Train
        'learning_rate': 0.01,
        'start_epoch': 1,
        'num_epochs': 100,
        'has_eval': True,
        'early_stop': True,
        'patience': 13,
        'optim': 'adam',
        # Test
        'model_load_dir': '/home/data/02_SSD4TB/suzy/models/attunet/attunet.pt',
        'output_img_dir': '/home/data/02_SSD4TB/suzy/models/attunet/output/img',
        'output_mask_dir': '/home/data/02_SSD4TB/suzy/models/attunet/output/mask',
        'output_overlay_dir': '/home/data/02_SSD4TB/suzy/models/attunet/output/overlay',
        # Log
        'use_tensorboard': True,
        'use_visdom': False,
        # Compute
        'num_workers': 32,
        'device': 'cuda',
        'num_gpu': 1,
        'cuda': True,
        'data_parallel': True,
        # Misc
        'random_seed': 42,
        'mode': 'train'
    }


def _cfg_srgan():
    """Configurations for SRGAN
    """
    return {
        # Data
        'crop_size': 88,
        'upscale_factor': 4,
        'train_set': '/home/data/02_SSD4TB/suzy/datasets/public/div2k/DIV2K_train_HR',
        'valid_set': '/home/data/02_SSD4TB/suzy/datasets/public/div2k/DIV2K_valid_HR',
        'test_set': '/home/suzy/images',
        'batch_size_train': 64,
        'batch_size_valid': 1,
        'batch_size_test': 1,
        # Model
        'model_path': '/home/data/02_SSD4TB/suzy/models/srgan/epochs/netG_epoch_4_1.pth',
        'model_save_dir': '/home/data/02_SSD4TB/suzy/models/srgan',
        'save_csv_epoch': 1,
        # Train
        'start_epoch': 1,
        'num_epochs': 100,
        'has_eval': True,
        'optim': 'adam',
        # Log
        'save_epoch': 1,
        # Compute
        'num_workers': 4,
        'cuda': True,
        # Misc
        'mode': 'train'
    }


def _cfg_cyclegan():
    """Configurations for CycleGAN
    """
    return {
        # Data
        'dataroot': '/home/data/02_SSD4TB/suzy/datasets/public/monet2photo/',
        'batch_size': 1,
        'size': 256,
        'in_ch': 3,
        'out_ch': 3,
        'val_split': 0.2,
        'shuffle_data': True,
        'balanced': True,
        # Model
        'model': 'cyclegan',
        'save_dir': '/home/data/02_SSD4TB/suzy/models/cyclegan',
        'generator_A2B': '/home/data/02_SSD4TB/suzy/models/cyclegan/netG_A2B.pth',
        'generator_B2A': '/home/data/02_SSD4TB/suzy/models/cyclegan/netG_B2A.pth',
        # Train
        'start_epoch': 1,
        'num_epochs': 200,
        'has_eval': False,
        'learning_rate': 0.0002,
        'decay_epoch': 100,
        'patience': 10,
        'optim': 'adam',
        # Test
        'output_dir': '/home/data/02_SSD4TB/suzy/models/cyclegan/output',
        # Log
        'use_tensorboard': True,
        'save_epoch': 1,
        # Compute
        'num_workers': 8,
        'cuda': True,
        'num_gpu': 1,
        'device': 'cuda:2',
        # Misc
        'mode': 'train',
        'random_seed': 42
    }


def _cfg_yolov3():
    """Configurations for SRGAN
    """
    return {
        # Data
        'batch_size': 8,
        'data_config': '/home/suzy/gitrepos/tuttelikz/farabio/farabio/models/detection/yolov3/config/coco.data',
        'img_size': 416,
        # Model
        'model_def': '/home/suzy/gitrepos/tuttelikz/farabio/farabio/models/detection/yolov3/config/yolov3.cfg',
        'pretrained_weights': '/home/data/02_SSD4TB/suzy/models/yolov3/weights/darknet53.conv.74',
        'weights_path': '/home/data/02_SSD4TB/suzy/models/yolov3/weights/yolov3.weights',
        # Train
        'num_epochs': 100,
        'gradient_accumulations': 2,
        'compute_map': False,
        'multiscale_training': True,
        'optim': 'adam',
        'econf_thres': 0.5,
        # Test
        'iou_thres': 0.5,
        'conf_thres': 0.001,
        'nms_thres': 0.5,
        'detect': False,
        # Detect
        'image_folder': '/home/data/02_SSD4TB/suzy/datasets/public/coco/images/check',
        'dconf_thres': 0.8,
        'dnms_thres': 0.4,
        'dbatch_size': 1,
        # Log
        'checkpoint_interval': 1,
        'evaluation_interval': 1,
        'logdir': '/home/data/02_SSD4TB/suzy/models/yolov3/logs',
        'chckpt_dir': '/home/data/02_SSD4TB/suzy/models/yolov3/checkpoints',
        'output_dir': '/home/data/02_SSD4TB/suzy/models/yolov3/output',
        # Compute
        'n_cpu': 32,
        'device': 2,
        'data_parallel': True,
        # Misc
        'mode': 'train',
    }


def _cfg_fasterrcnn():
    """Configurations for Faster-RCNN
    """
    return {
        # Data
        'voc_data_dir': '/home/data/02_SSD4TB/suzy/datasets/public/pascalvoc/VOC2007',
        'min_size': 600,
        'max_size': 1000,
        # Model
        'load_path': None,
        'save_path': "/home/data/02_SSD4TB/suzy/models/faster-rcnn/checkpoints/",
        'load_optimizer': True,
        'pretrained_model': "vgg16",
        # Train
        'data': "voc",
        'start_epoch': 1,
        'num_epochs': 14,
        'rpn_sigma': 3.,
        'roi_sigma': 1.,
        'weight_decay': 0.0005,
        'scale_epoch': 9,
        'lr': 1e-3,
        'lr_decay': 0.1,
        'use_adam': False,
        'use_chainer': False,
        'use_drop': False,
        'has_eval': True,
        'eval_interval': 1,
        'debug_file': "/tmp/debugf",
        # Test
        'test_num': 10000,
        # Log
        'use_visdom': True,
        'env': "faster-rcnn",
        'port': 8097,
        'plot_every': 40,
        'save_optimizer': False,
        # Compute
        'num_workers': 8,
        'test_num_workers': 8,
        'cuda': True,
        # Misc
        'mode': 'train'
    }


default_cfgs = {
    'unet': _cfg_unet(),
    'attunet': _cfg_attunet(),
    'srgan': _cfg_srgan(),
    'cyclegan': _cfg_cyclegan(),
    'yolov3': _cfg_yolov3(),
    'faster_rcnn': _cfg_fasterrcnn()
}
