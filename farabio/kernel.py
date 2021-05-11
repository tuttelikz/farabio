import time
from farabio.core.configs import default_cfgs
from farabio.utils.helpers import EasyDict as edict
from farabio.models.classification.class_trainer import ClassTrainer
from farabio.models.segmentation.unet.unet_trainer import UnetTrainer
from farabio.models.segmentation.attunet.attunet_trainer import AttunetTrainer
from farabio.models.superres.srgan.srgan_trainer import SrganTrainer
from farabio.models.translation.cyclegan.cyclegan_trainer import CycleganTrainer
from farabio.models.detection.yolov3.yolo_trainer import YoloTrainer
from farabio.models.detection.faster_rcnn.faster_rcnn_trainer import FasterRCNNTrainer


models = {
    "classification": {
        ("vgg", "resnet", "preactresnet", "googlenet",
         "densenet", "resnext", "mobilenet", "mobilenet2",
         "dpn92", "shufflenet2", "efficientnet", "regnet",
         "simpledla"): ClassTrainer,
    },
    "segmentation": {
        "unet": UnetTrainer,
        "attunet": AttunetTrainer,
    },
    "superres": {
        "srgan": SrganTrainer,
    },
    "translation": {
        "cyclegan": CycleganTrainer,
    },
    "detection": {
        "yolov3": YoloTrainer,
        "faster_rcnn": FasterRCNNTrainer
    }
}


if __name__ == "__main__":
    itime = time.time()

    # Choose from list
    model = ("classification", "resnet")
    #model = ("segmentation", "unet")

    if model[0] == "classification":
        cfg = default_cfgs[model[0]]
        config = edict(cfg)
        config.arch = model[-1]
        trnr = ClassTrainer(config)
    else:
        cfg = default_cfgs[model[-1]]
        config = edict(cfg)
        trnr = models[model[0]][model[-1]](config)

    if config.mode == 'train':
        trnr.train()

    # if config.mode == 'train':
    #     trnr.train()
    # elif config.mode == 'test':
    #     trnr.test()
    # elif config.mode == 'detect':
    #     assert model == "yolov3", "detect mode works only for yolo!"
    #     trnr.detect_perform()

    # etime = time.time() - itime
    # print(f'Complete in {etime // 60}m {etime % 60: .2f}s')
