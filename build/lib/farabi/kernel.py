import time
from farabio.core.configs import get_config
from farabio.models.segmentation.unet.unet_trainer import UnetTrainer
from farabio.models.segmentation.attunet.attunet_trainer import AttunetTrainer
from farabio.models.superres.srgan.srgan_trainer import SrganTrainer
from farabio.models.translation.cyclegan.cyclegan_trainer import CycleganTrainer
from farabio.models.detection.yolov3.yolo_trainer import YoloTrainer
from farabio.models.detection.faster_rcnn.faster_rcnn_trainer import FasterRCNNTrainer


models = {
    "unet": UnetTrainer,
    "attunet": AttunetTrainer,
    "srgan": SrganTrainer,
    "cyclegan": CycleganTrainer,
    "yolov3": YoloTrainer,
    "faster_rcnn": FasterRCNNTrainer
}


if __name__ == "__main__":
    itime = time.time()

    model = "faster_rcnn"
    config, _ = get_config(model)

    trnr = models[model](config)
    if config.mode == 'train':
        trnr.train()
    elif config.mode == 'test':
        trnr.test()
    elif config.mode == 'detect':
        assert model == "yolov3", "detect mode works only for yolo!"
        trnr.detect_perform()

    etime = time.time() - itime
    print(f'Complete in {etime // 60}m {etime % 60: .2f}s')
