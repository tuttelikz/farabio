import os
import time
from farabi.data.dataloader import get_trainloader, get_testloader
from trainer import Trainer
from config import get_config
from farabi.utils.helpers import x, makedirs


if __name__ == "__main__":
    start_time = time.time()

    # Non-semantic
    # base_dir = '/home/DATA_Lia/data_02/DATASET_SUZY/SM_XENOPUS/DATA/Dataset/'
    # models_pth = "/home/DATA_Lia/data_02/DATASET_SUZY/MODELS/segmentation"

    # pth_train, pth_test = x(base_dir, 'Train_200909_all'), x(
    #     base_dir, 'Test_200909_all')
    # pth_train_img, pth_train_lbl = x(pth_train, 'Image'), x(pth_train, 'Label')
    # pth_test_img, pth_test_lbl = x(pth_test, 'Image'), x(pth_test, 'Label')

    # if wanna start with semantic segmentation:
    root_dir = "/home/DATA_Lia/data_02/DATASET_SUZY/SM_XENOPUS/DATA/Dataset/Train_200909_all/"
    pth_train_img = os.path.join(root_dir, "Image-sanmo-bal-aug")
    pth_train_lbl = os.path.join(root_dir, "Mask-sanmo-bal-aug-class")

    config, unparsed = get_config()

    if config.semantic is True:
        date = time.strftime("%y%m%d", time.localtime())
        model_dir = '/home/DATA_Lia/data_02/DATASET_SUZY/MODELS/segmentation'
        model_save_dir = os.path.join(model_dir, "semattunet_" + date)
        makedirs(model_save_dir)

    config.model_save_dir = model_save_dir

    if config.mode == 'train':
        data_loader = get_trainloader(
            config, pth_train_img, pth_train_lbl, augment=False) # True augment
        trnr = Trainer(config, data_loader, mode='train')
        trnr.train()

    elif config.mode == 'test':
        data_loader = get_testloader(config, pth_test_img)
        trnr = Trainer(config, data_loader, mode='test')
        trnr.test()

    time_elapsed = time.time() - start_time

    print(
        f'Complete in {time_elapsed // 60}m {time_elapsed % 60: .2f}s')
