import time
import torch
from farabio.data.datasets import ListDataset, ImageFolder
from parsers import parse_data_config
from config import get_config
from trainer import Trainer
from farabio.utils.helpers import load_classes


if __name__ == "__main__":
    start_time = time.time()
    config, unparsed = get_config()

    # Get data configuration
    data_config = parse_data_config(config.data_config)
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])

    # Get dataloader
    train_dataset = ListDataset(train_path, augment=True,
                                multiscale=config.multiscale_training)

    valid_dataset = ListDataset(valid_path, img_size=config.img_size,
                                augment=False, multiscale=False)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.n_cpu,
        pin_memory=True,
        collate_fn=train_dataset.collate_fn,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=1,
        collate_fn=valid_dataset.collate_fn
    )

    if config.mode == 'train':
        trnr = Trainer(config, train_loader, valid_loader,
                       class_names, mode='train')
        trnr.train()

    elif config.mode == 'test':
        trnr = Trainer(config, train_loader, valid_loader,
                       class_names, mode='test')
        trnr.test()

    elif config.mode == 'detect':
        data_loader = torch.utils.data.DataLoader(
            ImageFolder(config.image_folder, img_size=config.img_size),
            batch_size=config.dbatch_size,
            shuffle=False,
            num_workers=config.n_cpu,
        )

        trnr = Trainer(config, train_loader, data_loader,
                       class_names, mode='detect')
        trnr.detect_perform()

    time_elapsed = time.time() - start_time

    print(
        f'Complete in {time_elapsed // 60}m {time_elapsed % 60: .2f}s')
