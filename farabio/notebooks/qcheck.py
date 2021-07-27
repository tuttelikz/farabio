import os
import torchvision
from farabio.data.biodatasets import ChestXrayDataset
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np

_path = "/home/data/02_SSD4TB/suzy/datasets/public/chest-xray"
#_path = "/home/data/02_SSD4TB/suzy/datasets/public/testy"

valid_dataset = ChestXrayDataset(root=_path, download=True, mode="val")

#valid_dataset.visualize_dataset()

# #print(valid_dataset.__getitem__(0)[0])
# train_loader = DataLoader(valid_dataset, batch_size=4, shuffle=True)
# inputs, classes = next(iter(train_loader))
# class_names = valid_dataset.classes

# #inputs, classes = next(iter(train_loader))
# out = torchvision.utils.make_grid(inputs)

# def imshow(inp, title=None):
#     inp = inp.numpy().transpose((1, 2, 0))
#     mean = np.array([0.485, 0.456, 0.406])
#     std = np.array([0.229, 0.224, 0.225])
#     inp = std * inp + mean
#     inp = np.clip(inp, 0, 1)
#     fig = plt.gcf()
#     plt.imshow(inp)
#     if title is not None:
#         plt.title(title)
#     plt.pause(0.001)
#     return fig

# imshow(out, title=[class_names[x] for x in classes])

