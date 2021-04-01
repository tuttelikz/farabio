import os
import random
import colorsys
import subprocess
import numpy as np
import seaborn as sns
from PIL import Image
from collections import OrderedDict
import torch
from torch.autograd import Variable


##########################
# Print
##########################
def dump(obj):
    """Prints every attribute of object

    Parameters
    ----------
    obj : object
        Object of interest
    """
    for attr in dir(obj):
        print("obj.%s = %r" % (attr, getattr(obj, attr)))


##########################
# Paths
##########################
def x(adir, a): return os.path.join(adir, a)
def _sdir(path): return sorted(os.listdir(path))
def _matches(a, img_dir): return [i for i in os.listdir(img_dir) if a in i]


def makedirs(*args):
    for path in args:
        if not os.path.exists(path):
            os.makedirs(path)


def load_classes(path):
    """
    Loads class labels at 'path'
    """
    fp = open(path, "r")
    names = fp.read().split("\n")[:-1]
    return names


def calc_weights(*args):
    """Calculates weights for torch.utils.data.WeightedRandomSampler

    Returns
    -------
    list
        list of weights for each class

    Example
    -------
    >>> wts = calc_weights(iver_,iwr_,ag1_,c59_,cnt_)
    """
    sum_ = 0
    counts_ = []
    for i in range(len(args)):
        sum_ += len(args[i])
        counts_.append(len(args[i]))

    weights = sum_ / torch.Tensor(counts_, dtype=torch.float)
    return weights


##########################
# Buffer
##########################
class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (
            max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))


##########################
# Image
##########################
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.bmp', '.PNG', '.JPG', '.JPEG', '.BMP'])


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


##########################
# Colors
##########################
def equal_colors(n_colors, int_flag=True):
    list_ = sns.color_palette("husl", n_colors)[:]
    if int_flag:
        list_clr = []
        for item in list_:
            list_clr.append(tuple([int(255*x) for x in item[:]]))
    elif not int_flag:
        list_clr = list_
    return list_clr


def _raw_equal_colors(num_colors, int_flag=True):
    colors = []
    for i in np.arange(0., 360., 360. / num_colors):
        hue = i/360.
        lightness = (50 + np.random.rand() * 10)/100.
        saturation = (90 + np.random.rand() * 10)/100.

        rgbs = colorsys.hls_to_rgb(hue, lightness, saturation)

        if int_flag:
            colors.append(tuple([int(255*x) for x in rgbs]))
        elif not int_flag:
            colors.append(rgbs)

    return colors


##########################
# CPU/GPU
##########################
def to_cpu(tensor):
    return tensor.detach().cpu()


def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map


def parallel_state_dict(state_dict):
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        if 'module' not in k:
            k = 'module.'+k
        else:
            k = k.replace('features.module.', 'module.features.')
        new_state_dict[k] = v

    return new_state_dict


def state_dict_from_namespace(cfg):
    return {k: getattr(cfg, k) for k, _ in cfg.__dict__.items()
            if not k.startswith('_')}


##########################
# Convert
##########################
def tonumpy(data):
    if isinstance(data, np.ndarray):
        return data
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()


def totensor(data, cuda=True):
    if isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data)
    if isinstance(data, torch.Tensor):
        tensor = data.detach()
    if cuda:
        tensor = tensor.cuda()
    return tensor


def scalar(data):
    if isinstance(data, np.ndarray):
        return data.reshape(1)[0]
    if isinstance(data, torch.Tensor):
        return data.item()


##########################
# Dicts
##########################
def dict_info(sample_dict):
    """Returns number of elements for each key of dictionary

    Parameters
    ----------
    sample_dict : dict
        Dictionary of interest

    Returns
    -------
    tuple of dicts
        first tuple count of elements, second tuple max count
    """
    max_dict = {}
    count_dict = {}
    max_len = 0
    for key in sample_dict.keys():
        if len(sample_dict[key]) > max_len:
            max_dict['max'] = (key, len(sample_dict[key]))
            max_len = len(sample_dict[key])
        count_dict[key] = len(sample_dict[key])
        print(key, ": ", len(sample_dict[key]))

    print(max_dict['max'])

    return (count_dict, max_dict)


class EasyDict(dict):
    """EasyDict definition from https://github.com/makinacorpus/easydict
    """

    def __init__(self, d=None, **kwargs):
        if d is None:
            d = {}
        if kwargs:
            d.update(**kwargs)
        for k, v in d.items():
            setattr(self, k, v)
        # Class attributes
        for k in self.__class__.__dict__.keys():
            if not (k.startswith('__') and k.endswith('__')) and not k in ('update', 'pop'):
                setattr(self, k, getattr(self, k))

    def __setattr__(self, name, value):
        if isinstance(value, (list, tuple)):
            value = [self.__class__(x)
                     if isinstance(x, dict) else x for x in value]
        elif isinstance(value, dict) and not isinstance(value, self.__class__):
            value = self.__class__(value)
        super(EasyDict, self).__setattr__(name, value)
        super(EasyDict, self).__setitem__(name, value)

    __setitem__ = __setattr__

    def update(self, e=None, **f):
        d = e or dict()
        d.update(f)
        for k in d:
            setattr(self, k, d[k])

    def pop(self, k, d=None):
        delattr(self, k)
        return super(EasyDict, self).pop(k, d)
