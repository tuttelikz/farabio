import colorsys
import os
import torch
from farabi.utils.losses import bbox_iou, bbox_wh_iou
from torch.autograd import Variable
import random
import seaborn as sns
import numpy as np
import subprocess


def _sdir(path): return sorted(os.listdir(path))
def _rjoin(root, fname): return os.path.join(root, fname)
def _matches(a, img_dir): return [i for i in os.listdir(img_dir) if a in i]


def dump(obj):
    """Prints every attribute of object

    Parameters
    ----------
    obj : object
        Object of interest
    """
    for attr in dir(obj):
        print("obj.%s = %r" % (attr, getattr(obj, attr)))


def x(adir, a): return os.path.join(adir, a)


def to_cpu(tensor):
    return tensor.detach().cpu()


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


def rescale_boxes(boxes, current_dim, original_shape):
    """ Rescales bounding boxes to the original shape """
    orig_h, orig_w = original_shape
    # The amount of padding that was added
    pad_x = max(orig_h - orig_w, 0) * (current_dim / max(original_shape))
    pad_y = max(orig_w - orig_h, 0) * (current_dim / max(original_shape))
    # Image height and width after padding is removed
    unpad_h = current_dim - pad_y
    unpad_w = current_dim - pad_x
    # Rescale bounding boxes to dimension of original image
    boxes[:, 0] = ((boxes[:, 0] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 1] = ((boxes[:, 1] - pad_y // 2) / unpad_h) * orig_h
    boxes[:, 2] = ((boxes[:, 2] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 3] = ((boxes[:, 3] - pad_y // 2) / unpad_h) * orig_h
    return boxes


def build_targets(pred_boxes, pred_cls, target, anchors, ignore_thres):

    BoolTensor = torch.cuda.BoolTensor if pred_boxes.is_cuda else torch.BoolTensor
    FloatTensor = torch.cuda.FloatTensor if pred_boxes.is_cuda else torch.FloatTensor

    nB = pred_boxes.size(0)
    nA = pred_boxes.size(1)
    nC = pred_cls.size(-1)
    nG = pred_boxes.size(2)

    # Output tensors
    obj_mask = BoolTensor(nB, nA, nG, nG).fill_(0)
    noobj_mask = BoolTensor(nB, nA, nG, nG).fill_(1)
    class_mask = FloatTensor(nB, nA, nG, nG).fill_(0)
    iou_scores = FloatTensor(nB, nA, nG, nG).fill_(0)
    tx = FloatTensor(nB, nA, nG, nG).fill_(0)
    ty = FloatTensor(nB, nA, nG, nG).fill_(0)
    tw = FloatTensor(nB, nA, nG, nG).fill_(0)
    th = FloatTensor(nB, nA, nG, nG).fill_(0)
    tcls = FloatTensor(nB, nA, nG, nG, nC).fill_(0)

    # Convert to position relative to box
    target_boxes = target[:, 2:6] * nG
    gxy = target_boxes[:, :2]
    gwh = target_boxes[:, 2:]
    # Get anchors with best iou
    ious = torch.stack([bbox_wh_iou(anchor, gwh) for anchor in anchors])
    best_ious, best_n = ious.max(0)
    # Separate target values
    b, target_labels = target[:, :2].long().t()
    gx, gy = gxy.t()
    gw, gh = gwh.t()
    gi, gj = gxy.long().t()
    # Set masks
    obj_mask[b, best_n, gj, gi] = 1
    noobj_mask[b, best_n, gj, gi] = 0

    # Set noobj mask to zero where iou exceeds ignore threshold
    for i, anchor_ious in enumerate(ious.t()):
        noobj_mask[b[i], anchor_ious > ignore_thres, gj[i], gi[i]] = 0

    # Coordinates
    tx[b, best_n, gj, gi] = gx - gx.floor()
    ty[b, best_n, gj, gi] = gy - gy.floor()
    # Width and height
    tw[b, best_n, gj, gi] = torch.log(gw / anchors[best_n][:, 0] + 1e-16)
    th[b, best_n, gj, gi] = torch.log(gh / anchors[best_n][:, 1] + 1e-16)
    # One-hot encoding of label
    tcls[b, best_n, gj, gi, target_labels] = 1
    # Compute label correctness and iou at best anchor
    class_mask[b, best_n, gj, gi] = (
        pred_cls[b, best_n, gj, gi].argmax(-1) == target_labels).float()
    iou_scores[b, best_n, gj, gi] = bbox_iou(
        pred_boxes[b, best_n, gj, gi], target_boxes, x1y1x2y2=False)

    tconf = obj_mask.float()
    return iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf


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

    weights = sum_ / torch.tensor(counts_, dtype=torch.float)
    return weights


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


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.bmp', '.PNG', '.JPG', '.JPEG', '.BMP'])


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


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
