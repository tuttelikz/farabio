import torch.nn.functional as F
import torch
import torch.nn as nn
from torchvision.models.vgg import vgg16
from tqdm import tqdm
import numpy as np
from torch.autograd import Variable
try:
    from itertools import  ifilterfalse
except ImportError: # py3k
    from itertools import  filterfalse as ifilterfalse


class DiceLoss(nn.Module):
    r"""The Dice coefficient, or Dice-Sørensen coefficient.
    
    It is a common metric for pixel segmentation that can also be modified to act as a loss function.

    .. math::

        D S C=\frac{2|X \cap Y|}{|X|+|Y|}

    Examples
    ----------
    >>> dice_loss = DiceLoss()
    >>> dice_loss(outputs, targets)
    """

    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

class DiceBCELoss(nn.Module):
    r"""Dice combined with BCE
    
    This loss combines Dice loss with the standard binary cross-entropy (BCE) loss 
    that is generally the default for segmentation models. Combining the two methods 
    allows for some diversity in the loss, while benefitting from the stability of BCE. 
    The equation for multi-class BCE by itself will be familiar to anyone who has studied 
    logistic regression.

    .. math::

        J(\mathbf{w})=\frac{1}{N} \sum_{n=1}^{N} H\left(p_{ n}, q_{n}\right)=-\frac{1}{N} \sum_{n=1}^{N}\left[y_{n} \log \hat{y}_{n}+\left(1-y_{n}\right) \log \left(1-\hat{y}_{n}\right)\right]

    Examples
    ----------
    >>> dice_bce_loss = DiceBCELoss()
    >>> dice_bce_loss(outputs, targets)
    """

    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        bce_weight = 0.5
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        loss_final = BCE * bce_weight + dice_loss * (1 - bce_weight)
        return loss_final

class IoULoss(nn.Module):
    r"""The IoU metric, or Jaccard Index.
    
    It is similar to the Dice metric and is calculated 
    as the ratio between the overlap of the positive instances between two sets, 
    and their mutual combined values.

    .. math::
    
        J(A, B)=\frac{|A \cap B|}{|A \cup B|}=\frac{|A \cap B|}{|A|+|B|-|A \cap B|}

    Examples
    ----------
    >>> iou_loss = IoULoss()
    >>> iou_loss(outputs, targets)
    """

    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)
                
        return 1 - IoU

class FocalLoss(nn.Module):
    r"""Focal Loss from [RetinaNet]_

    Notes
    -------
    .. math::

        \mathrm{FL}\left(p_{t}\right)=-\alpha_{t}\left(1-p_{t}\right)^{\gamma} \log \left(p_{t}\right)

    where:
    :math:`p_t` is the model's estimated probability for each class.

    It was introduced by Facebook AI Research in 2017 
    to combat extremely imbalanced datasets where positive cases were relatively rare.

    Figure excerpt from [amaarora]_:

    .. image:: ../imgs/focal_loss.png
        :width: 400

    With the help of hyperparameters, :math:`\alpha` and :math:`\gamma`.

    The focusing parameter :math:`\gamma` smoothly adjusts the rate at which easy examples are down-weighted. When :math:`\gamma = 0`, 
    focal loss is equivalent to categorical cross-entropy, and as :math:`\gamma` is increased, the effect of the modulating factor 
    is likewise increased (:math:`\gamma = 2` works best in experiments).

    And, :math:`\alpha` is a weighting factor. If the :math:`\alpha = 1`, then class 1 and class 0 (in binary case)
    have same weights, so :math:`\alpha` balances the importance of positive/negative examples in this way.

    Examples
    ----------
    >>> focal_loss = FocalLoss()
    >>> focal_loss(outputs, targets)

    References
    ---------------
    .. [RetinaNet] https://arxiv.org/abs/1708.02002  
    .. [amaarora] https://amaarora.github.io/2020/06/29/FocalLoss.html
    """

    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=0.8, gamma=0.2, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #first compute binary cross-entropy 
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE
                       
        return focal_loss

class TverskyLoss(nn.Module):
    r"""Tversky loss from [1]_

    Notes
    -------
    .. math::

        \mathrm{S}(P, G, \alpha ; \beta)=\frac{|P G|}{|P G|+\alpha|P G|+\beta|G P|}

    where:
       - :math:`P` and :math:`G` are the predicted and ground truth binary labels.
       - :math:`\alpha` and :math:`\beta` control the magnitude of the penalties for FPs and FNs, respectively.
    
    Notes:
       - :math:`\alpha = \beta = 0.5` => dice coeff
       - :math:`\alpha = \beta = 1` => tanimoto coeff
       - :math:`\alpha + \beta = 1` => F beta coeff

    Examples
    ----------
    >>> tversky_loss = TverskyLoss()
    >>> tversky_loss(outputs, targets)

    References
    ---------------
    .. [1] https://arxiv.org/abs/1706.05721
    """

    def __init__(self, weight=None, size_average=True):
        super(TverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=0.5, beta=0.5):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
       
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        
        return 1 - Tversky

class FocalTverskyLoss(nn.Module):
    r"""A variant on the Tversky loss.
    
    It also includes the gamma modifier from Focal Loss from [1]_.
    
    Examples
    ----------
    >>> focal_tversky_loss = FocalTverskyLoss()
    >>> focal_tversky_loss(outputs, targets)

    References
    ---------------
    .. [1] https://arxiv.org/abs/1810.07842
    """


    def __init__(self, weight=None, size_average=True):
        super(FocalTverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=0.5, beta=0.5, gamma=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
        
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        FocalTversky = (1 - Tversky)**gamma
                       
        return FocalTversky

# --------------------------- HELPER FUNCTIONS ---------------------------
def isnan(x):
    return x != x

def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc

#PyTorch
def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = (labels != ignore)
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels

def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1: # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard

def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        loss = mean(lovasz_hinge_flat(*flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore))
                          for log, lab in zip(logits, labels))
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss

def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.
    signs = 2. * labels.float() - 1.
    errors = (1. - logits * Variable(signs))
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), Variable(grad))
    return loss

#=====
#Multi-class Lovasz loss
#=====

def lovasz_softmax(probas, labels, classes='present', per_image=False, ignore=None):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
              Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
    """
    if per_image:
        loss = mean(lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes)
                          for prob, lab in zip(probas, labels))
    else:
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore), classes=classes)
    return loss


def lovasz_softmax_flat(probas, labels, classes='present'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    if probas.numel() == 0:
        # only void pixels, the gradients should be 0
        return probas * 0.
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float() # foreground for class c
        if (classes is 'present' and fg.sum() == 0):
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = (Variable(fg) - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
    return mean(losses)

class LovaszHingeLoss(nn.Module):
    r"""The Lovász-Softmax loss.
    
    A tractable surrogate for the optimization 
    of the intersection-over-union measure in neural networks from [1]_.

    Examples
    ---------------
    >>> lovasz_hinge_loss = LovaszHingeLoss()
    >>> lovasz_hinge_loss(outputs, targets)

    References
    ---------------
    .. [1] https://arxiv.org/abs/1705.08790
    """

    def __init__(self, weight=None, size_average=True):
        super(LovaszHingeLoss, self).__init__()

    def forward(self, inputs, targets):
        inputs = F.sigmoid(inputs)    
        Lovasz = lovasz_hinge(inputs, targets, per_image=False)                       
        return Lovasz

class Losses:
    """Losses class combines BCE and Dice loss

    Attributes
    ----------
    imgpath : str
        image path of interest
    fsize : str
        file size in human readable format
    img : numpy.array
        image
    h : int
        height
    w : int
        width
    ch : int
        channels
    pmax : int | float
        max pixel value
    pmin : int | float
        min pixel value
    img_r : numpy.array
        red channel image
    img_g : numpy.array
        green channel image
    img_b : numpy.array
        blue channel image

    Methods
    -------
    bce_loss(self, pred, target)
        Returns binary cross-entropy loss
    dice_loss(self, pred, target, smooth=1.)
        Returns dice loss
    slice_img(img, slices, orien)
        Slices image into pieces
    pad_img(self, droi, simg=None)
        Pads image into predefined ROI size
    approx_bcg(self, channel='blue')
        Approximates background image
    blend_img(self, ref_, overlap=0.2, ratio=0.5)
        Blends two images taking the overage of overlap area
    mask_img(self, bw)
        Masks image
    profile_img(self, pt1, pt2)
        Plots image x/y profile between two coordinates
    """

    def bce_loss(self, pred, target):
        """Returns binary cross-entropy loss

        Parameters
        ----------
        pred : torch.Tensor
            predictions tensor array
        target : torch.Tensor
            target tensor array

        Returns
        -------
        torch.Tensor
            binary cross-entropy loss
        """
        bce = F.binary_cross_entropy_with_logits(pred, target)
        return bce

    def dice_loss(self, pred, target, smooth=1.):
        """Returns dice loss

        Parameters
        ----------
        pred : torch.Tensor
            predictions tensor array
        target : torch.Tensor
            target tensor array
        smooth : float, optional
            smoothening coefficient, by default 1.

        Returns
        -------
        torch.Tensor
            dice loss
        """
        pred = pred.contiguous()
        target = target.contiguous()

        intersection = (pred * target).sum(dim=2).sum(dim=2)
        loss = (1 - ((2. * intersection + smooth) /
                     (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))

        return loss.mean()

    def calc_loss(self, pred, target, bce_weight=0.5):
        """Calculates combined BCE and Dice losses

        Parameters
        ----------
        pred : torch.Tensor
            predictions tensor array
        target : torch.Tensor
            target tensor array
        bce_weight : float, optional
            weight to give for bce loss, by default 0.5

        Returns
        -------
        torch.Tensor
            bce and dice loss combined
        """
        bce = self.bce_loss(pred, target)
        dice = self.dice_loss(torch.sigmoid(pred), target)

        loss = bce * bce_weight + dice * (1 - bce_weight)

        return loss

    def cat_loss(self, pred, target):
        """[summary]

        Parameters
        ----------
        pred : [type]
            [description]
        target : [type]
            [description]

        Returns
        -------
        [type]
            [description]
        """
        loss = torch.nn.CrossEntropyLoss()
        categ_loss = loss(pred, target)
        return categ_loss

    def extract_loss(self, pred, target, device=True, cat_weight=0.5):
        """Calculates combined categorical and Dice losses

        Parameters
        ----------
        pred : torch.Tensor
            predictions tensor array
        target : torch.Tensor
            target tensor array
        bce_weight : float, optional
            weight to give for bce loss, by default 0.5

        Returns
        -------
        torch.Tensor
            bce and dice loss combined
        """
        categ = self.cat_loss(pred, target)

        # coefficient of max label
        pred = torch.sigmoid(pred)
        coef_tensor = torch.argmax(pred.sum(2).sum(2)[:, 1:], dim=1)

        # empty tensor
        ttensor = torch.zeros(target.unsqueeze(1).size(),
                              dtype=torch.float32)  # [4, 1, 512, 512]
        # this was previously torch.device to set on same gpu
        #ttensor = ttensor.to(device=device)
        if device:
            ttensor = ttensor.cuda()

        # get the mask of max label
        for ij in range(len(coef_tensor)):
            ttensor[ij, :, :, :] = pred[ij, coef_tensor[ij]+1, :, :]

        # dice
        dice = self.dice_loss(ttensor, target.unsqueeze(1))

        # combined loss
        loss = categ * cat_weight + dice * (1 - cat_weight)

        return loss

    def _smooth_l1_loss(self, x, t, in_weight, sigma):
        sigma2 = sigma ** 2
        diff = in_weight * (x - t)
        abs_diff = diff.abs()
        flag = (abs_diff.data < (1. / sigma2)).float()
        y = (flag * (sigma2 / 2.) * (diff ** 2) +
            (1 - flag) * (abs_diff - 0.5 / sigma2))
        return y.sum()

    def _fast_rcnn_loc_loss(self, pred_loc, gt_loc, gt_label, sigma):
        in_weight = torch.zeros(gt_loc.shape).cuda()
        # Localization loss is calculated only for positive rois.
        # NOTE:  unlike origin implementation,
        # we don't need inside_weight and outside_weight, they can calculate by gt_label
        in_weight[(gt_label > 0).view(-1, 1).expand_as(in_weight).cuda()] = 1
        loc_loss = self._smooth_l1_loss(pred_loc, gt_loc, in_weight.detach(), sigma)
        # Normalize by total number of negtive and positive rois.
        # ignore gt_label==-1 for rpn_loss
        loc_loss /= ((gt_label >= 0).sum().float())
        return loc_loss


class GeneratorLoss(nn.Module):
    """Generator loss from VGG16
    """

    def __init__(self):
        super(GeneratorLoss, self).__init__()
        vgg = vgg16(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()
        self.tv_loss = TVLoss()

    def forward(self, out_labels, out_images, target_images):
        # Adversarial Loss
        adversarial_loss = torch.mean(1 - out_labels)
        # Perception Loss
        perception_loss = self.mse_loss(self.loss_network(
            out_images), self.loss_network(target_images))
        # Image Loss
        image_loss = self.mse_loss(out_images, target_images)
        # TV Loss
        tv_loss = self.tv_loss(out_images)
        return image_loss + 0.001 * adversarial_loss + 0.006 * perception_loss + 2e-8 * tv_loss


class TVLoss(nn.Module):
    """Tversky loss
    """

    def __init__(self, tv_loss_weight=1):
        """[summary]

        Parameters
        ----------
        tv_loss_weight : int, optional
            [description], by default 1
        """
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        """[summary]

        Parameters
        ----------
        x : [type]
            [description]

        Returns
        -------
        [type]
            [description]
        """
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        """[summary]

        Parameters
        ----------
        t : [type]
            [description]

        Returns
        -------
        [type]
            [description]
        """
        return t.size()[1] * t.size()[2] * t.size()[3]


def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in tqdm(unique_classes, desc="Computing AP"):
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum()
            tpc = (tp[i]).cumsum()

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1])

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype("int32")


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def get_batch_statistics(outputs, targets, iou_threshold):
    """ Compute true positives, predicted scores and predicted labels per sample """
    batch_metrics = []
    for sample_i in range(len(outputs)):

        if outputs[sample_i] is None:
            continue

        output = outputs[sample_i]
        pred_boxes = output[:, :4]
        pred_scores = output[:, 4]
        pred_labels = output[:, -1]

        true_positives = np.zeros(pred_boxes.shape[0])

        annotations = targets[targets[:, 0] == sample_i][:, 1:]
        target_labels = annotations[:, 0] if len(annotations) else []
        if len(annotations):
            detected_boxes = []
            target_boxes = annotations[:, 1:]

            for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):

                # If targets are found break
                if len(detected_boxes) == len(annotations):
                    break

                # Ignore if label is not one of the target labels
                if pred_label not in target_labels:
                    continue

                iou, box_index = bbox_iou(
                    pred_box.unsqueeze(0), target_boxes).max(0)
                if iou >= iou_threshold and box_index not in detected_boxes:
                    true_positives[pred_i] = 1
                    detected_boxes += [box_index]
        batch_metrics.append([true_positives, pred_scores, pred_labels])
    return batch_metrics


def bbox_wh_iou(wh1, wh2):
    wh2 = wh2.t()
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
    return inter_area / union_area


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,
                                          0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,
                                          0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.4):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """

    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])
    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        image_pred = image_pred[image_pred[:, 4] >= conf_thres]
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Object confidence times class confidence
        score = image_pred[:, 4] * image_pred[:, 5:].max(1)[0]
        # Sort by it
        image_pred = image_pred[(-score).argsort()]
        class_confs, class_preds = image_pred[:, 5:].max(1, keepdim=True)
        detections = torch.cat(
            (image_pred[:, :5], class_confs.float(), class_preds.float()), 1)
        # Perform non-maximum suppression
        keep_boxes = []
        while detections.size(0):
            large_overlap = bbox_iou(detections[0, :4].unsqueeze(
                0), detections[:, :4]) > nms_thres
            label_match = detections[0, -1] == detections[:, -1]
            # Indices of boxes with lower confidence scores, large IOUs and matching labels
            invalid = large_overlap & label_match
            weights = detections[invalid, 4:5]
            # Merge overlapping bboxes by order of confidence
            detections[0, :4] = (
                weights * detections[invalid, :4]).sum(0) / weights.sum()
            keep_boxes += [detections[0]]
            detections = detections[~invalid]
        if keep_boxes:
            output[image_i] = torch.stack(keep_boxes)

    return output


def xywh2xyxy(x):
    """[summary]

    Parameters
    ----------
    x : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y
