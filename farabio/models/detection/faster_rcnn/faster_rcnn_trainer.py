import os
import ipdb
import time
from collections import namedtuple
import matplotlib
from tqdm import tqdm
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import data as data_
from farabio.core.convnettrainer import ConvnetTrainer
from farabio.models.detection.faster_rcnn.dataset import Dataset, TestDataset, inverse_normalize
from farabio.models.detection.faster_rcnn.faster_rcnn_vgg16 import FasterRCNNVGG16
from farabio.models.detection.faster_rcnn.creator_tool import AnchorTargetCreator, ProposalTargetCreator
import farabio.utils.helpers as helpers
from farabio.utils.losses import Losses
from farabio.utils.metrics import eval_detection_voc
from farabio.utils.meters import ConfusionMeter, AverageValueMeter
from farabio.utils.visdom import FasterRCNNViz, visdom_bbox
# fix for ulimit
# https://github.com/pytorch/pytorch/issues/973#issuecomment-346405667
import resource
#from farabio.models.detection.faster_rcnn.config import opt

# Start train with: python train.py train --env='fasterrcnn' --plot-every=100
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))

matplotlib.use('agg')

LossTuple = namedtuple('LossTuple',
                       ['rpn_loc_loss',
                        'rpn_cls_loss',
                        'roi_loc_loss',
                        'roi_cls_loss',
                        'total_loss'
                        ])


class FasterRCNNTrainer(ConvnetTrainer):
    """FasterRCNNTrainer trainer class. Override with custom methods here.

    The losses include:

    * :obj:`rpn_loc_loss`: The localization loss for \
        Region Proposal Network (RPN).
    * :obj:`rpn_cls_loss`: The classification loss for RPN.
    * :obj:`roi_loc_loss`: The localization loss for the head module.
    * :obj:`roi_cls_loss`: The classification loss for the head module.
    * :obj:`total_loss`: The sum of 4 loss above.

    Args:
        faster_rcnn (model.FasterRCNN):
            A Faster R-CNN model that is going to be trained.
    """

    def define_train_attr(self):
        self._start_epoch = self.config.start_epoch
        self._num_epochs = self.config.num_epochs
        self._has_eval = self.config.has_eval
        self._eval_interval = self.config.eval_interval
        self._test_num = self.config.test_num
        self.rpn_sigma = self.config.rpn_sigma
        self.roi_sigma = self.config.roi_sigma
        self._scale_epoch = self.config.scale_epoch

    def define_model_attr(self):
        self.anchor_target_creator = AnchorTargetCreator()
        self.proposal_target_creator = ProposalTargetCreator()
        self._backbone = True
        self.load_optimizer = self.config['load_optimizer']
        self._load_path = self.config['load_path']
        self._best_path = None

    def define_log_attr(self):
        self._use_visdom = self.config.use_visdom
        # indicators for training status
        self.rpn_cm = ConfusionMeter(2)
        self.roi_cm = ConfusionMeter(21)
        self.meters = {k: AverageValueMeter()
                       for k in LossTuple._fields}  # average loss
        self._save_path = self.config.save_path
        self._save_optimizer = self.config.save_optimizer
        self._plot_every = self.config.plot_every

        if self._use_visdom:
            # visdom wrapper
            self.vis = FasterRCNNViz(env=self.config.env)

    def define_misc_attr(self):
        self._mode = self.config.mode

    def get_trainloader(self):
        print('load data')
        self.dataset = Dataset(self.config)
        self.train_loader = data_.DataLoader(self.dataset,
                                             batch_size=1,
                                             shuffle=True, \
                                             # pin_memory=True,
                                             num_workers=self.config.num_workers)

    def get_testloader(self):
        testset = TestDataset(self.config)
        self.test_loader = data_.DataLoader(testset,
                                            batch_size=1,
                                            num_workers=self.config.test_num_workers,
                                            shuffle=False,
                                            pin_memory=True
                                            )

    def build_model(self):
        self.faster_rcnn = FasterRCNNVGG16()
        print('model construct completed')

        if self.config.cuda:
            self.faster_rcnn.cuda()

        # target creator create gt_bbox gt_label etc as training targets.
        self.loc_normalize_mean = self.faster_rcnn.loc_normalize_mean
        self.loc_normalize_std = self.faster_rcnn.loc_normalize_std
        self.optimizer = self.faster_rcnn.get_optimizer()

    def load_model(self):
        state_dict = torch.load(self.config.load_path)
        if 'model' in state_dict:
            self.faster_rcnn.load_state_dict(state_dict['model'])
        else:  # legacy way, for backward compatibility
            self.faster_rcnn.load_state_dict(state_dict)
        if 'optimizer' in state_dict and self.load_optimizer:
            self.optimizer.load_state_dict(state_dict['optimizer'])

    def save(self, **kwargs):

        save_dict = dict()

        save_dict['model'] = self.faster_rcnn.state_dict()
        #save_dict['config'] = opt._state_dict()
        save_dict['config'] = helpers.state_dict_from_namespace(self.config)
        save_dict['other_info'] = kwargs
        save_dict['vis_info'] = self.vis.state_dict()

        if self._save_optimizer:
            save_dict['optimizer'] = self.optimizer.state_dict()

        timestr = time.strftime('%m%d%H%M')
        save_path = f"{self._save_path}fasterrcnn_{timestr}"
        for k_, v_ in list(kwargs.items()):
            save_path += f'_{v_}'

        save_dir = os.path.dirname(save_path)
        helpers.makedirs(save_dir)

        print("saving")
        self.save_model(save_dict, save_path)
        self.vis.save([self.vis.env])
        self._load_path = save_path

    def save_model(self, save_dict, save_path):
        torch.save(save_dict, save_path)

    def on_train_start(self):
        if self.config.load_path is not None:
            self.load_model()
            print(f'load pretrained model from {self.config.load_path}')

        self.best_map = 0
        self.lr_ = self.config.lr

    def start_logger(self):
        self.vis.text(self.dataset.db.label_names, win='labels')

    def on_train_epoch_start(self):
        self.reset_meters()
        self.train_epoch_iter = tqdm(enumerate(self.train_loader))

    def on_start_training_batch(self, args):
        self.ii = args[0]
        self.img = args[-1][0]
        self.bbox_ = args[-1][1]
        self.label_ = args[-1][2]
        self.scale = args[-1][3]

    def training_step(self):
        self.scale = helpers.scalar(self.scale)
        self.img, self.bbox, self.label = self.img.cuda(
        ).float(), self.bbox_.cuda(), self.label_.cuda()

        self.optimizer_zero_grad()
        self.forward()
        self.loss_backward()
        self.optimizer_step()
        self.update_meters()

        if (self.ii + 1) % self._plot_every == 0:
            self.visdom_plot()

    def on_evaluate_epoch_start(self):
        self.pred_bboxes, self.pred_labels, self.pred_scores = list(), list(), list()
        self.gt_bboxes, self.gt_labels, self.gt_difficults = list(), list(), list()

        self.valid_epoch_iter = tqdm(enumerate(self.test_loader))

    def on_evaluate_batch_start(self, args):
        self.ii = args[0]
        self.imgs = args[-1][0]
        self.sizes = args[-1][1]
        self.gt_bboxes_ = args[-1][2]
        self.gt_labels_ = args[-1][3]
        self.gt_difficults_ = args[-1][4]

    def on_evaluate_epoch_end(self):
        self.eval_result = eval_detection_voc(
            self.pred_bboxes, self.pred_labels, self.pred_scores,
            self.gt_bboxes, self.gt_labels, self.gt_difficults,
            use_07_metric=True)

    def visdom_plot(self):
        if os.path.exists(self.config.debug_file):
            ipdb.set_trace()

        # plot loss
        self.vis.plot_many(self.get_meter_data())

        # plot groud truth bboxes
        ori_img_ = inverse_normalize(helpers.tonumpy(self.img[0]))
        gt_img = visdom_bbox(ori_img_,
                             helpers.tonumpy(self.bbox_[0]),
                             helpers.tonumpy(self.label_[0]))
        self.vis.img('gt_img', gt_img)

        # plot predicti bboxes
        _bboxes, _labels, _scores = self.faster_rcnn.predict(
            [ori_img_], visualize=True)
        pred_img = visdom_bbox(ori_img_,
                               helpers.tonumpy(_bboxes[0]),
                               helpers.tonumpy(_labels[0]).reshape(-1),
                               helpers.tonumpy(_scores[0]))
        self.vis.img('pred_img', pred_img)

        # rpn confusion matrix(meter)
        self.vis.text(
            str(self.rpn_cm.value().tolist()), win='rpn_cm')
        # roi confusion matrix
        self.vis.img('roi_cm', helpers.totensor(
            self.roi_cm.conf, False).float())

    def on_epoch_end(self):
        self.vis.plot('test_map', self.eval_result['map'])
        lr_ = self.faster_rcnn.optimizer.param_groups[0]['lr']
        log_info = 'lr:{}, map:{},loss:{}'.format(str(lr_),
                                                  str(self.eval_result['map']),
                                                  str(self.get_meter_data()))
        self.vis.log(log_info)

        if self.eval_result['map'] > self.best_map:
            self.best_map = self.eval_result['map']
            self.save(best_map=self.best_map)

        if self._epoch == self._scale_epoch:
            self.load_model()
            self.faster_rcnn.scale_lr(self.config.lr_decay)
            lr_ = lr_ * self.config.lr_decay

        if self._epoch == self._num_epochs:
            self.stop_train()

    def evaluate_batch(self, *args):
        sizes = [self.sizes[0][0].item(), self.sizes[1][0].item()]
        pred_bboxes_, pred_labels_, pred_scores_ = self.faster_rcnn.predict(self.imgs, [
            sizes])
        self.gt_bboxes += list(self.gt_bboxes_.numpy())
        self.gt_labels += list(self.gt_labels_.numpy())
        self.gt_difficults += list(self.gt_difficults_.numpy())
        self.pred_bboxes += pred_bboxes_
        self.pred_labels += pred_labels_
        self.pred_scores += pred_scores_

        if self.ii == self._test_num:
            self.exit_trainer()

    def forward(self):
        n = self.bbox.shape[0]
        if n != 1:
            raise ValueError('Currently only batch size 1 is supported.')

        _, _, H, W = self.img.shape
        img_size = (H, W)

        features = self.faster_rcnn.extractor(self.img)

        rpn_locs, rpn_scores, rois, roi_indices, anchor = \
            self.faster_rcnn.rpn(features, img_size, self.scale)

        # Since batch size is one, convert variables to singular form
        bbox = self.bbox[0]
        label = self.label[0]
        rpn_score = rpn_scores[0]
        rpn_loc = rpn_locs[0]
        roi = rois

        # Sample RoIs and forward
        # it's fine to break the computation graph of rois,
        # consider them as constant input
        sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator(
            roi,
            helpers.tonumpy(bbox),
            helpers.tonumpy(label),
            self.loc_normalize_mean,
            self.loc_normalize_std)
        # NOTE it's all zero because now it only support for batch=1 now
        sample_roi_index = torch.zeros(len(sample_roi))
        roi_cls_loc, roi_score = self.faster_rcnn.head(
            features,
            sample_roi,
            sample_roi_index)

        # ------------------ RPN losses -------------------#
        gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(
            helpers.tonumpy(bbox),
            anchor,
            img_size)
        gt_rpn_label = helpers.totensor(gt_rpn_label).long()
        gt_rpn_loc = helpers.totensor(gt_rpn_loc)
        rpn_loc_loss = Losses()._fast_rcnn_loc_loss(
            rpn_loc,
            gt_rpn_loc,
            gt_rpn_label.data,
            self.rpn_sigma)

        # NOTE: default value of ignore_index is -100 ...
        rpn_cls_loss = F.cross_entropy(
            rpn_score, gt_rpn_label.cuda(), ignore_index=-1)
        _gt_rpn_label = gt_rpn_label[gt_rpn_label > -1]
        _rpn_score = helpers.tonumpy(
            rpn_score)[helpers.tonumpy(gt_rpn_label) > -1]
        self.rpn_cm.add(helpers.totensor(_rpn_score, False),
                        _gt_rpn_label.data.long())

        # ------------------ ROI losses (fast rcnn loss) -------------------#
        n_sample = roi_cls_loc.shape[0]
        roi_cls_loc = roi_cls_loc.view(n_sample, -1, 4)
        roi_loc = roi_cls_loc[torch.arange(0, n_sample).long().cuda(),
                              helpers.totensor(gt_roi_label).long()]
        gt_roi_label = helpers.totensor(gt_roi_label).long()
        gt_roi_loc = helpers.totensor(gt_roi_loc)

        roi_loc_loss = Losses()._fast_rcnn_loc_loss(
            roi_loc.contiguous(),
            gt_roi_loc,
            gt_roi_label.data,
            self.roi_sigma)

        roi_cls_loss = nn.CrossEntropyLoss()(roi_score, gt_roi_label.cuda())

        self.roi_cm.add(helpers.totensor(roi_score, False),
                        gt_roi_label.data.long())

        losses = [rpn_loc_loss, rpn_cls_loss, roi_loc_loss, roi_cls_loss]
        losses = losses + [sum(losses)]

        self.all_losses = LossTuple(*losses)

    def optimizer_zero_grad(self):
        self.optimizer.zero_grad()

    def loss_backward(self):
        self.all_losses.total_loss.backward()

    def optimizer_step(self):
        self.optimizer.step()

    ##########################
    # Native methods
    ##########################
    def update_meters(self):
        loss_d = {k: helpers.scalar(v)
                  for k, v in list(self.all_losses._asdict().items())}
        for key, meter in list(self.meters.items()):
            meter.add(loss_d[key])

    def reset_meters(self):
        for key, meter in list(self.meters.items()):
            meter.reset()
        self.roi_cm.reset()
        self.rpn_cm.reset()

    def get_meter_data(self):
        return {k: v.value()[0] for k, v in list(self.meters.items())}


"""Forward Faster R-CNN and calculate losses.

Here are notations used.

* :math:`N` is the batch size.
* :math:`R` is the number of bounding boxes per image.

Currently, only :math:`N=1` is supported.
(~torch.autograd.Variable)
Args:
    imgs : A variable with a batch of images.
    bboxes : A batch of bounding boxes.
        Its shape is :math:`(N, R, 4)`.
    labels : A batch of labels.
        Its shape is :math:`(N, R)`. The background is excluded from
        the definition, which means that the range of the value
        is :math:`[0, L - 1]`. :math:`L` is the number of foreground
        classes.
    scale (float): Amount of scaling applied to
        the raw image during preprocessing.

Returns:
    namedtuple of 5 losses
"""

"""Serialize models include optimizer and other info
return path where the model-file is stored.

Args:
    save_optimizer (bool): whether save optimizer.state_dict().
    save_path (string): where to save model, if it's None, save_path
        is generate using time str and info from kwargs.

Returns:
    save_path(str): the path to save models.
"""
