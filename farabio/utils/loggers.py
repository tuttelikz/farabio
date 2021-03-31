import tensorflow as tf
import tensorboardX as tb
from tensorboardX.summary import Summary
from visdom import Visdom
import time
import sys
import datetime
from farabio.data.transforms import tensor2image
import numpy as np
import torch
import skimage
from skimage.measure import label
from skimage import img_as_ubyte
from farabio.data.imgops import ImgOps


class Logger(object):
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = tf.summary.create_file_writer(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        tf.summary.scalar(name=tag, data=value, step=step)
        self.writer.flush()

    def list_of_scalars_summary(self, tag_value_pairs, step):
        """Log scalar variables."""
        for tag, value in tag_value_pairs:
            tf.summary.scalar(name=tag, data=value, step=step)
        self.writer.flush()


class TensorBoard(object):
    def __init__(self, model_dir):
        self.summary_writer = tb.FileWriter(model_dir)

    def scalar_summary(self, tag, value, step):
        summary = Summary(value=[Summary.Value(tag=tag, simple_value=value)])
        self.summary_writer.add_summary(summary, global_step=step)


class SrganViz():
    def __init__(self, n_epochs, batches_epoch):
        self.viz = Visdom()
        self.n_epochs = n_epochs
        self.batches_epoch = batches_epoch
        self.epoch = 1
        self.batch = 1
        self.prev_time = time.time()
        self.mean_period = 0
        self.losses = {}
        self.loss_windows = {}
        self.image_windows = {}

    def log(self, losses=None, images=None):
        self.mean_period += (time.time() - self.prev_time)
        self.prev_time = time.time()

        sys.stdout.write('\rEpoch %03d/%03d [%04d/%04d] -- ' %
                         (self.epoch, self.n_epochs, self.batch, self.batches_epoch))

        for i, loss_name in enumerate(losses.keys()):
            if loss_name not in self.losses:
                self.losses[loss_name] = losses[loss_name].data.cpu().numpy()
            else:
                self.losses[loss_name] += losses[loss_name].data.cpu().numpy()

            if (i+1) == len(losses.keys()):
                sys.stdout.write('%s: %.4f -- ' %
                                 (loss_name, self.losses[loss_name]/self.batch))
            else:
                sys.stdout.write('%s: %.4f | ' %
                                 (loss_name, self.losses[loss_name]/self.batch))

        batches_done = self.batches_epoch*(self.epoch - 1) + self.batch
        batches_left = self.batches_epoch * \
            (self.n_epochs - self.epoch) + self.batches_epoch - self.batch
        sys.stdout.write('ETA: %s' % (datetime.timedelta(
            seconds=batches_left*self.mean_period/batches_done)))

        # Draw images
        for image_name, tensor in images.items():
            if image_name not in self.image_windows:
                self.image_windows[image_name] = self.viz.image(
                    tensor2image(tensor.data), opts={'title': image_name})
            else:
                self.viz.image(tensor2image(
                    tensor.data), win=self.image_windows[image_name], opts={'title': image_name})

        # End of epoch
        if (self.batch % self.batches_epoch) == 0:
            # Plot losses
            for loss_name, loss in self.losses.items():
                if loss_name not in self.loss_windows:
                    self.loss_windows[loss_name] = self.viz.line(X=np.array([self.epoch]), Y=np.array([loss/self.batch]),
                                                                 opts={'xlabel': 'epochs', 'ylabel': loss_name, 'title': loss_name})
                else:
                    self.viz.line(X=np.array([self.epoch]), Y=np.array(
                        [loss/self.batch]), win=self.loss_windows[loss_name], update='append')
                # Reset losses for next epoch
                self.losses[loss_name] = 0.0

            self.epoch += 1
            self.batch = 1
            sys.stdout.write('\n')
        else:
            self.batch += 1


def output2images(tensor, rawtensor=None):
    images = []
    drug_color = {
        1: (255, 0, 0),
        2: (255, 0, 255),
        3: (0, 0, 255),
        4: (0, 255, 255),
        5: (255, 255, 0),
        6: (0, 255, 0),
        7: (255, 255, 255)
    }

    in_shape = (512, 512, 3)

    pred = torch.sigmoid(tensor)
    pred = (pred > 0.5).bool()

    for i in range(pred.size()[0]):
        img_hey = img_as_ubyte(
            rawtensor[i].permute(1, 2, 0).cpu().numpy())

        bw_img = np.squeeze(pred[i].cpu().numpy())
        label_max_drug = np.argmax(np.count_nonzero(bw_img, axis=(1, 2))[1:])
        bw_img = np.squeeze(bw_img[label_max_drug+1, :, :])
        labels = skimage.measure.label(bw_img, return_num=False)
        largestCC = labels == np.argmax(
            np.bincount(labels.flat, weights=bw_img.flat))

        img_o = np.zeros(in_shape, dtype='uint8')
        img_o[:, :, 0] = drug_color[label_max_drug+1][0]*largestCC
        img_o[:, :, 1] = drug_color[label_max_drug+1][1]*largestCC
        img_o[:, :, 2] = drug_color[label_max_drug+1][2]*largestCC

        img_over = ImgOps.blend_imgs(
            img_hey, img_o, overlap=1, ratio=0.7)

        img_over[:, :, 0][largestCC == 0] = 0
        img_over[:, :, 1][largestCC == 0] = 0
        img_over[:, :, 2][largestCC == 0] = 0

        images.append(img_over.transpose([2, 0, 1]))

    return images


def correct2images(tensor, rawtensor=None):
    images = []
    drug_color = {
        1: (255, 0, 0),
        2: (255, 0, 255),
        3: (0, 0, 255),
        4: (0, 255, 255),
        5: (255, 255, 0),
        6: (0, 255, 0),
        7: (255, 255, 255)
    }

    in_shape = (512, 512, 3)

    for i in range(tensor.size()[0]):
        img_hey = img_as_ubyte(
            rawtensor[i].permute(1, 2, 0).cpu().numpy())

        bw_img = np.squeeze(tensor[i].cpu().numpy())

        count_max = -5

        for ks, ims in drug_color.items():
            if np.count_nonzero(bw_img == ks) > count_max:
                count_max = np.count_nonzero(bw_img == ks)
                label_max_drug = ks

        img_o = np.zeros(in_shape, dtype='uint8')
        img_o[:, :, 0] = drug_color[label_max_drug][0] * \
            (bw_img != 0)  # (bw_img == label_max_drug)
        img_o[:, :, 1] = drug_color[label_max_drug][1] * \
            (bw_img != 0)  # (bw_img == label_max_drug)
        img_o[:, :, 2] = drug_color[label_max_drug][2] * \
            (bw_img != 0)  # (bw_img == label_max_drug)

        img_over = Imgops.blend_imgs(
            img_hey, img_o, overlap=1, ratio=0.7)

        img_over[:, :, 0][bw_img == 0] = 0
        img_over[:, :, 1][bw_img == 0] = 0
        img_over[:, :, 2][bw_img == 0] = 0

        images.append(img_over.transpose([2, 0, 1]))

    return images


class UnetViz():
    def __init__(self, n_epochs, batches_epoch):
        self.viz = Visdom()
        self.n_epochs = n_epochs
        self.batches_epoch = batches_epoch
        self.epoch = 1
        self.batch = 1
        self.prev_time = time.time()
        self.mean_period = 0
        self.losses = {}
        self.loss_windows = {}
        self.image_windows = {}

    def log(self, images=None):
        # for i, loss_name in enumerate(losses.keys()):
        #     if loss_name not in self.losses:
        #         self.losses[loss_name] = losses[loss_name]  # .data[0]
        #     else:
        #         self.losses[loss_name] += losses[loss_name]  # .data[0]

        for image_name, tensor in images.items():
            if image_name not in self.image_windows:
                if image_name == 'raw':
                    self.image_windows[image_name] = self.viz.images(
                        tensor, opts={'title': image_name})
                    raw_tensor = tensor
                elif image_name == 'correct':
                    self.image_windows[image_name] = self.viz.images(
                        correct2images(tensor, raw_tensor.data), opts={'title': image_name})
                elif image_name == 'predicted':
                    self.image_windows[image_name] = self.viz.images(
                        output2images(tensor, raw_tensor.data), opts={'title': image_name})

            elif image_name in self.image_windows:
                if image_name == 'raw':
                    self.viz.images(tensor, win=self.image_windows[image_name], opts={
                                    'title': image_name})
                    raw_tensor = tensor
                elif image_name == 'correct':
                    self.viz.images(correct2images(tensor.data, raw_tensor.data),
                                    win=self.image_windows[image_name], opts={'title': image_name})
                elif image_name == 'predicted':
                    self.viz.images(output2images(tensor.data, raw_tensor.data),
                                    win=self.image_windows[image_name], opts={'title': image_name})

        # for loss_name, loss in self.losses.items():
        #     if loss_name not in self.loss_windows:
        #         self.loss_windows[loss_name] = self.viz.line(X=np.array([self.batch]), Y=np.array([loss]),
        #                                                      opts={'xlabel': 'epochs', 'ylabel': loss_name, 'title': loss_name})
        #     else:
        #         self.viz.line(X=np.array([self.batch]), Y=np.array(
        #             [loss]), win=self.loss_windows[loss_name], update='append')
        #     # Reset losses for next epoch
        #     self.losses[loss_name] = 0.0
        #     self.batch += 1
