import os
import time
import datetime
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
from terminaltables import AsciiTable
import torch
from torch.autograd import Variable
from farabio.core.convnettrainer import ConvnetTrainer
from farabio.data.datasets import ListDataset, ImageFolder
from farabio.models.detection.yolov3.darknet import Darknet
from farabio.models.detection.yolov3.parsers import parse_data_config
from farabio.utils.helpers import load_classes, makedirs
from farabio.utils.regul import weights_init_normal
from farabio.utils.tensorboard import Logger
from farabio.utils.losses import get_batch_statistics, non_max_suppression, ap_per_class, xywh2xyxy
from farabio.utils.bboxtools import rescale_boxes

class YoloTrainer(ConvnetTrainer):
    """YoloTrainer trainer class. Override with custom methods here.

    Parameters
    ----------
    ConvnetTrainer : BaseTrainer
        Inherits ConvnetTrainer class
    """

    def get_trainloader(self):
        data_config = parse_data_config(self.data_config)
        self.class_names = load_classes(data_config["names"])

        train_path = data_config["train"]
        train_dataset = ListDataset(train_path, augment=True,
                                    multiscale=self.config.multiscale_training)

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.config.n_cpu,
            pin_memory=True,
            collate_fn=train_dataset.collate_fn,
        )

        if self._mode == 'detect':
            self.valid_loader = torch.utils.data.DataLoader(
                ImageFolder(
                    self.image_folder, img_size=self.img_size),
                batch_size=self.dbatch_size,
                shuffle=False,
                num_workers=self.config.n_cpu,
            )
        elif self._mode == 'train' or self._mode == 'test':
            valid_path = data_config["valid"]
            valid_dataset = ListDataset(valid_path, img_size=self.img_size,
                                        augment=False, multiscale=False)
            self.valid_loader = torch.utils.data.DataLoader(
                valid_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.n_cpu,
                collate_fn=valid_dataset.collate_fn
            )

    def define_data_attr(self):
        self.batch_size = self.config.batch_size
        self.dbatch_size = self.config.dbatch_size
        self.data_config = self.config.data_config
        self.img_size = self.config.img_size
        self.image_folder = self.config.image_folder
        self.checkpoint_interval = self.config.checkpoint_interval

    def define_model_attr(self):
        self.model_def = self.config.model_def

    def define_train_attr(self):
        self._eval_interval = self.config.evaluation_interval
        self.iou_thres = self.config.iou_thres
        self.conf_thres = self.config.conf_thres
        self.econf_thres = self.config.econf_thres
        self.nms_thres = self.config.nms_thres
        self._num_epochs = self.config.num_epochs
        self.chckpt_dir = self.config.chckpt_dir
        self.grad_acc = self.config.gradient_accumulations
        if self.config.optim == 'adam':
            self.optim = torch.optim.Adam

        makedirs(self.config.chckpt_dir)

    def define_test_attr(self):
        self.dconf_thres = self.config.dconf_thres
        self.dnms_thres = self.config.dnms_thres
        self.detect = self.config.detect
        self.output_dir = self.config.output_dir
        makedirs(self.config.output_dir)

    def define_log_attr(self):
        self.metrics = [
            "grid_size",
            "loss",
            "x",
            "y",
            "w",
            "h",
            "conf",
            "cls",
            "cls_acc",
            "recall50",
            "recall75",
            "precision",
            "conf_obj",
            "conf_noobj",
        ]
        self.logger = Logger(self.config.logdir)

    def define_compute_attr(self):
        self.n_cpu = self.config.n_cpu
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    def define_misc_attr(self):
        self._mode = self.config.mode

    def build_model(self):
        # Initiate model
        self.model = Darknet(self.model_def).to(self.device)

        if self._mode == 'train':
            self.model.apply(weights_init_normal)

            pretrained_weights = self.config.pretrained_weights
            if pretrained_weights.endswith(".pth"):
                # Custom Darknet weights
                self.model.load_state_dict(torch.load(pretrained_weights))
            else:
                # Darknet-53 on ImageNet
                self.model.load_darknet_weights(pretrained_weights)

        elif self._mode == 'test':
            weights_path = self.config.weights_path

            if weights_path.endswith(".weights"):
                # Load yolov3 coco weights
                self.model.load_darknet_weights(weights_path)
            else:
                # Load custom data checkpoint weights
                self.model.load_state_dict(torch.load(weights_path))

        elif self._mode == 'detect':
            weights_path = self.config.weights_path

            if weights_path.endswith(".weights"):
                # Load yolov3 coco weights
                self.model.load_darknet_weights(weights_path)
            else:
                # Load custom data checkpoint weights
                self.model.load_state_dict(torch.load(weights_path))

        self.optimizer = self.optim(self.model.parameters())

    def on_train_epoch_start(self):
        self.model.train()
        self.start_time = time.time()
        self.train_epoch_iter = enumerate(self.train_loader)

    def on_start_training_batch(self, args):
        self.batch_i = args[0]
        self.imgs = args[-1][1]
        self.targets = args[-1][2]

    def training_step(self):
        self.batches_done = len(self.train_loader) * self._epoch + self.batch_i

        self.imgs = Variable(self.imgs.to(self.device))
        self.targets = Variable(self.targets.to(self.device),
                                requires_grad=False)

        self.loss, outputs = self.model(self.imgs, self.targets)
        self.loss.backward()

        if self.batches_done % self.grad_acc == 0:
            # Accumulates gradient before each step
            self.optimizer.step()
            self.optimizer.zero_grad()

    def on_end_training_batch(self):
        # ----------------
        #   Log progress
        # ----------------
        log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (
            self._epoch, self._num_epochs, self.batch_i, len(self.train_loader))

        metric_table = [
            ["Metrics", *[f"YOLO Layer {i}" for i in range(len(self.model.yolo_layers))]]]

        # Log metrics at each YOLO layer
        for i, metric in enumerate(self.metrics):
            formats = {m: "%.6f" for m in self.metrics}
            formats["grid_size"] = "%2d"
            formats["cls_acc"] = "%.2f%%"
            row_metrics = [formats[metric] % yolo.metrics.get(
                metric, 0) for yolo in self.model.yolo_layers]
            metric_table += [[metric, *row_metrics]]

            # Tensorboard logging
            tensorboard_log = []
            for j, yolo in enumerate(self.model.yolo_layers):
                for name, metric in yolo.metrics.items():
                    if name != "grid_size":
                        tensorboard_log += [(f"{name}_{j+1}", metric)]

            tensorboard_log += [("loss", self.loss.item())]
            self.logger.list_of_scalars_summary(
                tensorboard_log, self.batches_done)

        log_str += AsciiTable(metric_table).table
        log_str += f"\nTotal loss {self.loss.item()}"

        # Determine approximate time left for epoch
        epoch_batches_left = len(self.train_loader) - (self.batch_i + 1)
        time_left = datetime.timedelta(
            seconds=epoch_batches_left * (time.time() - self.start_time) / (self.batch_i + 1))
        log_str += f"\n---- ETA {time_left}"

        print(log_str)

        self.model.seen += self.imgs.size(0)

    def on_epoch_end(self):
        if self._epoch % self.checkpoint_interval == 0:
            self.save_model()

    def save_model(self):
        torch.save(self.model.state_dict(),
                   os.path.join(f"{self.chckpt_dir}", f"yolov3_ckpt_{self._epoch}.pth"))

    def on_evaluate_epoch_start(self):
        self.model.eval()

        self.Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        self.labels = []
        self.sample_metrics = []  # List of tuples (TP, confs, pred)
        self.valid_epoch_iter = enumerate(
            tqdm(self.valid_loader, desc="Detecting objects"))

    def on_evaluate_batch_start(self, args):
        self.batch_i = args[0]
        self.imgs = args[-1][1]
        self.targets = args[-1][2]
        if self.targets is None:
            super()._next_loop = True

    def evaluate_batch(self, *args):
        # Extract labels
        self.labels += self.targets[:, 1].tolist()
        # Rescale target
        self.targets[:, 2:] = xywh2xyxy(self.targets[:, 2:])
        self.targets[:, 2:] *= self.img_size

        self.imgs = Variable(self.imgs.type(self.Tensor), requires_grad=False)

        with torch.no_grad():
            outputs = self.model(self.imgs)
            outputs = non_max_suppression(
                outputs, conf_thres=self.econf_thres, nms_thres=self.nms_thres)

        self.sample_metrics += get_batch_statistics(outputs,
                                                    self.targets, iou_threshold=self.iou_thres)

    def on_evaluate_epoch_end(self):
        # Concatenate sample statistics
        true_positives, pred_scores, pred_labels = [
            np.concatenate(x, 0) for x in list(zip(*self.sample_metrics))]
        precision, recall, AP, f1, ap_class = ap_per_class(
            true_positives, pred_scores, pred_labels, self.labels)

        evaluation_metrics = [
            ("val_precision", precision.mean()),
            ("val_recall", recall.mean()),
            ("val_mAP", AP.mean()),
            ("val_f1", f1.mean()),
        ]

        self.logger.list_of_scalars_summary(
            evaluation_metrics, self._epoch)

        # Print class APs and mAP
        if self._mode == 'train':
            ap_table = [["Index", "Class name", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, self.class_names[c], "%.5f" % AP[i]]]

            print(AsciiTable(ap_table).table)
            print(f"---- mAP {AP.mean()}")

        elif self._mode == "test":
            print("Average Precisions:")
            for i, c in enumerate(ap_class):
                print(
                    f"+ Class '{c}' ({self.class_names[c]}) - AP: {AP[i]}")
            print(f"mAP: {AP.mean()}")

    def test(self):
        print("Compute mAP...")
        self.evaluate_epoch()

    def detect_perform(self):
        self.get_detections()
        self.plot_bbox()

    def get_detections(self):
        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

        self.imgs = []  # Stores image paths
        self.img_detections = []  # Stores detections for each image index
        # for batch_i, (_, imgs, targets) in enumerate(tqdm(valid_loader, desc="Detecting objects")):
        print("\nPerforming object detection:")
        prev_time = time.time()
        #print("vloader: ", len(self.valid_loader))

        for batch_i, (img_paths, input_imgs) in enumerate(self.valid_loader):
            # Configure input
            input_imgs = Variable(input_imgs.type(Tensor))

            # Get detections
            with torch.no_grad():
                detections = self.model(input_imgs)
                detections = non_max_suppression(
                    detections, self.dconf_thres, self.dnms_thres)

            # Log progress
            current_time = time.time()
            inference_time = datetime.timedelta(
                seconds=current_time - prev_time)
            prev_time = current_time
            print("\t+ Batch %d, Inference Time: %s" %
                  (batch_i, inference_time))

            # Save image and detections
            self.imgs.extend(img_paths)
            self.img_detections.extend(detections)

    def plot_bbox(self):
        # Bounding-box colors
        cmap = plt.get_cmap("tab20b")
        colors = [cmap(i) for i in np.linspace(0, 1, 20)]

        print("\nSaving images:")
        # Iterate through images and save plot of detections
        for img_i, (path, detections) in enumerate(zip(self.imgs, self.img_detections)):

            print("(%d) Image: '%s'" % (img_i, path))

            # Create plot
            img = np.array(Image.open(path))
            plt.figure()
            fig, ax = plt.subplots(1)
            ax.imshow(img)

            # Draw bounding boxes and labels of detections
            if detections is not None:
                # Rescale boxes to original image
                detections = rescale_boxes(
                    detections, self.img_size, img.shape[:2])
                unique_labels = detections[:, -1].cpu().unique()
                n_cls_preds = len(unique_labels)
                bbox_colors = random.sample(colors, n_cls_preds)
                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

                    print("\t+ Label: %s, Conf: %.5f" %
                          (self.class_names[int(cls_pred)], cls_conf.item()))

                    box_w = x2 - x1
                    box_h = y2 - y1

                    color = bbox_colors[int(
                        np.where(unique_labels == int(cls_pred))[0])]
                    # Create a Rectangle patch
                    bbox = patches.Rectangle(
                        (x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
                    # Add the bbox to the plot
                    ax.add_patch(bbox)
                    # Add label
                    plt.text(
                        x1,
                        y1,
                        s=self.class_names[int(cls_pred)],
                        color="white",
                        verticalalignment="top",
                        bbox={"color": color, "pad": 0},
                    )

            # Save generated image with detections
            plt.axis("off")
            plt.gca().xaxis.set_major_locator(NullLocator())
            plt.gca().yaxis.set_major_locator(NullLocator())
            filename = os.path.basename(path).split(".")[0]
            output_path = os.path.join(self.output_dir, f"{filename}.png")
            plt.savefig(output_path, bbox_inches="tight", pad_inches=0.0)
            plt.close()
