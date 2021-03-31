import os
import time
import datetime
import random
import torch
import numpy as np
from PIL import Image
from torch.autograd import Variable
from tqdm import tqdm
from terminaltables import AsciiTable
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
from darknet import Darknet
from farabio.utils.helpers import rescale_boxes
from farabio.utils.regul import weights_init_normal
from farabio.utils.loggers import Logger
from farabio.utils.losses import get_batch_statistics, non_max_suppression, ap_per_class, xywh2xyxy


class Trainer(object):
    def __init__(self, config, train_loader, valid_loader, class_names, mode='train'):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(config.chckpt_dir, exist_ok=True)

        self.epochs = config.epochs
        self.grad_acc = config.gradient_accumulations
        self.chckpt_dir = config.chckpt_dir
        self.logger = Logger(config.logdir)
        self.checkpoint_interval = config.checkpoint_interval
        self.evaluation_interval = config.evaluation_interval
        self.img_size = config.img_size
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.class_names = class_names
        self.mode = mode

        self.iou_thres = config.iou_thres
        self.conf_thres = config.conf_thres
        self.nms_thres = config.nms_thres
        self.batch_size = config.batch_size
        self.detect = config.detect
        self.output_dir = config.output_dir

        if self.mode == 'train':
            self.build_model(config.model_def, config.pretrained_weights)
        elif self.mode == 'test':
            self.build_model(config.model_def, config.weights_path)
        elif self.mode == 'detect':
            self.build_model(config.model_def, config.weights_path)

        if config.optim == 'adam':
            optim = torch.optim.Adam

        self.optimizer = optim(self.model.parameters())

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

        self.image_folder = config.image_folder
        self.dconf_thres = config.dconf_thres
        self.dnms_thres = config.dnms_thres
        self.dbatch_size = config.dbatch_size

        # if self.mode == 'detect':
        #     self.detect

    def build_model(self, model_conf, *args):
        """Build model
        Parameters
        ----------
        epoch : int
            current epoch
        """
        # Initiate model
        self.model = Darknet(model_conf).to(self.device)

        if self.mode == 'train':
            self.model.apply(weights_init_normal)

            pretrained_weights = args[0]
            if pretrained_weights.endswith(".pth"):
                # Custom Darknet weights
                self.model.load_state_dict(torch.load(pretrained_weights))
            else:
                # Darknet-53 on ImageNet
                self.model.load_darknet_weights(pretrained_weights)

        elif self.mode == 'test':
            weights_path = args[0]

            if weights_path.endswith(".weights"):
                # Load yolov3 coco weights
                self.model.load_darknet_weights(weights_path)
            else:
                # Load custom data checkpoint weights
                self.model.load_state_dict(torch.load(weights_path))

        elif self.mode == 'detect':
            weights_path = args[0]

            if weights_path.endswith(".weights"):
                # Load yolov3 coco weights
                self.model.load_darknet_weights(weights_path)
            else:
                # Load custom data checkpoint weights
                self.model.load_state_dict(torch.load(weights_path))

    def train(self):
        for epoch in range(self.epochs):
            self.model.train()
            start_time = time.time()

            for batch_i, (_, imgs, targets) in enumerate(self.train_loader):
                batches_done = len(self.train_loader) * epoch + batch_i

                imgs = Variable(imgs.to(self.device))
                targets = Variable(targets.to(self.device),
                                   requires_grad=False)

                loss, outputs = self.model(imgs, targets)
                loss.backward()

                if batches_done % self.grad_acc == 0:
                    # Accumulates gradient before each step
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                # ----------------
                #   Log progress
                # ----------------

                log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (
                    epoch, self.epochs, batch_i, len(self.train_loader))

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
                    tensorboard_log += [("loss", loss.item())]
                    self.logger.list_of_scalars_summary(
                        tensorboard_log, batches_done)

                log_str += AsciiTable(metric_table).table
                log_str += f"\nTotal loss {loss.item()}"

                # Determine approximate time left for epoch
                epoch_batches_left = len(self.train_loader) - (batch_i + 1)
                time_left = datetime.timedelta(
                    seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
                log_str += f"\n---- ETA {time_left}"

                print(log_str)

                self.model.seen += imgs.size(0)

            if epoch % self.evaluation_interval == 0:
                print("\n---- Evaluating Model ----")
                # Evaluate the model on the validation set
                precision, recall, AP, f1, ap_class = self.evaluate(
                    self.model,
                    self.valid_loader,
                    iou_thres=0.5,
                    conf_thres=0.5,
                    nms_thres=0.5,
                    img_size=self.img_size,
                )

                evaluation_metrics = [
                    ("val_precision", precision.mean()),
                    ("val_recall", recall.mean()),
                    ("val_mAP", AP.mean()),
                    ("val_f1", f1.mean()),
                ]

                self.logger.list_of_scalars_summary(evaluation_metrics, epoch)

                # Print class APs and mAP
                ap_table = [["Index", "Class name", "AP"]]
                for i, c in enumerate(ap_class):
                    ap_table += [[c, self.class_names[c], "%.5f" % AP[i]]]
                print(AsciiTable(ap_table).table)
                print(f"---- mAP {AP.mean()}")

            if epoch % self.checkpoint_interval == 0:
                torch.save(self.model.state_dict(),
                           os.path.join(f"{self.chckpt_dir}", f"yolov3_ckpt_{epoch}.pth"))

    def evaluate(self, model, valid_loader, iou_thres, conf_thres, nms_thres, img_size):
        model.eval()

        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

        labels = []
        sample_metrics = []  # List of tuples (TP, confs, pred)
        for batch_i, (_, imgs, targets) in enumerate(tqdm(valid_loader, desc="Detecting objects")):

            if targets is None:
                continue

            # Extract labels
            labels += targets[:, 1].tolist()
            # Rescale target
            targets[:, 2:] = xywh2xyxy(targets[:, 2:])
            targets[:, 2:] *= img_size

            imgs = Variable(imgs.type(Tensor), requires_grad=False)

            with torch.no_grad():
                outputs = model(imgs)
                outputs = non_max_suppression(
                    outputs, conf_thres=conf_thres, nms_thres=nms_thres)

            sample_metrics += get_batch_statistics(outputs,
                                                   targets, iou_threshold=iou_thres)

        # Concatenate sample statistics
        true_positives, pred_scores, pred_labels = [
            np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
        precision, recall, AP, f1, ap_class = ap_per_class(
            true_positives, pred_scores, pred_labels, labels)

        return precision, recall, AP, f1, ap_class

    def test(self):
        print("Compute mAP...")

        precision, recall, AP, f1, ap_class = self.evaluate(
            self.model,
            self.valid_loader,
            iou_thres=self.iou_thres,
            conf_thres=self.conf_thres,
            nms_thres=self.nms_thres,
            img_size=self.img_size,
        )

        print("Average Precisions:")

        for i, c in enumerate(ap_class):
            print(f"+ Class '{c}' ({self.class_names[c]}) - AP: {AP[i]}")

        print(f"mAP: {AP.mean()}")

    def detect_perform(self):

        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

        imgs = []  # Stores image paths
        img_detections = []  # Stores detections for each image index

        print("\nPerforming object detection:")
        prev_time = time.time()
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
            imgs.extend(img_paths)
            img_detections.extend(detections)

        # Bounding-box colors
        cmap = plt.get_cmap("tab20b")
        colors = [cmap(i) for i in np.linspace(0, 1, 20)]

        print("\nSaving images:")
        # Iterate through images and save plot of detections
        for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):

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
