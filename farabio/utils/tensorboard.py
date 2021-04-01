import tensorflow as tf
import tensorboardX as tb
from tensorboardX.summary import Summary


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


# import tensorflow as tf
# import tensorboardX as tb
# from tensorboardX.summary import Summary
# import time
# import sys

# import numpy as np
# import torch
# import skimage
# from skimage.measure import label
# from skimage import img_as_ubyte
# from farabio.data.imgops import ImgOps
