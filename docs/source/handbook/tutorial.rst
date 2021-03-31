Tutorial
==========

.. contents:: Table of Contents
    :local:
    :depth: 3

How to train model
---------------------

Adjust the following code from :code:`farabio.kernel.py`:

.. literalinclude:: ../../../farabio/kernel.py
   :language: python

How to use Tensorboard
--------------------------

.. code-block:: bash

   $ tensorboard --logdir=<DIR-WHERE-TFRECORDS-FILE-STORED> --port 6006

How to train Yolo-v3 on custom dataset
----------------------------------------

1. Create custom configuration Yolo:

.. code-block:: bash

   $ cd farabio/models/detection/yolov3/config/          # Navigate to config dir
   $ bash create_custom_model.sh <num-classes>  # Will create custom model 'yolov3-custom.cfg'

2. Modify the 'custom.data' file according to the dataset:

.. code-block:: bash

   $ cd farabio/models/detection/config/          # Navigate to config dir
   $ nano custom.data                           # Change classes, train, valid and names fields

3. Start YOLO trainer with settings:

.. code-block:: bash

   $ python kernel.py --mode train --model_def <yolov3-custom.cfg> --data_config <custom.data> --pretrained_weights <darknet53.conv.74>     # Start training
   $ python kernel.py --mode <test|detect> --model_def <yolov3-custom.cfg> --data_config <custom.data> --weights_path <weights.pth>         # Start test/detecting

Linux related
----------------------------------------

Get the user of PID
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   $ ps -u -p <PID>