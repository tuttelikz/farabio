:py:mod:`~farabio.models.classification.class_trainer` Module
=========================================================================

:py:class:`~.ClassTrainer` class offers several classification models.

Available architectures
**************************

.. list-table:: CIFAR10-classification
   :widths: 25 25
   :header-rows: 1

   * - Model
     - Accuracy for CIFAR10
   * - `VGG16`_
     - 92.64%
   * - `ResNet18`_
     - 93.02%
   * - `ResNet50`_
     - 93.62%
   * - `ResNet101`_
     - 93.75%
   * - `RegNetX_200MF`_
     - 94.24%
   * - `RegNetY_400MF`_
     - 94.29% 
   * - `MobileNetV2`_
     - 94.43%
   * - `ResNeXt29(32x4d)`_
     - 94.73%
   * - `ResNeXt29(2x64d)`_
     - 94.82%
   * - `SimpleDLA`_
     - 94.89%
   * - `DenseNet121`_
     - 95.04%
   * - `PreActResNet18`_
     - 95.11%
   * - `DPN92`_
     - 95.16%
   * - `DLA`_
     - 95.47%


:py:class:`.arch.vgg.VGG`

.. figure:: ../imgs/vgg.png
   :width: 400

   `VGG16`_

:py:class:`.arch.resnet.ResNet`

.. figure:: ../imgs/resnet.png
   :width: 600

   `ResNet18`_

:py:class:`.arch.resnet.RegNet`

.. figure:: ../imgs/regnet.png
   :width: 400

   `RegNetX_200MF`_

:py:class:`.arch.mobilenetv2.MobileNetV2`

.. figure:: ../imgs/mobilenet2.png
   :width: 400

   `MobileNetV2`_

:py:class:`.arch.resnext.ResNeXt`

.. figure:: ../imgs/resnext.png
   :width: 400

   `ResNeXt29(32x4d)`_

:py:class:`.arch.densenet.DenseNet`

.. figure:: ../imgs/densenet.png
   :width: 400

   `DenseNet121`_

:py:class:`.arch.preact_resnet.PreActResNet`

.. figure:: ../imgs/preactresnet.png
   :width: 250

   `PreActResNet18`_

:py:class:`.arch.dpn.DPN`

.. figure:: ../imgs/dpn.png
   :width: 400

   `DPN92`_

:py:class:`.arch.dla_simple.SimpleDLA`

.. figure:: ../imgs/simpledla.png
   :width: 600

   `DLA`_

************
References
************

.. target-notes::

.. _`VGG16`: https://arxiv.org/abs/1409.1556
.. _`ResNet18`: https://arxiv.org/abs/1512.03385
.. _`ResNet50`: https://arxiv.org/abs/1512.03385
.. _`ResNet101`: https://arxiv.org/abs/1512.03385
.. _`RegNetX_200MF`: https://arxiv.org/abs/2003.13678
.. _`RegNetY_400MF`: https://arxiv.org/abs/2003.13678
.. _`MobileNetV2`: https://arxiv.org/abs/1801.04381
.. _`ResNeXt29(32x4d)`: https://arxiv.org/abs/1611.05431
.. _`ResNeXt29(2x64d)`: https://arxiv.org/abs/1611.05431
.. _`DenseNet121`: https://arxiv.org/abs/1608.06993
.. _`PreActResNet18`: https://arxiv.org/abs/1603.05027
.. _`DPN92`: https://arxiv.org/abs/1707.01629
.. _`SimpleDLA`: https://arxiv.org/abs/1707.064
.. _`DLA`: https://arxiv.org/pdf/1707.06484.pdf


:py:class:`~.ClassTrainer` class
-----------------------------------------

.. autoclass:: farabio.models.classification.class_trainer.ClassTrainer
   :members:
