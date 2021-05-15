:py:mod:`~farabio.utils.losses` Module
==================================================

:py:mod:`~farabio.utils.losses` module classes to load common loss functions.

:py:class:`~.DiceLoss`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The Dice coefficient, or Dice-Sørensen coefficient, is a common metric for 
pixel segmentation that can also be modified to act as a loss function:

.. math::

    D S C=\frac{2|X \cap Y|}{|X|+|Y|}

.. autoclass:: farabio.utils.losses.DiceLoss
   :members:

:py:class:`~.DiceBCELoss`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This loss combines Dice loss with the standard binary cross-entropy (BCE) loss 
that is generally the default for segmentation models. Combining the two methods 
allows for some diversity in the loss, while benefitting from the stability of BCE. 
The equation for multi-class BCE by itself will be familiar to anyone who has studied 
logistic regression:

.. math::

    J(\mathbf{w})=\frac{1}{N} \sum_{n=1}^{N} H\left(p_{n}, q_{n}\right)=-\frac{1}{N} \sum_{n=1}^{N}\left[y_{n} \log \hat{y}_{n}+\left(1-y_{n}\right) \log \left(1-\hat{y}_{n}\right)\right]

.. autoclass:: farabio.utils.losses.DiceBCELoss
   :members:

:py:class:`~.IoULoss`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The IoU metric, or Jaccard Index, is similar to the Dice metric and is calculated 
as the ratio between the overlap of the positive instances between two sets, 
and their mutual combined values.

.. math::
    
    J(A, B)=\frac{|A \cap B|}{|A \cup B|}=\frac{|A \cap B|}{|A|+|B|-|A \cap B|}

.. autoclass:: farabio.utils.losses.IoULoss
   :members:

:py:class:`~.FocalLoss`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Focal Loss was introduced by Lin et al of Facebook AI Research in 2017 
as a means of combatting extremely imbalanced datasets where positive cases 
were relatively rare. Their paper "Focal Loss for Dense Object Detection" 
is retrievable here: https://arxiv.org/abs/1708.02002. In practice, 
the researchers used an alpha-modified version of the function so I have included 
it in this implementation.

.. autoclass:: farabio.utils.losses.FocalLoss
   :members:

:py:class:`~.TverskyLoss`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

https://arxiv.org/abs/1706.05721

.. autoclass:: farabio.utils.losses.TverskyLoss
   :members:

:py:class:`~.FocalTverskyLoss`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
A variant on the Tversky loss that also includes the gamma modifier from Focal Loss.

.. autoclass:: farabio.utils.losses.FocalTverskyLoss
   :members:

:py:class:`~.LovaszHingeLoss`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: farabio.utils.losses.LovaszHingeLoss
   :members:
