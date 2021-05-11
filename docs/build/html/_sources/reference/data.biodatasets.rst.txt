:py:mod:`~farabio.data.biodatasets` Module
==================================================

:py:mod:`~farabio.data.biodatasets` module provides classes to load public biomedical datasets
in a PyTorch friendly manner.

:py:class:`~.ChestXrayDataset` Class
------------------------------------------------------------------------------

`Kaggle Chest X-Ray Images <https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia>`_ competition dataset to detect pneumonia.

.. code-block:: python

   train_dataset = ChestXrayDataset(root=".", transform=None, download=True)
   train_dataset.visualize_dataset()

.. image:: ../imgs/ChestXrayDataset.png
   :width: 300

Docs of :py:class:`~.ChestXrayDataset`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: farabio.data.biodatasets.ChestXrayDataset
   :members:

:py:class:`~.DSB18Dataset` Class
------------------------------------------------------------------------------

`Kaggle 2018 Data Science Bowl <https://www.kaggle.com/c/data-science-bowl-2018/overview>`_ competition dataset for segmented nuclei images.

.. code-block:: python

   train_dataset = DSB18Dataset(root=".", transform=None, download=False)
   train_dataset.visualize_dataset(5)

.. image:: ../imgs/DSB18Dataset.png
   :width: 300

Docs of :py:class:`~.DSB18Dataset`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: farabio.data.biodatasets.DSB18Dataset
   :members:

:py:class:`~.RetinopathyDataset` Class
------------------------------------------------------------------------------

Retina images taken using fundus photography from `Kaggle APTOS 2019 Blindness Detection <https://www.kaggle.com/c/aptos2019-blindness-detection/data>`_

.. code-block:: python

   train_dataset = RetinopathyDataset(root=".", transform=None, download=True)
   train_dataset.visualize_dataset(9)

.. image:: ../imgs/RetinopathyDataset.png
   :width: 300

Docs of :py:class:`~.RetinopathyDataset`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: farabio.data.biodatasets.RetinopathyDataset
   :members:


:py:class:`~.HistocancerDataset` Class
------------------------------------------------------------------------------

Histopathologic Cancer Detection from `Kaggle Histopathologic Cancer Detection <https://www.kaggle.com/c/histopathologic-cancer-detection/data>`_

.. code-block:: python

   train_dataset = HistocancerDataset(root=".", download=True, train=True)
   train_dataset.visualize_dataset()

.. image:: ../imgs/HistocancerDataset.png
   :width: 600

Docs of :py:class:`~.HistocancerDataset`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: farabio.data.biodatasets.HistocancerDataset
   :members:

:py:class:`~.RANZCRDataset` Class
------------------------------------------------------------------------------

Catheters presence and position detection from `RANZCR CLiP - Catheter and Line Position Challenge <https://www.kaggle.com/c/ranzcr-clip-catheter-line-classification/data>`_

.. code-block:: python

   train_dataset = RANZCRDataset(".", train=True, transform=None, download=True)
   train_dataset.visualize_dataset()

.. image:: ../imgs/RANZCRDataset.png
   :width: 600

Docs of :py:class:`~.RANZCRDataset`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: farabio.data.biodatasets.RANZCRDataset
   :members: