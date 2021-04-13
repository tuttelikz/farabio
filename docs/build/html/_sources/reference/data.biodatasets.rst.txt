:py:mod:`~farabio.data.biodatasets` Module
==================================================

:py:mod:`~farabio.data.biodatasets` module provides classes to load public biomedical datasets
in a PyTorch friendly manner.

.. autoclass:: farabio.data.biodatasets.ChestXrayDataset
   :members:

.. code-block:: python

   train_dataset = DSB18Dataset(root="/home/data/02_SSD4TB/suzy/datasets/public/", transform=None, download=False)
   dsb18_plt = train_dataset.visualize_dataset(5)
   dsb18_plt.show()

.. image:: ../imgs/DSB18Dataset.png
   :width: 300

.. autoclass:: farabio.data.biodatasets.DSB18Dataset
   :members:

.. autoclass:: farabio.data.biodatasets.RetinopathyDataset
   :members:

.. autoclass:: farabio.data.biodatasets.HistocancerDataset
   :members:
