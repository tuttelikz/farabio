:py:mod:`~farabio.data.biodatasets` Module
==================================================

:py:mod:`~farabio.data.biodatasets` module provides classes to load public biomedical datasets
in a PyTorch friendly manner.

:py:class:`~.ChestXrayDataset` Class
------------------------------------------------------------------------------

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

.. code-block:: python

   train_dataset = DSB18Dataset(root=".", transform=None, download=False)
   dsb18_plt = train_dataset.visualize_dataset(5)
   dsb18_plt.show()

.. image:: ../imgs/DSB18Dataset.png
   :width: 300

Docs of :py:class:`~.DSB18Dataset`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: farabio.data.biodatasets.DSB18Dataset
   :members:

:py:class:`~.RetinopathyDataset` Class
------------------------------------------------------------------------------

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

.. code-block:: python

   train_dataset = HistocancerDataset(root=".", download=True, train=True)
   train_dataset.visualize_dataset()

.. image:: ../imgs/HistocancerDataset.png
   :width: 300

Docs of :py:class:`~.HistocancerDataset`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: farabio.data.biodatasets.HistocancerDataset
   :members:
