Usage
============

How to install
-------------------------

1. Activate conda environment 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: bash

   $ conda create -n coolenv python=3.8
   $ conda activate coolenv

.. role:: bash(code)
   :language: bash

2. Install farabio
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
A. With pip:

.. code-block:: bash

   $ pip install farabio

B. Setup from source:

.. code-block:: bash
   
   $ python -m pip install --upgrade pip setuptools wheel  # ensure up-to-date
   $ pip install [-e] . -f https://download.pytorch.org/whl/torch_stable.html  # flag for editable mode 

.. role:: bash(code)
   :language: bash

Prerequisites
-------------------------
.. code-block:: text

   docutils==0.17.1,
   jupyterlab==3.1.6,
   matplotlib==3.4.3,
   numpy==1.21.1,
   pandas==1.3.1,
   scikit-image==0.18.2,
   scikit-learn==0.24.2,
   torch==1.9.0+cu111,
   torchvision==0.10.0+cu111,
   torchaudio==0.9.0
