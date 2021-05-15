Usage
============

Prerequisites
-------------------------
.. code-block:: text

   python=3.8
   numpy=1.19
   scikit_image=0.18
   Pillow=8.1
   torch=1.7
   torchvision=0.8
   matplotlib=3.3

.. raw:: html

   <details>
   <summary><a>list all prerequisites</a></summary>

.. code-block:: text

   albumentations=0.5.2
   autopep8=1.5.5
   fire=0.4.0
   imageio=2.9.0
   ipdb=0.13.6
   matplotlib=3.3.4
   numpy=1.19.4
   opencv_python=4.5.1.48
   pandas=1.2.2
   Pillow=8.1.1
   recommonmark=0.7.1
   scikit_image=0.18.1
   scipy=1.6.1
   seaborn=0.11.1
   sphinx-git=11.0.0
   tabulate=0.8.9
   tensorboardX=2.1
   tensorflow=2.4.1
   terminaltables=3.1.0
   torch=1.7
   torchvision=0.8
   torchsummary=1.5.1
   tqdm=4.58.0
   visdom=0.1.8.9

.. raw:: html

   </details>

.. role:: bash(code)
   :language: bash


How to install
-------------------------

1. Activate conda environment 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: bash

   $ conda create -n coolenv python=3.8
   $ conda activate coolenv
   $ conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch


2. Install prerequisites:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: bash

   $ git clone -b https://github.com/tuttelikz/farabio.git && cd farabio
   $ pip install -r requirements.txt

3. Install farabio
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
A. With pip:

.. code-block:: bash

   $ pip install farabio

B. Setup from source:

.. code-block:: bash
   
   $ python -m pip install --upgrade pip setuptools wheel  # ensure up-to-date
   $ pip install [-e] .                                    # flag for editable mode 