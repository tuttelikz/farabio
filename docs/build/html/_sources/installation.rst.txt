Getting started
=================

How to install
-------------------------

1. Activate conda environment 
---------------------------------
.. code-block:: bash

   $ conda create -n myenv python=3.8
   $ conda activate myenv

.. role:: bash(code)
   :language: bash

2. Install farabio
---------------------------------

A. With pip:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: bash

   $ pip install farabio -f https://download.pytorch.org/whl/torch_stable.html

B. Setup from source:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: bash
   
   $ git clone https://github.com/tuttelikz/farabio.git && cd farabio
   $ python -m pip install --upgrade pip setuptools wheel
   $ pip install . -f https://download.pytorch.org/whl/torch_stable.html

.. role:: bash(code)
   :language: bash
