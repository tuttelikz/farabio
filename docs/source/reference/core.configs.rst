:py:mod:`~farabi.core.configs` Module
==========================================


:py:mod:`~farabi.core.configs` module combines is a main place for model configurations. These 
settings are separately defined in their related functions. Each of these functions use group of 
arguments from argparse.ArgumentParser().

Argument groups
---------------------
* .. option:: data_arg
* .. option:: model_arg
* .. option:: train_arg
* .. option:: test_arg
* .. option:: log_arg
* .. option:: compute_arg
* .. option:: misc_arg

Non-lifecycle hooks
-----------------------

The :py:mod:`~farabi.core.config` package contains base configurations for models:

.. autofunction:: farabi.core.configs.define_unet
.. autofunction:: farabi.core.configs.define_srgan
.. autofunction:: farabi.core.configs.define_cyclegan
.. autofunction:: farabi.core.configs.define_yolo
.. autofunction:: farabi.core.configs.define_faster_rcnn