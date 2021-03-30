:py:mod:`~farabi.core.configs` Module
==========================================

:py:mod:`~farabi.core.configs` module combines is a main place for model configurations. These 
settings are separately defined in their related functions. Each of these functions use group of 
arguments from argparse.ArgumentParser().

Argument groups
---------------------
* :py:obj:`data`
* :py:obj:`model`
* :py:obj:`test`
* :py:obj:`test_arg`
* :py:obj:`log_arg`
* .. option:: compute_arg`
* .. option:: misc_arg`

Non-lifecycle hooks
-----------------------

The :py:mod:`~farabi.core.config` package contains base configurations for models:

.. autofunction:: farabi.core.configs._cfg_unet
.. autofunction:: farabi.core.configs._cfg_attunet
.. autofunction:: farabi.core.configs._cfg_srgan
.. autofunction:: farabi.core.configs._cfg_cyclegan
.. autofunction:: farabi.core.configs._cfg_yolov3
.. autofunction:: farabi.core.configs._cfg_fasterrcnn