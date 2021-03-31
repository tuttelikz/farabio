:py:mod:`~farabio.core.configs` Module
==========================================

:py:mod:`~farabio.core.configs` module combines is a main place for model configurations. These 
settings are separately defined in their related functions.

Categories of arguments
------------------------
* :py:obj:`data`
* :py:obj:`model`
* :py:obj:`test`
* :py:obj:`test_arg`
* :py:obj:`log_arg`
* :py:obj:`compute_arg`
* :py:obj:`misc_arg`

Default configurations for models
------------------------------------

The :py:mod:`~farabio.core.config` package contains base configurations for models:

.. autofunction:: farabio.core.configs._cfg_unet
.. autofunction:: farabio.core.configs._cfg_attunet
.. autofunction:: farabio.core.configs._cfg_srgan
.. autofunction:: farabio.core.configs._cfg_cyclegan
.. autofunction:: farabio.core.configs._cfg_yolov3
.. autofunction:: farabio.core.configs._cfg_fasterrcnn