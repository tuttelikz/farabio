Overview
==========

.. role::  raw-html(raw)
    :format: html

Deep learning has transformed many aspects of industrial pipelines recently. 
Scientists involved in biomedical imaging research are also benefiting from 
the power of AI to tackle complex challenges. Although academic community 
has widely accepted image processing tools, such as scikit-image, ImageJ, 
there is still a need for a tool which integrates deep learning 
into biomedical image analysis. We propose a minimal, 
but convenient Python package based on PyTorch with biomedical datasets, common deep learning models, 
and extended by flexible trainers.

What can I do with this package?
------------------------------------
- Load public biomedical datasets
- Load common deep learning models
- Do basic image preprocessing and transformations
- Customize training loops to your own needs

Package structure
------------------------
.. graphviz::
   
   graph farabioOverview {

   node [shape=box, colorscheme=set32 , style=rounded];

   farabio -- core;
   farabio -- data;
   farabio -- models;
   farabio -- utils;

   farabio  [fillcolor=1, style="rounded"]
   core  [fillcolor=2, style="rounded"]
   data  [fillcolor=2, style="rounded"]
   models  [fillcolor=2, style="rounded"]
   utils  [fillcolor=2, style="rounded"]
   }

How to contribute?
----------------------

You can contribute to this package by reporting issues and/or by sending pull request.

If you find a bug, please report it by opening an `issue on Git <https://github.com/TBL-UNIST/tbl-ai/issues/new>`_. 
