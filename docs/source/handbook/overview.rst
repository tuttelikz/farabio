Overview
==========

.. role::  raw-html(raw)
    :format: html

Deep learning has transformed the way biomedical engineers approach complex problems. Especially, whose work is associated with day-to-day medical/optical imaging equipment are able to integrate deep learning techniques rapidly. There are several biomedical applications we in our lab are interested to pursue in research, for instance, classfication for screening, segmentation for accurate border/region detection, super-resolution for quality enhancement and generative models, eg. digital histopathology.  

The pipeline of integrating deep learning for our applications is as follows:

| 1. Prepare data :raw-html:`&rarr;` 2. Customize model :raw-html:`&rarr;` 3. Train/test :raw-html:`&rarr;` 4. Analyze/Visualize

Purpose
--------
* Prototyping: code reuse
* Educational: tutorials

What can I do with this package?
------------------------------------
- Image data (pre)processing
- Train/customize deep learning models
- Visualize/monitor results

Package organization
------------------------

Diagram
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
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

How to report an issue
^^^^^^^^^^^^^^^^^^^^^^^^

If you find a bug, please report it by opening an `issue on Git <https://github.com/TBL-UNIST/tbl-ai/issues/new>`_. 

Clean code Caveat
^^^^^^^^^^^^^^^^^^^^
   - **Modules** should have short, *all-lowercase names. Underscores can be used* in the module name if it improves readability.
   - **Class names** should normally use the *CapWords* convention.
   - **Function names** should be *lowercase, with words separated by > underscores as necessary to improve readability.*
   - **Variable names** follow the *same convention as function names*.
   - If a **function argument**'s name clashes with a reserved keyword, it is generally better to append a single trailing underscore rather than use an abbreviation or spelling corruption. Thus *class_* is better than clss. (Perhaps better is to avoid such clashes by using a synonym.)
   - **Constants** are usually defined on a module level and written in all *capital letters with underscores separating words.*

   -- from `PEP 8 <https://www.python.org/dev/peps/pep-0008/#package-and-module-names>`_.
