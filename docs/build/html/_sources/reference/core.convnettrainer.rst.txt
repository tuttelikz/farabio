:py:mod:`~farabio.core.convnettrainer` Module
==============================================

:py:class:`~.ConvnetTrainer` class makes use of hooks. Hooks are a collection of methods which provide
quick access to exact entry in loop. In this way, we can override these methods with custom functionality
in either training, evaluation or test loops.

Non-lifecycle hooks
---------------------

Default methods
^^^^^^^^^^^^^^^^^^^^^^^^^
* :py:meth:`~.ConvnetTrainer.init_attr`
* :py:meth:`~.ConvnetTrainer.get_trainloader`
* :py:meth:`~.ConvnetTrainer.get_testloader`
* :py:meth:`~.ConvnetTrainer.build_model` | :py:meth:`~.ConvnetTrainer.build_parallel_model`

Methods to initalize class attributes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* :py:meth:`~.ConvnetTrainer.define_data_attr`
* :py:meth:`~.ConvnetTrainer.define_model_attr`
* :py:meth:`~.ConvnetTrainer.define_train_attr`
* :py:meth:`~.ConvnetTrainer.define_test_attr`
* :py:meth:`~.ConvnetTrainer.define_log_attr`
* :py:meth:`~.ConvnetTrainer.define_compute_attr`
* :py:meth:`~.ConvnetTrainer.define_misc_attr`

Native methods
^^^^^^^^^^^^^^^^^^^^^^^^^
* :py:meth:`~.ConvnetTrainer.loss_backward`
* :py:meth:`~.ConvnetTrainer.optimizer_step`
* :py:meth:`~.ConvnetTrainer.load_model` | :py:meth:`~.ConvnetTrainer.load_parallel_model`
* :py:meth:`~.ConvnetTrainer.save_model` | :py:meth:`~.ConvnetTrainer.save_parallel_model`

Lifecycle hooks
---------------------

Training loop
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. graphviz::

   digraph Trainloop {

      graph [fontsize=8];
      edge [fontsize=8];
      node [fontsize=8, shape=box, colorscheme=set36, style=rounded];

      train -> on_train_start;
      on_train_start -> start_logger;
      start_logger -> train_loop;
      train_loop -> on_train_end;
      train_loop -> train_epoch;
      train_epoch -> on_train_epoch_start;
      on_epoch_end -> on_train_epoch_start [label = "_num_epochs"];
      on_train_epoch_start -> train_batch;
      train_batch -> on_train_epoch_end;
      on_train_epoch_end -> evaluate_epoch;
      evaluate_epoch -> on_epoch_end;
      train_batch -> on_start_training_batch;
      on_start_training_batch -> training_step;
      training_step -> on_end_training_batch;
      on_end_training_batch -> on_start_training_batch [label = "train_epoch_iter"];

      train  [fillcolor=1, style="rounded,filled"]
      on_train_start  [fillcolor=2, style="rounded,filled"]
      start_logger  [fillcolor=2, style="rounded,filled"]
      train_loop  [fillcolor=2, style="rounded,filled"]
      on_train_end  [fillcolor=2, style="rounded,filled"]
      train_epoch  [fillcolor=3, style="rounded,filled"]
      on_train_epoch_start  [fillcolor=4, style="rounded,filled"]
      train_batch  [fillcolor=4, style="rounded,filled"]
      on_train_epoch_end  [fillcolor=4, style="rounded,filled"]
      evaluate_epoch  [fillcolor=4, style="rounded,filled,dashed"]
      on_epoch_end  [fillcolor=4, style="rounded,filled"]
      on_start_training_batch  [fillcolor=5, style="rounded,filled"]
      training_step  [fillcolor=5, style="rounded,filled"]
      on_end_training_batch  [fillcolor=5, style="rounded,filled"]

      }

Evaluation Loop
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. graphviz::

   digraph Evalloop {
      
      graph [fontsize=8];
      edge [fontsize=8];
      node [fontsize=8, shape=box, colorscheme=set36, style=rounded];

      evaluate_epoch -> on_evaluate_epoch_start
      on_evaluate_epoch_start -> on_evaluate_batch_start;
      on_evaluate_batch_start -> evaluate_batch;
      evaluate_batch -> on_evaluate_batch_end;
      on_evaluate_batch_end -> on_evaluate_epoch_end;
      on_evaluate_batch_end -> on_evaluate_batch_start [label = "valid_epoch_iter"];

      evaluate_epoch  [fillcolor=4, style="rounded,filled,dashed"]
      on_evaluate_epoch_start  [fillcolor=5, style="rounded,filled"]
      on_evaluate_epoch_end  [fillcolor=5, style="rounded,filled"]
      on_evaluate_batch_start  [fillcolor=6, style="rounded,filled"]
      evaluate_batch  [fillcolor=6, style="rounded,filled"]
      on_evaluate_batch_end  [fillcolor=6, style="rounded,filled"]

      }

Test loop
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. graphviz::

   digraph Testloop {

      graph [fontsize=8];
      edge [fontsize=8];
      node [fontsize=8, shape=box, colorscheme=set33 , style=rounded];

      test -> load_model;
      test -> load_parallel_model;
      load_model -> on_test_start;
      load_parallel_model -> on_test_start;
      on_test_start -> test_loop;
      test_loop -> on_test_end;
      test_loop -> on_start_test_batch;
      on_start_test_batch -> test_step;
      test_step -> on_end_test_batch;
      on_end_test_batch -> on_start_test_batch [label = "test_loop_iter"];

      test  [fillcolor=1, style="rounded,filled"]
      load_model  [fillcolor=2, style="rounded,filled"]
      load_parallel_model  [fillcolor=2, style="rounded,filled"]
      on_test_start  [fillcolor=2, style="rounded,filled"]
      test_loop  [fillcolor=2, style="rounded,filled"]
      on_test_end  [fillcolor=2, style="rounded,filled"]
      on_start_test_batch  [fillcolor=3, style="rounded,filled"]
      test_step  [fillcolor=3, style="rounded,filled"]
      on_end_test_batch  [fillcolor=3, style="rounded,filled"]

      }

Docs
---------------------

.. autoclass:: farabio.core.convnettrainer.ConvnetTrainer
   :members:

