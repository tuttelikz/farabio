Concepts
==========

.. contents:: Table of Contents
    :local:
    :depth: 3

Activation functions
------------------------

Binary Step
^^^^^^^^^^^^^^
.. math::

  f(x) = \left\{
        \begin{array}{lll}
            0 & for & x < x_{min} \\
            mx+b & for & x_{min} \leq x \leq x_{max}  \\
            1 & for & x > x_{max}
        \end{array}
    \right.

.. math::

  f'(x) = \left\{
        \begin{array}{lll}
            0 & for & x \neq 0  \\
            ? & for & x = 0
        \end{array}
    \right.

.. plot:: handbook/pyplots/binary_step.py


Piecewise Linear
^^^^^^^^^^^^^^^^^^

.. math::

  f(x) = \left\{
        \begin{array}{lll}
            0 & for & x < x_{min} \\
            mx+b & for & x_{min} \leq x \leq x_{max}  \\
            1 & for & x > x_{max}
        \end{array}
    \right.

.. math::

  f'(x) = \left\{
        \begin{array}{lll}
            0 & for & x < x_{min} \\
            m & for & x_{min} \leq x \leq x_{max}  \\
            0 & for & x > x_{max}
        \end{array}
    \right.

.. plot:: handbook/pyplots/piecewise_linear.py

Bipolar
^^^^^^^^^^^^^^^^^^

.. math::

  f(x) = \left\{
        \begin{array}{lll}
            -1 & for & x \leq 0  \\
            1 & for & x > 0
        \end{array}
    \right.

.. math::

  f'(x) = \left\{
        \begin{array}{lll}
            0 & for & x \neq 0  \\
            ? & for & x = 0
        \end{array}
    \right.

.. plot:: handbook/pyplots/bipolar.py

Sigmoid
^^^^^^^^^^^^^^^^^^

.. math::

  f(x)={\frac {1}{1+e^{-x}}}

.. math::

  f'(x)=f(x)(1-f(x))

.. plot:: handbook/pyplots/sigmoid.py

Bipolar Sigmoid
^^^^^^^^^^^^^^^^^^

.. math::

  f(x)={\frac {1-e^{-x}}{1+e^{-x}}}

.. math::

  f'(x)={\frac {2e^x}{(e^x+1)^2}}

.. plot:: handbook/pyplots/bipolar_sigmoid.py

Hyperbolic Tangent, TanH
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. math::

  f(x)={\frac {2}{1+e^{-2x}}}-1

.. math::

  f'(x)=1-f(x)^2

.. plot:: handbook/pyplots/tanh.py

Arctangent, ArcTan
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. math::

  f(x)=tan^{-1}(x)

.. math::

  f'(x)={\frac {1}{1+x^2}}

.. plot:: handbook/pyplots/arctan.py

Rectified Linear Units, ReLU
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. math::

  f(x) = \left\{
        \begin{array}{lll}
            0 & for & x \leq 0  \\
            x & for & x > 0
        \end{array}
    \right.

.. math::

  f'(x) = \left\{
        \begin{array}{lll}
            0 & for & x \leq 0  \\
            1 & for & x > 0
        \end{array}
    \right.

.. plot:: handbook/pyplots/relu.py

Leaky Rectified Linear Units, Leaky ReLU
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. math::

  f(x) = \left\{
        \begin{array}{lll}
            ax & for & x \leq 0  \\
            x & for & x > 0
        \end{array}
    \right.

.. math::

  f'(x) = \left\{
        \begin{array}{lll}
            a & for & x \leq 0  \\
            1 & for & x > 0
        \end{array}
    \right.

.. plot:: handbook/pyplots/leaky_relu.py

Exponential Linear Units, ELU
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. math::

  f(x) = \left\{
        \begin{array}{lll}
            a(e^x-1) & for & x \leq 0  \\
            x & for & x > 0
        \end{array}
    \right.

.. math::

  f'(x) = \left\{
        \begin{array}{lll}
            f(x)+a & for & x \leq 0  \\
            1 & for & x > 0
        \end{array}
    \right.

.. plot:: handbook/pyplots/elu.py

SoftPlus
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. math::

  f(x)=ln(1+e^x)

.. math::

  f'(x)={\frac {1}{1+e^{-x}}}

.. plot:: handbook/pyplots/softplus.py
