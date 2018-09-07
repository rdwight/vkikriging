.. vkikriging documentation master file, it should at least
   contain the root `toctree` directive.

==============================================================
 Kriging and Gradient-Enhanced Kriging for VKI Lecture Series
==============================================================

Python module implementing interpolation with *Gaussian process regression*, a.k.a. *Kriging*.
Features:

- Arbetrary dimensional interpolation.
- Use of gradients in reconstruction (*Gradient Enhanced Kriging*).
- Simple and univeral Kriging.
- Implementation follows Bayesian derivation of Kriging, descibed in the `tutorial`_.


.. note::

   This code was written for the `Von Karman Institute`_ lecture series on multi-disciplinary
   optimization, and is primarily intended as educational (accompanying the `tutorial`_).
   As such it sacrifices many possible code optimizations to clarity.  If you are planning
   to use it for **applications**, it may still be fast enough, depending on the size of your
   data.  Preferentially use routines in the module :any:`vkikriging.kriging_v2`.

.. _Von Karman Institute: https://www.vki.ac.be/
.. _tutorial: ./Notes_v3_2018-08.pdf


Installing
==========

With git (recommended for educational use)
------------------------------------------
The package is pure Python 3.  To get the source, clone the `GitHub`_ repository::

  git clone https://github.com/rdwight/vkikriging.git

and add `vkikriging` to your ``PYTHONPATH`` (e.g. with ``bash``)::

  export PYTHONPATH=/home/fred/vkikriging:$PYTHONPATH

You'll need ``numpy``, ``scipy``, ``matplotlib`` and ``algopy`` (automatic differentiation).
I recommend using an `Anaconda`_ Python install.

.. _Anaconda: https://www.anaconda.com/download/
  
With pip (recommended for applications)
---------------------------------------
The package is distributed via PyPI, so just use::

  pip install vkikriging

.. _GitHub: https://github.com/rdwight/vkikriging

Usage
=====

For experimenting with *cheap* functions in 1d and 2d, use classes :any:`Example1d` and
class :any:`Example2d`, which have a simple interface and do some convenient plotting.
For expensive functions and/or higher dimensions use directly the routines of
:any:`vkikriging.kriging_v2`.

Basic usage with plotting (1d)
------------------------------

Using the class :any:`Example1d` you only have to specify a function, its derivative,
the locations of sample points, and parameters for the Kriging/GEK model.  The following
approximates a sine curve on [-5,15] with 5 samples with gradients:

.. literalinclude:: ../examples/example1.py

Some timing information is printed to ``stdout``, and the following figure is plotted:

.. image:: Figure_1.png
	   
The black curve is the exact function, black circles and lines are samples of the function
and its derivative.  The heavy blue line is the GEK mean, with 1,2,3-standard-deviation
regions in light blue.  The thin blue lines are 10 independent samples from the
predictive distribution.

Basic usage with plotting (2d)
------------------------------

To achieve basic interpolation in 2d use the class :any:`Example2d`.  This expects an
object describing the function, with attributes ``f``, ``df``, ``xmin``, ``xmax``,
and ``xopt``.  See examples in :any:`vkikriging.test_functions`.  To approximate a
parabola with Sobol samples:

.. literalinclude:: ../examples/example2.py

Which produces the following figure:
  
.. image:: Figure_2.png
	   
The red circle is the exact minimum, and the crosses are the minima approximated by Kriging/GEK.

Higher-dimensions/expensive functions
-------------------------------------

The following code performs general-purpose sampling and reconstruction of a
``d``-dimensional function, with convergence analysis on number of samples:

.. literalinclude:: ../examples/example3.py

Which produces the following figure:

.. image:: Figure_3.png

	   
Documentation and API reference
===============================

.. toctree::
   :maxdepth: 4

   vkikriging
   example1d
   example2d
	      
====================
 Indices and tables
====================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

