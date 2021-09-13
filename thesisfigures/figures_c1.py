#
# plotting the figures in Chapter 1 of the thesis
#
"""
The module ``figures_c1`` contains several functions with names like
``fig01()`` or ``tab01()`` that reproduce the experiments and the
resulting figures or tables from chapter 1 of the thesis (Introduction).
These rely upon functions in other modules
particularly, for plotting the figures, the :class:`plot_helper.ThesisPlot` class.
"""
#
#
#
import numpy as np
from numpy.polynomial.polynomial import polyfit
from numpy.polynomial.polynomial import polyval
from tippingpoints import scaling_methods
from tippingpoints import noise_methods
from thesisfigures.plot_helper import ThesisPlot


