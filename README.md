# PhD_project
PhD project of Joshua Prettyman, University of Reading

Thesis completed November 2020 and published at 
[University of Reading](http://centaur.reading.ac.uk/98364/1/23022044_Prettyman_Thesis_Joshua%20Prettyman.pdf)

This project involves the application of established Early Warning Signal detection techniques
to time series of dynamical system variables containing tipping points, and an investigation of 
the pros and cons of a novel technique which uses the changing shape of the power spectrum close 
to a tipping point. 

The modules in the repo contain various techniques which allow the reproduction of experiments 
carried out during the PhD project, and many of the methods could equally be applied to other
data to draw fresh conclusions in the area of Early Warning Signals.

The individual modules in ``tippingpoints`` may be used to carry out certain techniques or methods developed during this project. 
For example, running ``scaling_methods.pse_sliding(t, z)`` will calculate the sliding-window power spectrum scaling exponent for time series ``z``.

In the modules ``figures_c1``, ``figures_c2`` there is the code ready to reproduce particular figures and tables from the thesis.
