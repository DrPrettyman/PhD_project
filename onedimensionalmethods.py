# plotting the figures in Chapter 2 of the thesis
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.polynomial import polyfit
from numpy.polynomial.polynomial import polyval
from tippingpoints import scaling_methods
from tippingpoints import noise_methods


class Plot1by2:
    """
    Class initialises a figure with subplots side-by-side and includes
    helper functions for plotting.
    The original pyplot functionality can be accessed directly through
    the object variables 'fig', 'ax1' and 'ax2', as in
    p = Plot1by2()
    p.ax1.set_xlim(0, 10)
    """
    def __init__(self, figure_number: str = '0', share_y_axis: bool = True):
        self.share_y_axis = share_y_axis
        plt.style.use('./thesis.mplstyle')
        self.standard_line_dict = {'line1': {'color': 'k', 'lw': 1.2, 'ls': '-'},
                                   'line2': {'color': (0.4, 0.4, 0.4), 'lw': 1.2, 'ls': '-'},
                                   'line3': {'color': (0.7, 0.7, 0.7), 'lw': 1.2, 'ls': '-'},
                                   'thin': {'color': (0.2, 0.2, 0.2), 'lw': 0.7, 'ls': '-'},
                                   'thin_red': {'color': 'r', 'lw': 0.7, 'ls': '-'},
                                   'thin_blue': {'color': 'b', 'lw': 0.7, 'ls': '-'},
                                   'scatter': {'color': (0.1, 0.1, 0.1), 'lw': 1, 'ls': '', 'marker': 'o',
                                               'markersize': 5},
                                   'dash': {'color': (0.1, 0.1, 0.1), 'lw': 2, 'ls': '--'},
                                   'dash_red': {'color': 'r', 'lw': 2, 'ls': '--'}}
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2)
        self.fig.canvas.manager.set_window_title(f'Thesis_JPrettyman figure {figure_number}')
        self.ax1.tick_params('both', top=True, bottom=True, left=True, right=True, direction='in')
        self.ax2.tick_params('both', top=True, bottom=True, left=True, right=True, direction='in')
        self.ax2.tick_params('y', labelleft=False)

        self.ax1.annotate(text='a', xy=(0, 0), xytext=(0.05, 0.93), xycoords='axes fraction', fontsize=20)
        self.ax2.annotate(text='b', xy=(0, 0), xytext=(0.05, 0.93), xycoords='axes fraction', fontsize=20)

    def plot_a(self, x: np.ndarray, y: np.ndarray, ls='line1', label: str = None):
        self.ax1.plot(x, y, **self.standard_line_dict[ls], label=label)

    def plot_b(self, x: np.ndarray, y: np.ndarray, ls='line1', label: str = None):
        self.ax2.plot(x, y, **self.standard_line_dict[ls], label=label)

    def axes_labels(self, x1: str = '', y1: str = '', x2: str = '', y2: str = ''):
        self.ax1.set(xlabel=x1, ylabel=y1)
        self.ax2.set(xlabel=x2, ylabel=y2)

    def show(self):
        if self.share_y_axis:
            ylim1 = self.ax1.get_ylim()
            ylim2 = self.ax2.get_ylim()
            ylim_new = (min(ylim1[0], ylim2[0]), max(ylim1[1], ylim2[1]))
            self.ax1.set_ylim(ylim_new)
            self.ax2.set_ylim(ylim_new)
        plt.show()


def fig01():
    return


def fig02():
    return


def fig03(n: int = 10**4):
    """
    Plots fig 2.3 of the thesis.

    This is a simple demonstration of fitting a linear best-fit
    to a periodogram, in this case the periodograms for a
    white noise series and a random walk, both of length 'n'

    :param n: length of series
    """
    # We create a simple Gaussian white noise series and a random walk, both of length n
    white_noise = noise_methods.white_noise(n, eta=1)
    random_walk = noise_methods.random_walk(n, eta=1)

    # calculate the PS exponent and psdx of each series using scaling_methods.pse()
    w_pse, w_freq, w_psdx =\
        scaling_methods.pse(white_noise, binning=True, window_limits=(-2, -1))
    r_pse, r_freq, r_psdx = \
        scaling_methods.pse(random_walk, binning=True, window_limits=(-2, -1))

    # print PS exponent
    print(f'PS exponent value for white noise: {w_pse}')
    print(f'PS exponent value for random walk: {r_pse}')

    # calculate the best fit line
    w_fit_poly = polyfit(w_freq, w_psdx, deg=1)
    r_fit_poly = polyfit(r_freq, r_psdx, deg=1)
    w_fit = polyval(w_freq, w_fit_poly)
    r_fit = polyval(r_freq, r_fit_poly)

    # we adjust the psdx and linear fit to fit on the same axes,
    # without loss of the important information (gradient)
    w_psdx_adj = w_psdx - w_fit[0]
    r_psdx_adj = r_psdx - r_fit[0]
    w_fit_adj = w_fit - w_fit[0]
    r_fit_adj = r_fit - r_fit[0]

    p = Plot1by2(figure_number='2.3')
    p.plot_a(w_freq, w_psdx_adj, 'line1')
    p.plot_b(r_freq, r_psdx_adj, 'line1')
    p.plot_a(w_freq, w_fit_adj, 'dash_red')
    p.plot_b(r_freq, r_fit_adj, 'dash_red')
    p.show()
    return


def fig04():
    return


def fig05():
    return


def fig06(n: int = 10**4, no_tests: int = 200):
    """
    Plots fig 2.6 of the thesis.

    A comparison of the lag-1 ACF and the ACF scaling exponent
    The comparison uses, as example time series, 200 short-range
    correlated (AR(1)) models, and 200 long-range correlated (AR(63))
    models with varying parameters.

    :param n:        length of each generated AR(1) and AR(63) series
    :param no_tests: number of tests for each model
    """
    ar1_param_values = np.linspace(0, 1, no_tests)
    ar63_param_values = np.linspace(0, 2, no_tests)

    acfe_values_1 = np.zeros(no_tests)
    acfe_values_63 = np.zeros(no_tests)
    acf1_values_1 = np.zeros(no_tests)
    acf1_values_63 = np.zeros(no_tests)

    for i in range(no_tests):
        z_1 = noise_methods.ar1(n, mu=ar1_param_values[i])
        z_63 = noise_methods.ar63(n, lamb=ar63_param_values[i])
        print(f'calculating acf values for test {i}')
        acf1_values_1[i] = scaling_methods.acf(z_1)
        acf1_values_63[i] = scaling_methods.acf(z_63)
        acfe_values_1[i] = scaling_methods.acf_scaling(z_1)
        acfe_values_63[i] = scaling_methods.acf_scaling(z_63)

    p = Plot1by2(figure_number='2.6')

    # Plot ACF1 and ACF exponent values of the short-range data
    # on the first axis
    p.plot_a(ar1_param_values, acf1_values_1, 'line1')
    p.plot_a(ar1_param_values, acfe_values_1, 'line2')

    # Plot ACF1 and ACF exponent values of the long-range data
    # on the second axis
    p.plot_b(ar63_param_values, acf1_values_63, 'line1', label='lag-1 ACF')
    p.plot_b(ar63_param_values, acfe_values_63, 'line2', label='ACF exponent')
    p.ax2.legend()

    p.axes_labels(x1='AR(1) model parameter',
                  x2='AR(63) model parameter',
                  y1='Exponent value')

    p.show()

    return


def fig07(n: int = 10**4, no_tests: int = 10):
    """
    Plots fig 2.7 of the thesis (page 67)

    A plot of the analytically derived relationship between the
    AR(1) model parameter mu and the PS scaling exponent beta,
    alongside a scatter plot of the experimentally obtained
    relationship. The two plots should show a similar pattern.

    The analytical relationship between beta and mu is given
    by eqn2.86 of in the thesis (page 65):
    beta = log10[(1 + mu^2 - 2*mu*cos(0.2*pi)) /
                    (1 + mu^2 - 2*mu*cos(0.02*pi))]
    The relationship is obtained experimentally by generating
    several AR(1) series with different parameter mu and, for
    each series, calculating the lag-1 ACF (which reconstructs
    the value mu) and the numerical PS exponent.

    :param n:        length of each generated AR(1) series
    :param no_tests: number of mu values to be tested between 0 and 1
    """
    mu_values = np.linspace(0, 1, no_tests)
    beta_values = np.zeros(no_tests)
    beta_expected = np.zeros(no_tests)
    acf1_values = np.zeros(no_tests)
    for i in range(no_tests):
        z = noise_methods.ar1(n, mu=mu_values[i], eta=1.)
        beta_values[i], freq, pdsx = scaling_methods.pse(z)
        acf1_values[i] = scaling_methods.acf(z, lag=1)
        beta_expected[i] = np.log10(
            (1 + mu_values[i] ** 2 - 2 * mu_values[i] * np.cos(0.2 * np.pi)) /
            (1 + mu_values[i] ** 2 - 2 * mu_values[i] * np.cos(0.02 * np.pi)))

    p = Plot1by2(figure_number='2.7')
    p.plot_a(mu_values, beta_expected, 'line1')
    p.plot_b(acf1_values, beta_values, 'scatter')
    p.show()
    return
