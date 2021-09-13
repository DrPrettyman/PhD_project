#
# plotting the figures in Chapter 2 of the thesis
#
"""
The module ``figures_c2`` contains several functions with names like
``fig01()`` or ``tab01()`` that reproduce the experiments and the
resulting figures or tables from chapter 2 of the thesis (one-dimensional
tipping point techniques). These rely upon functions in other modules
particularly, for plotting the figures, the :class:`plot_helper.ThesisPlot` class.
"""
#
# import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.polynomial import polyfit
from numpy.polynomial.polynomial import polyval
from tippingpoints import scaling_methods
from tippingpoints import noise_methods
from thesisfigures.plot_helper import ThesisPlot


def fig01():
    """
    Plots fig 2.1 of the thesis (page 42).

    Analysis of artificial red noise with scaling exponents measured using three
    different methods.

    Panel a: Red noise is generated using the method shown in equation 2.8.
    Panel b: The ACF of the red noise data is calculated for different lags and
    the exponent (negative slope) measured in the range :math:`10 ≤ s ≤ 100` (dashed lines).

    We note that the ACF1 indicator (C(1)) is 0.84.
    The ACF of a white noise series is also plotted for comparison,
    in this case :math:`C(s) = 0` for :math:`s ≥ 1` and the exponent is also zero.

    Panel c: DFA calculated for the data and the exponent (slope) measured in the range
    :math:`10 ≤ s ≤ 100`.

    Panel d: The power spectrum of the data, and the exponent (negative slope)
    measured in the frequency range :math:`10^{-2} ≤ f ≤ 10^{-1}`.

    """
    return


def fig02():
    """
    Plots fig 2.2 of the thesis (page 44).

    The detrending step in the order-2 DFA algorithm (equation 2.11).

    The cumulative sum of a pink-noise time series :math:`z(t)` is shown with
    the quadratic best fit (red line) in each segment of length 20
    (marked by dashed vertical lines).
    :return:
    """
    return


def fig03(n: int = 10**4):
    """
    Plots fig 2.3 of the thesis (page 48).

    This is a simple demonstration of fitting a linear best-fit
    to a periodogram, in this case the periodograms for a
    white noise series and a random walk, both of length *n*.

    :param n: length of series
    """
    # We create a simple Gaussian white noise series and a random walk, both of length n
    t = np.linspace(0, 1, n)
    white_noise = noise_methods.white_noise(n, eta=1)
    random_walk = noise_methods.random_walk(n, eta=1)

    # calculate the PS exponent and psdx of each series using scaling_methods.pse()
    # in the range :math:`-2<\log(f)<-1`
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

    # plot the figure
    p = ThesisPlot(figure_number='2.3', pattern=[[0, 0, 2]])

    p.plot(t, white_noise, 0, line_style='line3')
    p.plot(t, random_walk, 0, line_style='line1')

    p.plot(w_freq, w_psdx_adj, 1, line_style='line3')
    p.plot(w_freq, w_fit_adj, 1, line_style='dash_red')

    p.plot(r_freq, r_psdx_adj, 2, line_style='line1')
    p.plot(r_freq, r_fit_adj, 2, line_style='dash_red')

    p.axes_labels(0, 't', 'z(t)')
    p.axes_labels(1, 'log(f)', 'spectral density: log S(f)')
    p.axes_labels(2, 'log(f)', 'spectral density: log S(f)')

    p.show()
    return


def fig04():
    """
    Plots fig 2.4 of the thesis (page 57).

    DFA exponent α plotted against the PS exponent :math:`β` for short-range
    correlated (panel a) and long-range correlated (panel b) noise
    series of length 104 with varying correlation parameters.

    The result for each noise series is represented by one marker,
    the expected linear relationship, shown in red, is :math:`α = (1+β)/2`
    (see equation 2.43).
    """
    return


def fig05():
    """
    Plots fig 2.5 of the thesis (page 58).

    Standard deviation of the ACF and DFA scaling exponents over
    50 noise series of length 1000, for each noise correlation
    parameter :math:`λ = 0, 0.02, 0.04, ..., 1`.

    The mean values of the exponents range between 0 and 1.2 (for ACF),
    and between 0.5 and 1 (for DFA).
    """
    return


def fig06(n: int = 10**4, no_tests: int = 200):
    """
    Plots fig 2.6 of the thesis (page 59).

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

    p = ThesisPlot(figure_number='2.6', pattern=[[0, 2]])

    # Plot ACF1 and ACF exponent values of the short-range data
    # on the first axis
    p.plot(ar1_param_values, acf1_values_1, axes_no=0, line_style='line1')
    p.plot(ar1_param_values, acfe_values_1, axes_no=0, line_style='line2')

    # Plot ACF1 and ACF exponent values of the long-range data
    # on the second axis
    p.plot(ar63_param_values, acf1_values_63, axes_no=1,
           line_style='line1', label='lag-1 ACF')
    p.plot(ar63_param_values, acfe_values_63, axes_no=1,
           line_style='line2', label='ACF exponent')
    p.legend(1)

    p.axes_labels(0, xlabel='AR(1) model parameter', ylabel='Exponent value')
    p.axes_labels(1, xlabel='AR(63) model parameter', ylabel='Exponent value')
    p.share_ylims()
    p.show()
    return


def fig07(n: int = 10**4, no_tests: int = 10):
    """
    Plots fig 2.7 of the thesis (page 67).

    A plot of the analytically derived relationship between the
    AR(1) model parameter mu and the PS scaling exponent beta,
    alongside a scatter plot of the experimentally obtained
    relationship. The two plots should show a similar pattern.

    The analytical relationship between beta and mu is given
    by eqn2.86 of in the thesis (page 65)::

        beta = log10[(1 + mu^2 - 2*mu*cos(0.2*pi)) / (1 + mu^2 - 2*mu*cos(0.02*pi))]

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

    p = ThesisPlot(figure_number='2.7', pattern=[[0, 0]])
    p.plot(mu_values, beta_expected, axes_no=0, line_style='line1')
    p.plot(acf1_values, beta_values, axes_no=1, line_style='scatter')
    p.axes_labels(0, xlabel=r'AR(1) model parameter $\mu$',
                  ylabel=r'Power spectrum exponent $\beta$')
    p.axes_labels(1, xlabel='ACF1(X)', ylabel='PS(X)')
    p.share_ylims()
    p.show()
    return


def fig08(n: int = 10**6):
    """
    Plots fig 2.8 of the thesis (page 69).

    A plot showing a power spectrum containing a 'crossover', in this
    case the periodogram of the time series :math:`z(t)`: the sum of a random walk
    :math:`W_t = W_{t-1} + \\zeta_t`
    and a white noise series :math:`\\eta_t`. The point at which the crossover occurs is
    determined by the std of the white noise.

    :param n: length of the time series ``z``
    """
    print('Setting parameters...')
    # mu is chosen so that the crossover will occur in the middle
    # of the frequency range :math:`10^{-2} ≤ f ≤ 10^{-1}`.
    mu = (10 ** (3/2))/(2 * np.pi)
    # The window limits are actually larger than the frequency range
    # so that we can see the whole picture in context.
    local_window_lims = (-3, -0.5)

    print('generating time series...')
    # We generate a random walk and a white noise series then add them together
    random_walk = noise_methods.random_walk(n, 1)
    white_noise = noise_methods.white_noise(n, mu)
    z = random_walk + white_noise

    print('Calculating the expected power spectra...')
    f = np.linspace(10 ** (local_window_lims[0]), 10 ** (local_window_lims[1]), n)
    f_log = np.log10(f)
    w_expected = np.log10((mu ** 2) * np.ones(n))
    r_expected = np.log10((1 / (4 * np.pi ** 2)) * f ** (-2))
    z_expected = np.log10((1 / (4 * np.pi ** 2)) * f ** (-2) + (mu / (2 * np.pi)) * f ** (-1) + mu ** 2)

    print('calculating PSD for white noise...')
    w_pse, w_freq, w_psdx = scaling_methods.pse(white_noise, window_limits=local_window_lims)
    print('calculating PSD for random walk...')
    r_pse, r_freq, r_psdx = scaling_methods.pse(random_walk, window_limits=local_window_lims)
    print('calculating PSD for combined series...')
    z_pse, z_freq, z_psdx = scaling_methods.pse(z, window_limits=local_window_lims)

    print('Plotting results...')
    p = ThesisPlot(pattern=[[0]], figure_number='2.8')
    p.plot(w_freq, w_psdx, line_style='thin')
    p.plot(r_freq, r_psdx, line_style='thin_red')
    p.plot(z_freq, z_psdx, line_style='thin_blue')
    p.plot(f_log, w_expected, line_style='dash')
    p.plot(f_log, r_expected, line_style='dash')
    p.plot(f_log, z_expected, line_style='thick')
    p.axes_labels(xlabel=r'log[f]', ylabel=r'log[S(f)]')
    p.print_figure_properties()
    p.save_figure('png')
    p.show()
    return


def ar1_ps(f, mu, sigma):
    """
    A function of frequency *f*, AR parameter *mu* and noise level *sigma*.

    :param f:
    :param mu:
    :param sigma:
    :return:
    """
    return (sigma**2) / (1 + mu**2 - 2*mu*np.cos(2*np.pi*f))


def ar1_ps_indicator(f, mu):
    return (4*np.pi*mu*f*np.sin(2*np.pi*f)) / (1 + mu**2 - 2*mu*np.cos(2*np.pi*f))


def fig10():
    """
    Plots fig 2.10 of the thesis (page 76).

    Panel a: The power spectrum of the AR(1) process (see equation 2.81)
    is plotted on a log-log scale for various values of the parameter μ.
    Note the ‘white-noise’ (flat) part of the power spectrum for small f
    and the ’red- noise’ (negative gradient) part for large f .

    Panel b: The PS indicator (see equation 2.84) is plotted as a function
    of f for the same μ values.

    :return:
    """
    mu_values = np.array([0.7, 0.8, 0.9, 0.999])

    log_f = np.linspace(-3.5, -0.3, 10**3)
    f = 10**log_f

    psind_values = np.zeros((len(f), len(mu_values)))
    psdx_values = np.zeros((len(f), len(mu_values)))

    for _i in range(len(mu_values)):
        psind_values[:, _i] = ar1_ps_indicator(f, mu_values[_i])
        psdx_values[:, _i] = np.log10(ar1_ps(f, mu_values[_i], 1))

    p = ThesisPlot(figure_number='2.10', pattern=[[0, 0]])

    _styles = ['thin_red', 'thin_green', 'line1', 'thin_blue']
    for _i in range(len(mu_values)):
        p.plot(log_f, psdx_values[:, _i], axes_no=0, line_style=_styles[_i],
               label=fr'$\mu =$ {mu_values[_i]}')
        p.plot(log_f, psind_values[:, _i], axes_no=1, line_style=_styles[_i])
    p.legend(0)

    p.v_dash(1, -2)
    p.v_dash(1, -1)

    p.axes_labels(0, xlabel='log[f]', ylabel='log[S(f)]')
    p.axes_labels(1, xlabel='log[f]', ylabel=r'PS indicator, $B_f(\mu)$')

    p.show()
    return


def fig11():
    """"
    The PS indicator is plotted as a function of μ for various
    values of f . Note that for larger f the PS indicator has
    a maximum value < 2 while for smaller f the indicator shows
    the characteristic increasing (’reddening’) trend
    only in the μ > 0.9 range.
    """

    mu = np.linspace(0, 1, 10*4)
    logf_values = np.array([-2.5, -2., -1.5, -1., -0.5])

    psind_values = np.zeros((len(mu), len(logf_values)))
    for _i in range(len(logf_values)):
        psind_values[:, _i] = ar1_ps_indicator(10**logf_values[_i], mu)

    p = ThesisPlot(figure_number='2.11', pattern=[[0]])

    _styles = ['r', 'g', 'k', 'b', 'm']
    for _i in range(len(logf_values)):
        p.plot(mu, psind_values[:, _i], line_style=_styles[_i],
               label=f'log(f) = {logf_values[_i]}')
    p.legend(0)

    p.axes_labels(0, xlabel=r'$\mu$', ylabel=r'PS indicator, $B_f(\mu)$')

    p.show()
    return
