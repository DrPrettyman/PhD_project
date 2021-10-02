"""
The module ``scaling_methods`` provides various functions for
calculating scaling properties of time series, in particular calculating:
- lag-1 autocorrelation function
- DFA scaling exponent
- Power Spectrum scaling exponent

And, in particular, each of these calculated in a sliding window.
"""
#
#
#

import numpy as np
import matplotlib.pyplot as plt
from tippingpoints import noise_methods
from numpy.polynomial.polynomial import polyfit
from numpy.polynomial.polynomial import polyval

# ****************************************
# ** "Sliding window" indicator methods **
# ****************************************


def acf_sliding(t: np.ndarray, z: np.ndarray, window_size: int = 200,
                lag: int = 1, increment: int = 1):
    """
    For a time series :math:`z(t)` we select a slice of the data defined by
    a sliding window e.g. for ``window_size=200`` the first data slice would
    be ``z[0:200]``.

    We then measure the ACF (usually lag-1 ACF) of this slice and progress
    the window by an increment, e.g. for ``increment=1`` the second data slice
    would be ``z[1:201]``.

    This function returns ``y_out``: the series of these ACF values, and
    ``t_out``, the corresponding time variable.

    :param t: time variable
    :param z: dependent variable
    :param window_size: length of the sliding window
    :param lag: the lag used in the ACF calculation
    :param increment: the increment by which the window is progressed at each step

    :returns: ``numpy.ndarray``: t_out, y_out
    """
    y = np.zeros(z.shape[0])
    test_range = range(window_size - 1, z.shape[0], increment)
    for i in test_range:
        window = z[(i+1 - window_size):i+1]
        y[i] = acf(window, lag=lag)
    t_out = np.array([t[i] for i in test_range])
    y_out = np.array([y[i] for i in test_range])
    return t_out, y_out


def dfa_sliding(t: np.ndarray, z: np.ndarray, window_size: int = 200,
                order: int = 2, no_segment_lengths: int = 8,
                increment: int = None):
    """
    For a time series :math:`z(t)` we select a slice of the data defined by
    a sliding window e.g. for ``window_size=200`` the first data slice would
    be ``z[0:200]``.

    We then measure the DFA exponent of this slice and progress
    the window by an increment, e.g. for ``increment=1`` the second data slice
    would be ``z[1:201]``. The increment, if not user-specified, is set at
    :math:`0.2` times the square root of the series length.

    This function returns ``y_out``: the series of these ACF values, and
    ``t_out``, the corresponding time variable.

    :param t: time variable
    :param z: dependent variable
    :param window_size: length of the sliding window
    :param order: the polynomial order used in the DFA exponent calculation
    :param increment: the increment by which the window is progressed at each step

    :returns: ``numpy.ndarray``: t_out, y_out
    """
    if increment is None:
        increment = int(0.2 * np.sqrt(z.shape[0]))
        if increment < 1:
            increment = 1
    y = np.zeros(z.shape[0])
    test_range = range(window_size - 1, z.shape[0], increment)
    for i in test_range:
        window = z[(i+1 - window_size):i+1]
        y[i] = dfa(window, order=order, no_segment_lengths=no_segment_lengths)
    t_out = np.array([t[i] for i in test_range])
    y_out = np.array([y[i] for i in test_range])
    return t_out, y_out


def pse_sliding(t: np.ndarray, z: np.ndarray, window_size: int = 200,
                binning: bool = True, window_limits: (float, float) = (-2, -1),
                increment: int = 1):
    """
    For a time series :math:`z(t)` we select a slice of the data defined by
    a sliding window e.g. for ``window_size=200`` the first data slice would
    be ``z[0:200]``.

    We then measure the power spectrum scaling exponent of this slice and progress
    the window by an increment, e.g. for ``increment=1`` the second data slice
    would be ``z[1:201]``.

    This function returns ``y_out``: the series of these ACF values, and
    ``t_out``, the corresponding time variable.

    :param t: time variable
    :param z: dependent variable
    :param window_size: length of the sliding window
    :param binning: whether or not to use logarithmic binning on the periodogram
                    as part of the PS exponent calculation
    :param increment: the increment by which the window is progressed at each step

    :returns: ``numpy.ndarray``: t_out, y_out
    """
    y = np.zeros(z.shape[0])
    test_range = range(window_size - 1, z.shape[0], increment)
    for i in test_range:
        window = z[(i+1 - window_size):i+1]
        y[i], freq, psdx = pse(window, binning=binning, window_limits=window_limits)
    t_out = np.array([t[i] for i in test_range])
    y_out = np.array([y[i] for i in test_range])
    return t_out, y_out


# ***********************************
# ** Scaling exponent calculations **
# ***********************************


def acf(z: np.ndarray, lag: int = 1):
    """
    Calculates the autocorrelation function of a time series :math:`z`.

    :param z: Time series
    :param lag: The autocorrelation lag

    :return: ``float``: autocorrelation function of a time series ``z``
    """
    d1 = z[:-lag] - z.mean(0)
    d2 = z[lag:] - z.mean(0)
    return (d1 * d2).mean(0) / z.var(0)


def acf_scaling(z: np.ndarray):
    """
    Calculates the autocorrelation scaling exponent of a time series :math:`z`.

    This calculation involves calculating the ACF with lag :math:`l`
    for :math:`l = 10, 11, 12, ..., 100` and then calculating the negative gradient
    of the log-log plot :math:`l` against ACF.

    :param z: The time series

    :return: ``float``: autocorrelation scaling exponent of a time series ``z``
    """
    lag_values = np.arange(10, 100, 1)
    acf_values = np.zeros(len(lag_values))
    for i in range(len(lag_values)):
        acf_values[i] = acf(z, lag=lag_values[i])
        if acf_values[i] < 0.01:
            acf_values[i] = 0.01
    lag_values_log = np.log10(lag_values)
    acf_values_log = np.log10(acf_values)
    linear_fit_poly = polyfit(lag_values_log, acf_values_log, deg=1)
    return -linear_fit_poly[1]


def dfa(z: np.ndarray, order: int = 2, no_segment_lengths: int = 8, view: bool = False):
    s_min = 10
    s_max = z.shape[0] // 4
    segment_lengths = np.floor(10. ** np.linspace(np.log10(s_min), np.log10(s_max), no_segment_lengths))
    f_values = np.zeros_like(segment_lengths)
    for i in range(len(f_values)):
        f = dfa_f_coefficient(z, segment_length=int(segment_lengths[i]), order=order)
        f_values[i] = f / np.sqrt(segment_lengths[i])
    linear_fit_poly = polyfit(np.log10(segment_lengths), np.log10(f_values), deg=1)
    if view:
        fig, ax = plt.subplots()
        ax.plot(np.log10(segment_lengths), np.log10(f_values), )
        plt.show()
    return linear_fit_poly[1] + 0.5


def pse(z: np.ndarray, binning: bool = True, window_limits: (float, float) = (-2, -1)):
    fs = 1.
    xdft = np.fft.fft(z)[range(z.shape[0]//2+1)]
    psdx = (1. / (fs * z.shape[0])) * abs(xdft) ** 2
    psdx[1:-1] = 2. * psdx[1:-1]
    freq = np.linspace(0, fs / 2, z.shape[0]//2+1)
    if psdx.shape != freq.shape:
        raise RuntimeError(f'psdx and freq do not match. shapes = {psdx.shape}, {freq.shape}')
    # filter out any non-positive values before taking logs
    positive_indices = [i for i in range(freq.shape[0])
                        if freq[i] > 0 and psdx[i] > 0]
    freq_log = np.log10(freq[positive_indices])
    psdx_log = np.log10(psdx[positive_indices])
    # Trim the data to only the window
    freq_log, psdx_log = trim_to_window(freq_log, psdx_log, window_limits)
    # perform a linear fit to the logarithmic psdx,
    # with logbinning first if requested
    if binning:
        freq_log, psdx_log = logbin(freq_log, psdx_log)
    linear_fit_poly = polyfit(freq_log, psdx_log, deg=1)
    return -linear_fit_poly[1], freq_log, psdx_log


# ************************
# ** Additional methods **
# ************************


def dfa_f_coefficient(z: np.ndarray, segment_length: int, order: int = 2):
    y = z.cumsum(0)
    n_segments = int(z.shape[0] // segment_length)
    forward_plot = np.zeros(z.shape[0])
    backward_plot = np.zeros(z.shape[0])
    fs_forward = np.zeros(n_segments)
    fs_backward = np.zeros(n_segments)
    for nu in range(n_segments):
        segment_indices = \
            np.arange(nu * segment_length, (nu + 1) * segment_length)
        segment_fit = \
            polyval(segment_indices, polyfit(segment_indices, y[segment_indices], deg=order))
        forward_plot[segment_indices] = segment_fit
        y_s = y[segment_indices] - segment_fit
        fs_forward[nu] = (1. / segment_length) * (y_s ** 2).sum()
    for nu in range(n_segments):
        segment_indices = \
            z.shape[0] - 1 - np.arange(nu * segment_length, (nu + 1) * segment_length)
        segment_fit = \
            polyval(segment_indices, polyfit(segment_indices, y[segment_indices], deg=order))
        backward_plot[segment_indices] = segment_fit
        y_s = y[segment_indices] - segment_fit
        fs_backward[nu] = (1. / segment_length) * (y_s ** 2).sum()
    return np.sqrt((fs_forward.sum() + fs_backward.sum()) / (2 * n_segments))


def logbin(t_log: np.ndarray, z_log: np.ndarray):
    if t_log.shape != z_log.shape:
        raise ValueError('t_log and z_log must have the same shape')
    elif t_log.shape[0] <= 10:
        raise ValueError('t_log must be longer than 10 elements')
    # Split the window into subwindows
    # First establish how many windows to use
    no_subwindows = 10
    while True:
        subwindow_length = (t_log[-1] - t_log[0]) / no_subwindows
        subwindow1 = np.nonzero(t_log < t_log[0] + subwindow_length)
        if len(subwindow1[0]) > 2:
            break
        else:
            no_subwindows -= 1
        if no_subwindows < 2:
            raise ValueError('Insufficient resolution in t_log:'
                             ' cannot establish first sub-window.')
    # Now split the window into sub-windows
    subwindow_dict = {'index': [], 'lt': [], 'lz': [],
                      't_log_new': [], 'z_log_new': [], 'part_len': []}
    for i in range(no_subwindows):
        index = np.nonzero((t_log >= t_log[0] + i * subwindow_length) &
                           (t_log < t_log[0] + (i + 1) * subwindow_length))
        subwindow_dict['index'].append(index)
        subwindow_dict['lt'].append(t_log[index])
        subwindow_dict['lz'].append(z_log[index])
    # Partition:
    # For each subwindow(after the first), reduce the
    # number of points to the same as in the first subwindow
    # by partitioning and taking the mean of LZ in each partition.
    # Each subwindow should be split into 'size(subwindow_dict(1).LT,1)' partitions
    subwindow_dict['t_log_new'].append(subwindow_dict['lt'][0])
    subwindow_dict['z_log_new'].append(subwindow_dict['lz'][0])
    no_parts = len(subwindow_dict['lt'][0])
    sw_start = subwindow_dict['lt'][0][0]
    sw_end = subwindow_dict['lt'][0][-1]
    subwindow_dict['part_len'].append((sw_end - sw_start) / no_parts)
    for i in range(1, no_subwindows):
        sw_start = subwindow_dict['lt'][i][0]
        sw_end = subwindow_dict['lt'][i][-1]
        subwindow_dict['part_len'].append((sw_end - sw_start) / no_parts)
        z_log_new = np.zeros([no_parts, 1])
        t_log_new = np.zeros([no_parts, 1])
        for k in range(no_parts):
            lt_0 = sw_start + k * subwindow_dict['part_len'][i]
            lt_1 = lt_0 + subwindow_dict['part_len'][i]
            part_indices = np.nonzero((subwindow_dict['lt'][i] >= lt_0) &
                                      (subwindow_dict['lt'][i] <= lt_1))
            t_log_new[k] = subwindow_dict['lt'][i][part_indices].mean()
            z_log_new[k] = subwindow_dict['lz'][i][part_indices].mean()
        subwindow_dict['t_log_new'].append(t_log_new.reshape(-1))
        subwindow_dict['z_log_new'].append(z_log_new.reshape(-1))
    # join up all the sub-windows
    t_log_new = np.zeros([no_subwindows * no_parts])
    z_log_new = np.zeros([no_subwindows * no_parts])
    for i in range(no_subwindows):
        new_window_index = np.arange(i * no_parts, (i + 1) * no_parts)
        t_log_new[new_window_index] = subwindow_dict['t_log_new'][i]
        z_log_new[new_window_index] = subwindow_dict['z_log_new'][i]
    # If there are any non-finite values in either array (t_log_new or z_log_new)
    # we filter them out.
    if not (all(np.isfinite(t_log_new)) and all(np.isfinite(z_log_new))):
        finite_indices = [i for i in range(t_log_new.shape[0])
                          if np.isfinite(t_log_new[i])
                          and np.isfinite(z_log_new[i])]
        t_log_new = t_log_new[finite_indices]
        z_log_new = z_log_new[finite_indices]
    return t_log_new, z_log_new


def trim_to_window(t: np.ndarray, z: np.ndarray, window_limits: (float, float)):
    """
    :param t: time variable eg. np.linspace(0, 1, 1000)
    :param z: dependent time series variable
    :param window_limits: lower and upper bounds in t
    :return:
    """
    window_indices = \
        np.nonzero((t >= window_limits[0]) & (t <= window_limits[1]))
    t_out = t[window_indices]
    z_out = z[window_indices]
    return t_out, z_out


# ***********
# ** Tests **
# ***********


def acf_sliding_test():
    n = 10 ** 4
    red_noise = noise_methods.random_walk(n)
    t = np.linspace(0, 1, n)
    t_out, y = acf_sliding(t, red_noise, 100)
    print('len(t_out) = {0}, len(y) = {1}'.format(len(t_out), len(y)))
    return


def acf_test():
    """
    If we have AR(1) models with different parameters mu
    the output from the acf method with lag 1 should be equal to mu
    """
    print('')
    print('ACF method test:')
    print('')
    print('white noise / random walk test')
    n = 10 ** 5
    white_noise = np.random.randn(n)
    red_noise = noise_methods.random_walk(n)
    ar1_0 = noise_methods.ar1(n, mu=0, eta=1)
    ar1_1 = noise_methods.ar1(n, mu=1, eta=1)
    mu_w = acf(white_noise, 1)
    mu_r = acf(red_noise, 1)
    mu_ar1_0 = acf(ar1_0, 1)
    mu_ar1_1 = acf(ar1_1, 1)
    print('ACF1 for white noise = {}'.format(mu_w))
    print('ACF1 for red noise = {}'.format(mu_r))
    print('ACF1 for AR1 (mu=0) = {}'.format(mu_ar1_0))
    print('ACF1 for AR1 (mu=1) = {}'.format(mu_ar1_1))
    print('')
    print('AR(1) model relationship tests:')
    n = 10 ** 4
    no_tests = 200
    mu_values = np.linspace(0, 1, no_tests)
    acf_values = np.zeros(no_tests)
    for i in range(no_tests):
        z = noise_methods.ar1(n, mu=mu_values[i], eta=1.)
        acf_values[i] = acf(z, 1)
    fig, ax = plt.subplots()
    ax.plot(mu_values, acf_values, )
    plt.show()
    return


def dfa_test():
    """
    DFA method should return the correct alpha values for the AR(1) model
    """
    print('')
    print('DFA method test:')
    print('')
    print('white noise / random walk test')
    n = 10 ** 4
    white_noise = np.random.randn(n)
    red_noise = noise_methods.random_walk(n)
    ar1_0 = noise_methods.ar1(n, mu=0, eta=1)
    ar1_1 = noise_methods.ar1(n, mu=1, eta=1)
    mu_w = dfa(white_noise)
    mu_r = dfa(red_noise, view=True)
    mu_ar1_0 = dfa(ar1_0)
    mu_ar1_1 = dfa(ar1_1)
    print('DFA for white noise = {}'.format(mu_w))
    print('DFA for red noise = {}'.format(mu_r))
    print('DFA for AR1 (mu=0) = {}'.format(mu_ar1_0))
    print('DFA for AR1 (mu=1) = {}'.format(mu_ar1_1))
    return


def dfa_f_coefficient_test():
    """
    In MATLAB the dfa F coefficient method returns the following
    F values for a random walk with varying length n, segment_length
    and order:
    """
    print(' \n DFA f_coefficient method test:\n')
    n = 10 ** 4
    z = (np.random.randn(n)).cumsum()
    f = dfa_f_coefficient(z, segment_length=15, order=3)
    print('F_expected = 0.73')
    print('F_obtained = {}'.format(f))
    print('')

    n = 10 ** 4
    z = (np.random.randn(n)).cumsum()
    f = dfa_f_coefficient(z, segment_length=10, order=2)
    print('F_expected = 0.65')
    print('F_obtained = {}'.format(f))
    print('')

    n = 10 ** 5
    z = (np.random.randn(n)).cumsum()
    f = dfa_f_coefficient(z, segment_length=30, order=2)
    print('F_expected = 3.31')
    print('F_obtained = {}'.format(f))
    print('')
    return


def pse_test():
    """
    If we have AR(1) models with different parameters mu
        and evaluate the pse value beta for each
    Then the mu and beta values should follow the curve:
        mu = b - sqrt(b^2 - 1) where
        b = [cos(0.2*pi) - 10^beta * cos(0.2*pi)] / [1 - 10^beta]
    """
    print('')
    print('PSE method test:')
    print('')
    print('white noise / random walk test')
    n = 10 ** 5
    white_noise = np.random.randn(n)
    red_noise = noise_methods.random_walk(n)
    ar1_0 = noise_methods.ar1(n, mu=0, eta=1)
    ar1_1 = noise_methods.ar1(n, mu=1, eta=1)
    beta_w, f, pdsx = pse(white_noise)
    beta_r, f, pdsx = pse(red_noise)
    beta_ar1_0, f, pdsx = pse(ar1_0)
    beta_ar1_1, f, pdsx = pse(ar1_1)
    print('PSE for white noise = {}'.format(beta_w))
    print('PSE for red noise = {}'.format(beta_r))
    print('PSE for AR1 (mu=0) = {}'.format(beta_ar1_0))
    print('PSE for AR1 (mu=1) = {}'.format(beta_ar1_1))
    print('')
    print('AR(1) model relationship tests:')
    n = 10 ** 4
    no_tests = 200
    mu_values = np.linspace(0, 1, no_tests)
    beta_values = np.zeros(no_tests)
    beta_expected = np.zeros(no_tests)
    for i in range(no_tests):
        z = noise_methods.ar1(n, mu=mu_values[i], eta=1.)
        beta_values[i], f, pdsx = pse(z)
        beta_expected[i] = np.log10(
            (1 + mu_values[i] ** 2 - 2 * mu_values[i] * np.cos(0.2 * np.pi)) /
            (1 + mu_values[i] ** 2 - 2 * mu_values[i] * np.cos(0.02 * np.pi)))
    fig, ax = plt.subplots()
    ax.plot(mu_values, beta_values, )
    ax.plot(mu_values, beta_expected, )
    plt.show()
    return


def plt_test():
    n = 10 ** 3
    t = np.linspace(0, 1, n)
    z1 = np.sin(5 * t)
    z2 = np.cos(2 * t) + 0.1 * np.random.randn(n)
    z3 = np.random.randint(-1, 2, n)

    fig, ax = plt.subplots(3)
    ax[0].plot(t, z1, )
    ax[1].plot(t, z2, )
    ax[2].plot(t, z3, )

    ax[2].set(xlabel='time (t)')
    ax[0].grid()
    plt.show()
    return
