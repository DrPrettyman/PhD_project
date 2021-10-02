#
#
#
"""
The module ``numerical_methods`` provides the Euler_Maruyama method and the Milstein method
to integrate a stochastic equation of the form :math:`dZ = a(Z)dt + \\sigma dW`
"""
#
#
#
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter


def dW(delta_t):
    """" Random sample normal distribution"""
    return np.random.normal(loc=0.0, scale=np.sqrt(delta_t))


def eulermaruyama(a, sigma: float, z_0: float = 0, t_0: float = 0, t_end: float = 1, n: int = 1000,
                  ensemble_size: int = 1, radius: tuple = (-10, 10)):
    """
    Integrates a stochastic equation of the form
    :math:`dZ = a(Z)dt + \\sigma dW` according to the Euler-Maruyama method

    Returns the time variable ``ts`` and the integrated series ``ys``.

    :param a:       function of t and z returning a single value
    :param sigma:   standard deviation of stochastic noise
    :param z_0:     initial condition in z
    :param t_0:     start time
    :param t_end:   end time
    :param n:       length of output. The integration time delta is therefore defined as ``(t_end - t_0) / n``
    :param ensemble_size:   for ensemble size :math:`m` the output ``ys`` will be a :math:`n` by :math:`m` array
    :param radius:          defines a range constraining the integration. The integration stops if `
                            `y < radius[0]`` or ``y > radius[1]`` in order to prevent blow-up

    :returns: ``numpy.ndarray``: ts, ys``
    """
    dt = float(t_end - t_0) / n
    # epsilon = 10 ** (-5)
    ts = np.arange(t_0, t_end + dt, dt)
    ys = np.zeros([ts.shape[0], ensemble_size])
    print(f'ts.shape = {ts.shape}, ys.shape = {ys.shape}')
    ys[0, :] = z_0
    print('Integrating SDE using Euler-Maruyama method:')
    bstart = perf_counter()
    for j in range(ensemble_size):
        print(f'    integrating ensemble member {j + 1} of {ensemble_size}... ', end='')
        astart = perf_counter()
        for i in range(1, ts.size):
            if i % 10**5 == 0:
                print(f'{i * 100 / ts.size}%')
            t = ts[i]
            y = ys[i - 1, j]
            if y < radius[0] or y > radius[1]:
                break
            ys[i, j] = y + a(t, y) * dt + sigma * dW(dt)
        aend = perf_counter()
        print(f'({aend - astart}s)')
    bend = perf_counter()
    print(f'    ...integration complete. Total time: {bend - bstart}seconds.')
    return ts, ys


def milstein_method(a, b, z_0: float = 0, t_0: float = 0, t_end: float = 1, n: int = 1000,
                    ensemble_size: int = 1, radius: tuple = (-10, 10)):
    """
    Integrates a stochastic equation of the form
    :math:`dZ = a(Z)dt + b(Z)dW` according to the Milstein method

    Returns the time variable ``ts`` and the integrated series ``ys``.

    :param a:       function of t and z returning a single value
    :param b:       function of t and z returning a single value
    :param z_0:     initial condition in z
    :param t_0:     start time
    :param t_end:   end time
    :param n:       length of output. The integration time delta is therefore defined as ``(t_end - t_0) / n``
    :param ensemble_size:   for ensemble size :math:`m` the output ``ys`` will be a :math:`n` by :math:`m` array
    :param radius:          defines a range constraining the integration. The integration stops if `
                            `y < radius[0]`` or ``y > radius[1]`` in order to prevent blow-up

    :returns: ``numpy.ndarray``: ts, ys``
    """
    dt = float(t_end - t_0) / n
    epsilon = 10 ** (-5)
    ts = np.arange(t_0, t_end + dt, dt)
    ys = np.zeros([ts.shape[0], ensemble_size])
    print(f'ts.shape = {ts.shape}, ys.shape = {ys.shape}')
    ys[0, :] = z_0
    print('Integrating SDE using Milstein method:')
    bstart = perf_counter()
    for j in range(ensemble_size):
        print(f'    integrating ensemble member {j+1} of {ensemble_size}... ', end='')
        astart = perf_counter()
        for i in range(1, ts.size):
            t = ts[i]
            y = ys[i - 1, j]
            if y < radius[0] or y > radius[1]:
                break
            b_prime = (b(t, y+epsilon) - b(t, y-epsilon))\
                      / (2*epsilon)
            dWt = dW(dt)
            ys[i, j] = y + a(t, y) * dt + b(t, y) * dWt \
                       + 0.5 * b(t, y) * b_prime * (dWt ** 2 - dt)
        aend = perf_counter()
        print(f'({aend-astart}s)')
    bend = perf_counter()
    print(f'    ...integration complete. Total time: {bend-bstart}seconds.')
    return ts, ys


def euler_test():
    a = lambda t, z: - 4 * z**3 + 2 * t * z
    sigma = 0.3
    #
    # The potential function looks like
    #     ∫ -(-4z^3 + 2tz) dz = z^4 - tz^2
    # With minima (stable points of the system)
    # where -4z^3 + 2tz = 0, i.e.
    #                 z = ±√(t/2)
    # and a maximum (unstable node) at z = 0
    #
    ts, ys = eulermaruyama(a, sigma, z_0=0, n=10**7, t_0=-4, t_end=4,
                           ensemble_size=1, radius=(-2.2, 2.2))
    plt.plot(ts, ys)
    plt.plot(ts, np.sqrt(0.5 * ts))
    plt.plot(ts, -np.sqrt(0.5 * ts))
    plt.show()


def milstein_test():
    # a = lambda t, z: - (4 * z ** 3) + (2 * t * z)
    a = lambda t, z: - z**4 + (t * z**2)
    b = lambda t, z: 0.1
    ts, ys = milstein_method(a, b, z_0=0, n=10**5, t_0=-5, t_end=5, ensemble_size=6, radius=(-2, 2))
    plt.plot(ts, ys)
    plt.plot(ts, np.sqrt(0.5 * ts))
    plt.plot(ts, -np.sqrt(0.5 * ts))
    plt.plot(ts, np.sqrt(ts))
    plt.plot(ts, -np.sqrt(ts))
    plt.show()


def milstein_example(num_sims=3):
    # Milstein Method
    # One Second and thousand grid points
    t_init = 0
    t_end = 10
    N = 10 ** 5 # Compute 1000 grid points
    dt = float(t_end - t_init) / N
    epsilon = 10 ** (-5)

    # Initial Conditions
    y_init = 0.7
    mu = 3
    sigma = 1

    # vectors to fill
    ts = np.arange(t_init, t_end + dt, dt)
    ys = np.zeros(N + 1)
    ys[0] = y_init

    # a = lambda t, z: mu * z
    # b = lambda t, z: sigma * z * t

    a = lambda t, z: - (4 * z ** 3) + (2 * 1 * z)
    b = lambda t, z: 0.1

    # Loop
    for _ in range(num_sims):
        for i in range(1, ts.size):
            t = (i - 1) * dt
            y = ys[i - 1]
            b_prime = (b(t, y-epsilon) + b(t, y+epsilon)) / 2
            dWt = dW(dt)
            # Milstein method
            # ys[i] = y + mu * dt * y + sigma * y * dW(dt) + 0.5* sigma**2 * (dW(dt)**2 - dt)
            ys[i] = y + a(t, y) * dt + b(t, y) * dWt + 0.5 * b(t, y) * b_prime * (dWt ** 2 - dt)
        plt.plot(ts, ys)

    # Plot
    plt.xlabel("time (t)")
    plt.grid()
    h = plt.ylabel("y")
    h.set_rotation(0)
    plt.show()
