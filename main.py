import numpy as np
# from timeseries_module import TimeSeries
from thesisfigures import onedimensionalmethods


# import pandas as pd
# import scipy as sp


class CoolClass:
    def __init__(self, z: np.ndarray, t: np.ndarray = None):
        self.z = z
        if t is None:
            self.t = np.linspace(0, 1, self.z.shape[0])
        else:
            self.t = t

    def show(self):
        print(f'z = {self.z}, t = {self.t}')


if __name__ == '__main__':

    onedimensionalmethods.fig07(no_tests=10)

    # c = CoolClass(np.array([5, 6, 7]), np.zeros(4))
    # c.show()
    #
    # d1 = {'name': 'Pingu', 'age': 4, 'height': 40, 'height_units': 'inches', 'array': np.arange(0, 3, 0.5)}
    # new_dict = {'age': 5, 'height': 42}
    #
    # print(d1)
    #
    # for k, v in new_dict.items():
    #     if k in d1.keys():
    #         d1[k] = v
    #
    # print(d1)

    # white_noise = np.random.randn(10**3)
    # ts2 = ts.TimeSeries(white_noise)
    # ts2.set_acf_indicator()
    #
    # print(ts2.acf_indicator.shape)
    # ts2.print_indicator_kwargs('acf')

    # milstein_example()
    # euler_test()
    #
    #
    # xdata = np.array([-1, 0, 1, 2, 3])
    # ydata = np.array([1, 0, 1, 4, 9])
    #
    # p = Polynomial.fit(xdata, ydata, deg=2)
    # q = np.polynomial.polynomial.polyfit(xdata, ydata, deg=2)
    #
    # print(p)
    # print(q)

    # scaling_methods.acf_sliding_test()
    # scaling_methods.acf_test()
    # scaling_methods.dfa_test()
    # scaling_methods.pse_test()
    # scaling_methods.colour_noise_test()


