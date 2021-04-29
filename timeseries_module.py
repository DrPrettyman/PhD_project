import numpy as np
from tippingpoints import scaling_methods


class TimeSeries(np.ndarray):
    def __new__(cls, input_array, info=None):
        obj = np.asarray(input_array).view(cls)
        return obj

    def __init__(self):
        self.acf1_indicator_dict = {'window_size': 200,
                                    'lag': 1,
                                    'increment': 1}
        self.dfa_indicator_dict = {'window_size': 200,
                                   'order': 2,
                                   'no_segment_lengths': 8,
                                   'increment': 1}
        self.pse_indicator_dict = {'window_size': 200,
                                   'binning': True,
                                   'window_limits': (-2, -1),
                                   'increment': 1}

    @property
    def acf1_indicator(self):
        return self._acf1_indicator

    @property
    def pse_indicator(self):
        return self._pse_indicator

    @acf1_indicator.setter
    def acf1_indicator(self):
        self._acf1_indicator = scaling_methods.acf_sliding(self._t, self, **self.acf1_indicator_dict)

    @pse_indicator.setter
    def pse_indicator(self):
        self._pse_indicator = scaling_methods.pse_sliding(self._t, self, **self.pse_indicator_dict)


if __name__ == '__main__':
    print('Hello, World!')


def noise_methods():
    return None