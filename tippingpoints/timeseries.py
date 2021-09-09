import matplotlib.pyplot as plt
import numpy as np
from tippingpoints import scaling_methods


class TimeSeries:
    def __init__(self, z: np.ndarray, t: np.ndarray = None):
        self._z = z
        if t is None:
            self._t = np.linspace(0, 1, self._z.shape[0])
        else:
            self._t = t
        self._acf_indicator = np.zeros_like(self._z)
        self._dfa_indicator = np.zeros_like(self._z)
        self._pse_indicator = np.zeros_like(self._z)

        self._acf_indicator_t = np.zeros_like(self._t)
        self._dfa_indicator_t = np.zeros_like(self._t)
        self._pse_indicator_t = np.zeros_like(self._t)

        self._acf_indicator_kwargs = {'window_size': 200,
                                      'lag': 1,
                                      'increment': 1}
        self._dfa_indicator_kwargs = {'window_size': 200,
                                      'order': 2,
                                      'no_segment_lengths': 8,
                                      'increment': None}
        self._pse_indicator_kwargs = {'window_size': 200,
                                      'binning': True,
                                      'window_limits': (-2, -1),
                                      'increment': 1}

    def set_acf_indicator(self, alternative_dict=None):
        if alternative_dict is not None:
            self.set_indicator_kwargs('acf', alternative_dict)
        self._acf_indicator_t, self._acf_indicator =\
            scaling_methods.acf_sliding(self._t, self._z, **self._acf_indicator_kwargs)

    def set_dfa_indicator(self, alternative_dict=None):
        if alternative_dict is not None:
            self.set_indicator_kwargs('dfa', alternative_dict)
        self._dfa_indicator_t, self._dfa_indicator =\
            scaling_methods.dfa_sliding(self._t, self._z, **self._dfa_indicator_kwargs)

    def set_pse_indicator(self, alternative_dict=None):
        if alternative_dict is not None:
            self.set_indicator_kwargs('pse', alternative_dict)
        self._pse_indicator_t, self._pse_indicator =\
            scaling_methods.pse_sliding(self._t, self._z, **self._pse_indicator_kwargs)

    @property
    def acf_indicator(self):
        return self._acf_indicator

    @property
    def dfa_indicator(self):
        return self._dfa_indicator

    @property
    def pse_indicator(self):
        return self._pse_indicator

    @acf_indicator.setter
    def acf_indicator(self, alternative_dict=None):
        self.set_acf_indicator(alternative_dict=alternative_dict)

    @dfa_indicator.setter
    def dfa_indicator(self, alternative_dict=None):
        self.set_dfa_indicator(alternative_dict=alternative_dict)

    @pse_indicator.setter
    def pse_indicator(self, alternative_dict=None):
        self.set_pse_indicator(alternative_dict=alternative_dict)

    def set_indicator_kwargs(self, indicator_name, new_dict):
        if indicator_name == 'acf':
            for k, v in new_dict.items():
                if k in self._acf_indicator_kwargs.keys():
                    self._acf_indicator_kwargs[k] = v
                else:
                    raise RuntimeError(f'Could not assign value {v} to non-existent key {k}')
        elif indicator_name == 'dfa':
            for k, v in new_dict.items():
                if k in self._dfa_indicator_kwargs.keys():
                    self._dfa_indicator_kwargs[k] = v
                else:
                    raise RuntimeError(f'Could not assign value {v} to non-existent key {k}')
        elif indicator_name == 'pse':
            for k, v in new_dict.items():
                if k in self._pse_indicator_kwargs.keys():
                    self._pse_indicator_kwargs[k] = v
                else:
                    raise RuntimeError(f'Could not assign value {v} to non-existent key {k}')
        else:
            raise ValueError(f'There is no indicator named {indicator_name}')

    def print_indicator_kwargs(self, indicator_name):
        if indicator_name == 'acf':
            indicator_dict = self._acf_indicator_kwargs
        elif indicator_name == 'dfa':
            indicator_dict = self._dfa_indicator_kwargs
        elif indicator_name == 'pse':
            indicator_dict = self._pse_indicator_kwargs
        else:
            raise ValueError(f'There is no indicator named {indicator_name}')
        print(f'kwargs for {indicator_name} indicator:')
        for k, v in indicator_dict.items():
            print(f'   {k}: {v}')

    def plot_indicator(self, indicator_name):
        if indicator_name == 'acf':
            indicator = self._acf_indicator
            t = self._acf_indicator_t
        elif indicator_name == 'dfa':
            indicator = self._dfa_indicator
            t = self._dfa_indicator_t
        elif indicator_name == 'pse':
            indicator = self._pse_indicator
            t = self._pse_indicator_t
        else:
            raise ValueError(f'There is no indicator named {indicator_name}')
        fig, ax = plt.subplots()
        ax.plot(t, indicator, )
        plt.show()


if __name__ == '__main__':
    print('Hello, World!')
