import matplotlib.pyplot as plt
import numpy as np


def standard_line_dict():
    """
    returns a standard dictionary of axes.plot kwargs so that style is
    consistent throughout all plots
    """
    dict00 = {'line1': {'color': 'k', 'lw': 1.2, 'ls': '-'},
              'line2': {'color': (0.4, 0.4, 0.4), 'lw': 1.2, 'ls': '-'},
              'line3': {'color': (0.7, 0.7, 0.7), 'lw': 1.2, 'ls': '-'},
              'thin': {'color': (0.2, 0.2, 0.2), 'lw': 0.7, 'ls': '-'},
              'thin_red': {'color': 'r', 'lw': 0.7, 'ls': '-'},
              'thin_blue': {'color': 'b', 'lw': 0.7, 'ls': '-'},
              'scatter': {'color': (0.1, 0.1, 0.1), 'lw': 1, 'ls': '', 'marker': 'o',
                          'markersize': 5},
              'dash': {'color': (0.1, 0.1, 0.1), 'lw': 2, 'ls': '--'},
              'dash_red': {'color': 'r', 'lw': 2, 'ls': '--'}}
    return dict00


def standard_tick_params():
    """
    returns a standard dictionary of axes.tick_params kwargs so that style is
    consistent throughout all plots
    """
    dict01 = {'axes': 'both',
              'top': True, 'bottom': True, 'left': True, 'right': True,
              'direction': 'in'}
    return dict01


class Plot1by2:
    """
    Class initialises a figure with subplots side-by-side and includes
    helper functions for plotting.
    The original pyplot functionality can be accessed directly through
    the object variables 'fig', 'ax1' and 'ax2', as in
    p = Plot1by2()
    p.ax1.set_xlim(0, 10)
    """
    def __init__(self, figure_number: str = '0'):
        plt.style.use('./thesisfigures/mplstyles/thesis.mplstyle')
        self.standard_line_dict = standard_line_dict()

        # create tha figure
        self.fig = plt.figure()
        self.fig.set_size_inches((12, 6))
        self.fig.canvas.manager.set_window_title(f'Thesis_JPrettyman figure {figure_number}')

        # add subplots
        self.ax1 = self.fig.add_subplot(1, 2, 1)
        self.ax2 = self.fig.add_subplot(1, 2, 2)

        # set the tick_params for the axes
        self.ax1.tick_params(standard_tick_params())
        self.ax2.tick_params(standard_tick_params())

        # add labels 'a' and 'b'
        self.ax1.annotate(text='a', xy=(0, 0), xytext=(0.05, 0.93), xycoords='axes fraction', fontsize=20)
        self.ax2.annotate(text='b', xy=(0, 0), xytext=(0.05, 0.93), xycoords='axes fraction', fontsize=20)

    def plot_a(self, x: np.ndarray, y: np.ndarray, ls='line1', label: str = None):
        self.ax1.plot(x, y, **self.standard_line_dict[ls], label=label)

    def plot_b(self, x: np.ndarray, y: np.ndarray, ls='line1', label: str = None):
        self.ax2.plot(x, y, **self.standard_line_dict[ls], label=label)

    def axes_labels(self, x1: str = '', y1: str = '', x2: str = '', y2: str = ''):
        self.ax1.set(xlabel=x1, ylabel=y1)
        self.ax2.set(xlabel=x2, ylabel=y2)

    def sharey(self):
        self.ax2.tick_params('y', labelleft=False)
        self.ax2.set(ylabel='')
        ylim1 = self.ax1.get_ylim()
        ylim2 = self.ax2.get_ylim()
        ylim_new = (min(ylim1[0], ylim2[0]), max(ylim1[1], ylim2[1]))
        self.ax1.set_ylim(ylim_new)
        self.ax2.set_ylim(ylim_new)

    def show(self):
        plt.show()
