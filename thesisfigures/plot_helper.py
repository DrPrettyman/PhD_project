import matplotlib.pyplot as plt
import numpy as np
import string


def standard_line_dict(transparent=True):
    """
    returns a standard dictionary of axes.plot kwargs so that style is
    consistent throughout all plots
    """
    dict00 = {'line1': {'color': 'k', 'lw': 1.2, 'ls': '-'},
              'line2': {'color': (0.3, 0.3, 0.3), 'lw': 1.2, 'ls': '-'},
              'line3': {'color': (0.6, 0.6, 0.6), 'lw': 1.2, 'ls': '-'},
              'k': {'color': 'k', 'lw': 1.2, 'ls': '-'},
              'r': {'color': 'r', 'lw': 1.2, 'ls': '-'},
              'g': {'color': 'g', 'lw': 1.2, 'ls': '-'},
              'b': {'color': 'b', 'lw': 1.2, 'ls': '-'},
              'm': {'color': 'm', 'lw': 1.2, 'ls': '-'},
              'thick': {'color': 'k', 'lw': 2, 'ls': '-'},
              'thin': {'color': (0.3, 0.3, 0.3), 'lw': 0.7, 'ls': '-', 'alpha': 0.7},
              'thin_red': {'color': 'r', 'lw': 0.7, 'ls': '-', 'alpha': 0.7},
              'thin_blue': {'color': 'b', 'lw': 0.7, 'ls': '-', 'alpha': 0.7},
              'thin_green': {'color': 'g', 'lw': 0.7, 'ls': '-', 'alpha': 0.7},
              'scatter': {'color': (0.1, 0.1, 0.1), 'ls': '', 'marker': 'o',
                          'markersize': 5},
              'dash': {'color': (0.1, 0.1, 0.1), 'lw': 2, 'ls': '--'},
              'dash_red': {'color': 'r', 'lw': 2, 'ls': '--'},
              'black_o': {'color': (0.1, 0.1, 0.1), 'lw': 1.2, 'ls': '--',
                          'marker': 'o', 'markersize': 3},
              'red_^': {'color': 'r', 'lw': 1.2, 'ls': '--',
                        'marker': '^', 'markersize': 3},
              'v_dash': {'color': (0.3, 0.3, 0.3), 'lw': 1.5, 'ls': '--'},
              'v_dash_marker': {'color': (0.3, 0.3, 0.3), 'lw': 1.5, 'ls': '--',
                                'marker': 'o', 'markersize': 3}
              }
    if not transparent:
        for style in dict00.keys():
            dict00[style].update({'alpha': 1})
    return dict00


# def figure_properties(pattern: list[list[int]], aspect: float = None, vertical_gap_size: str = 'large'):
def figure_properties(pattern, aspect: float = None, vertical_gap_size: str = 'large'):
    """
    Takes a subplot pattern and returns the positions of the subplots with
    standardised padding around the subplots.

    :param pattern: The pattern parameter is in top-to-bottom order.
                    The pattern [[0, 2], [0]] defines two subplots on the top
                    row, with vsmall spacing between, and a single plot on the
                    bottom row.
    :param aspect:
    :param vertical_gap_size:
    :return:
    """
    fig_width = 10.   # Standard figure width 10 inches
    gap_large = 0.7   # Gap including axes labels, i.e. bottom and left of axes
    gap_small = 0.4   # Gap including tick labels but no axes labels
    gap_vsmall = 0.2  # Very narrow gap i.e. top and right of axes

    # It is convenient for our methods to store these values in a list
    gap_sizes = [gap_large, gap_small, gap_vsmall, 0.]

    # The vertical gaps between axes on different rows can be
    # adjusted with kwarg 'vertical_gap_size
    if vertical_gap_size == 'none':
        v_gap = 0.
    elif vertical_gap_size == 'vsmall':
        v_gap = gap_vsmall
    elif vertical_gap_size == 'small':
        v_gap = gap_small
    else:
        v_gap = gap_large

    no_rows = len(pattern)
    v_gaps_inches = [gap_vsmall] + [v_gap for _ in range(no_rows - 1)] + [gap_large]

    if aspect is None:
        aspect = [1. for _row in range(no_rows)]
        for _row in range(no_rows):
            no_cols = len(pattern[_row])
            if no_cols == 1:
                aspect[_row] = 0.5

    axes_layout = [[] for _ in range(len(pattern))]
    no_plots = 0
    for _row in range(len(pattern)):
        for _col in range(len(pattern[_row])):
            axes_layout[_row].append(no_plots)
            no_plots += 1

    h_gaps = [[] for _row in range(no_rows)]
    axes_width = [0. for _row in range(no_rows)]
    axes_height = [0. for _row in range(no_rows)]
    axes_width_inches = [0. for _row in range(no_rows)]
    axes_height_inches = [0. for _row in range(no_rows)]
    no_cols = [1 for _row in range(no_rows)]
    for _row in range(no_rows):
        no_cols[_row] = len(pattern[_row])

        h_gaps_inches = [gap_sizes[pattern[_row][_col]] for _col in range(no_cols[_row])]
        h_gaps_inches[0] = gap_large  # always have a large gap at the left
        h_gaps_inches.append(gap_vsmall)  # and a vsmall gap at the right
        h_gaps[_row] = [x/fig_width for x in h_gaps_inches]

        axes_width_inches[_row] = (fig_width - sum(h_gaps_inches)) / no_cols[_row]
        axes_width[_row] = axes_width_inches[_row] / fig_width
        axes_height_inches[_row] = axes_width_inches[_row] * aspect[_row]
        axes_height[_row] = axes_width[_row] * aspect[_row]

    fig_height = sum(axes_height_inches) + sum(v_gaps_inches)
    axes_height = [x/fig_height for x in axes_height_inches]

    v_gaps = [x/fig_height for x in v_gaps_inches]

    fig_height_relative = sum(axes_height) + sum(v_gaps)
    if abs(fig_height_relative - 1.) > 0.01:
        print(f'relative figure height = {fig_height_relative}')

    axes_positions = []
    for _row in range(no_rows):
        bottom = 1 - (sum(v_gaps[:(_row+1)]) + sum(axes_height[:(_row+1)]))
        for _col in range(no_cols[_row]):
            left = sum(h_gaps[_row][:(_col+1)]) + _col * axes_width[_row]
            axes_positions.append([left, bottom, axes_width[_row], axes_height[_row]])

    return (fig_width, fig_height), no_plots, axes_positions, axes_layout


class ThesisPlot:
    """
    Class initialises a figure with subplots side-by-side and includes
    helper functions for plotting.
    The original pyplot functionality can be accessed directly through
    the object variables 'fig', 'ax', as in
    p = Plot1by2()
    p.ax[0].set_xlim(0, 10)
    """
#     def __init__(self, pattern: list[list[int]], figure_number: str = '', **pattern_kwargs):
    def __init__(self, pattern, figure_number: str = '', **pattern_kwargs):
        plt.style.use('./thesisfigures/mplstyles/thesis.mplstyle')
        self.figure_number = figure_number
        self.standard_line_dict = standard_line_dict(transparent=True)
        self.pattern = pattern
        self.tick_params = {'top': True, 'bottom': True,
                            'left': True, 'right': True,
                            'direction': 'in'}

        # create the figure
        self.fig = plt.figure()
        self.fig_size, self.no_plots, self.axes_positions, self.axes_layout = \
            figure_properties(self.pattern, **pattern_kwargs)
        self.fig.set_size_inches(self.fig_size)
        self.fig.canvas.manager.set_window_title(f'Thesis_JPrettyman figure {self.figure_number}')

        # add subplots
        self.ax = []
        for _i in range(self.no_plots):
            self.ax.append(self.fig.add_axes(self.axes_positions[_i]))

        # set the tick_params for the axes and annotate with letters
        for _i in range(self.no_plots):
            self.ax[_i].tick_params('both', **self.tick_params)
            if self.no_plots > 1:
                self.ax[_i].annotate(text=string.ascii_lowercase[_i], xy=(0, 0), xytext=(0.05, 0.93),
                                     xycoords='axes fraction', fontsize=20)

        print('Drawing figure with {0} subplots with layout {1}...'
              .format(self.no_plots, self.axes_layout))

    def set_transparent_false(self):
        # transparency must be set to False by calling this method before plotting if the figure is
        # intended to be saved in a format that does not support transparency (e.g. eps)
        self.standard_line_dict = standard_line_dict(transparent=False)
        print('Transparency set to False...')

    def plot(self, x: np.ndarray, y: np.ndarray, axes_no: int = 0,
             line_style='line1', label: str = None):
        self.ax[axes_no].plot(x, y, label=label, **self.standard_line_dict[line_style])

    def v_dash(self, axes_no: int = 0, x_coord: float = 0., markers: bool = False):
        _ylims = self.ax[axes_no].get_ylim()
        if markers:
            _line_style = 'v_dash_markers'
        else:
            _line_style = 'v_dash'
        self.ax[axes_no].plot([x_coord, x_coord], _ylims, **self.standard_line_dict[_line_style])
        self.ax[axes_no].set_ylim(_ylims)

    def axes_labels(self, axes_no: int = 0, xlabel: str = '', ylabel: str = ''):
        self.ax[axes_no].set(xlabel=xlabel, ylabel=ylabel)

    def legend(self, axes_no: int = 0):
        self.ax[axes_no].legend()

    def share_xlims(self, axes: list):
        _xlims = [self.ax[_i].get_xlim() for _i in axes]
        _xlim_new = (min([_xlims[_i][0] for _i in range(len(_xlims))]),
                     max([_xlims[_i][1] for _i in range(len(_xlims))]))
        for _i in axes:
            self.ax[_i].set_xlim(_xlim_new)

    def share_ylims(self, axes: list = None):
        if axes is None:
            axes = list(range(self.no_plots))
        _ylims = [self.ax[_i].get_ylim() for _i in axes]
        _ylim_new = (min([_ylims[_i][0] for _i in range(len(_ylims))]),
                     max([_ylims[_i][1] for _i in range(len(_ylims))]))
        for _i in axes:
            self.ax[_i].set_ylim(_ylim_new)

    def auto_share_ylims(self, row_no: int):
        _l = [[]]
        _k = 0
        for _i in range(len(self.pattern[row_no])):
            if self.pattern[row_no][_i] == 0:
                _l.append([])
                _k += 1
            _l[_k].append(self.axes_layout[row_no][_i])
        if self.pattern[row_no][0] == 0:
            _l.pop(0)
        for _j in range(len(_l)):
            if len(_l[_j]) > 0:
                print(f'   ::: {_l[_j]}')
                self.share_ylims(_l[_j])

    def print_prep(self):
        """
        Prepares the figure for display or save by cleaning up
        the axes labels and tick labels
        """
        for _row in range(len(self.pattern)):
            for _col in range(len(self.pattern[_row])):
                axes_no = self.axes_layout[_row][_col]
                if _col > 0:
                    if self.pattern[_row][_col] > 0:
                        self.ax[axes_no].set(ylabel='')
                    if self.pattern[_row][_col] > 1:
                        ### Note!!: also do not display first xtick label
                        self.ax[axes_no].tick_params('y', labelleft=False)
            if len(self.pattern[_row]) > 1:
                print(f'Auto-sharing ylims for row {_row}:')
                self.auto_share_ylims(row_no=_row)

    def show(self):
        print('Showing figure...')
        self.print_prep()
        plt.show()

    def save_figure(self, fig_format='png'):
        fig_no = self.figure_number.split('.')[-1]
        while len(fig_no) < 2:
            fig_no = '0'+fig_no
        filename = '../../Desktop/fig'+fig_no+'.'+fig_format
        print('Saving figure as {0}'.format(filename)+'...')
        dpi_parameter = 600 if fig_format == 'png' else None
        plt.savefig(filename, dpi=dpi_parameter, format=fig_format)

    def print_figure_properties(self):
        print('Figure properties:')
        print(f' - figure size (w,h) : {self.fig_size}')
        print(f' - number of plots   : {self.no_plots}')
        print(f' - subplot pattern   : {self.pattern}')
        print(f' - subplot layout    : {self.axes_layout}')
        print(' - plot positions:')
        for _i in range(self.no_plots):
            print(f'     {_i}:  {self.axes_positions[_i]}')


def thesisplot_test():
    x1 = np.linspace(0, 1, 100)
    y1 = x1 ** 2
    y2 = 1.5 * np.sin(3*x1)
    x3 = np.linspace(-2, 2, 1000)
    y3 = x3 ** 3
    y4 = 2 * x3 ** 3
    y5 = 0.5 * x3 ** 3
    y6 = 0.9 * x3 + 2

    p = ThesisPlot(pattern=[[0, 2, 0, 1], [0, 2, 2], [0]])

    p.plot(x1, y1, axes_no=0)
    p.plot(x1, y2, axes_no=1)
    p.plot(x3, y3, axes_no=2)
    p.plot(x3, y4, axes_no=3)
    p.plot(x3, y5, axes_no=4)
    p.plot(x3, y6, axes_no=5)

    p.print_figure_properties()

    p.show()


if __name__ == '__main__':
    pass
