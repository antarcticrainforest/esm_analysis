"""Setup plot widgets in jupyter notebooks."""
import math

from ipywidgets import widgets, Layout
from IPython.display import display
from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib.transforms as mtransforms
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np


def _check(datasets, check_container, warn=None, accpet_none=True):
    if check_container is None and accpet_none:
        return check_container
    if not isinstance(check_container, (tuple, list, set)):
        check_container = [check_container, ]
    else:
        check_container = list(check_container)
    try:
        if len(datasets) > len(check_container):
            if warn:
                UserWarning('datasets have more entries than {}, '
                            'duplicating {}'.format(warn, warn))
            diff = len(datasets) - len(check_container)
            check_container += diff * [check_container[-1]]
        elif len(datasets) < len(check_container):
            if warn:
                UserWarning('more {} than datasets given, dropping ',
                            'exceeding {}'.format(warn, warn))
            check_container = check_container[:len(datasets)]
    except TypeError:
        pass
    return check_container


def _read_data(dataset, varname, step_var, sel_slice, avg_dim, tstep, dim):
    """
    The the data for a given step variable.

    Parameters
    ----------

    dataset : collection
    datasets that contain the data that needs to be displayed

    varname : collection
    variable names that are displayed

    step_variable : collection
    variable names of the stpping dimension (dimension that is changed)

    sel_slice : dict
    if not None a dict object defining the variable name and the index
    along which the dataset is sliced

    avg_dim : collection
    if not None a collection containing the variable names across which
    an average is taken

    tstep : int
    index of the stepping variable

    dim : dim
    the target dimension of the data set (2D or 1D)

    Returns
    -------

    xarray dataset of dimension dim: xarray.Dataset

    """
    if len(dataset[varname].shape) == dim:
        # There is not much more to do here:
        return dataset[varname].values
    if step_var is None:
        dset = dataset[varname]
    else:
        if isinstance(step_var, (int, float)):
            try:
                step_var = dataset.dims[int(step_var)]
            except IndexError:
                raise ValueError('Could not find step_variable in dataset')
        else:
            try:
                _ = dataset.variables[step_var]
            except KeyError:
                raise ValueError('Could not find step_variable in dataset')

        step_var_idx = dataset[varname].dims.index(step_var)
        dset = dataset[varname][{step_var: tstep}]
    if len(dset.shape) == dim:
        # There is not much more to do here:
        return dset.values

    if sel_slice is None:
        out_dims = []
        if isinstance(avg_dim, str):
            avg_dim = (avg_dim, )
        for d in avg_dim:
            if d in dset.dims and d not in out_dims:
                out_dims.append(d)
        # This indicates that we have to apply an avg along an axis
        return dset.mean(dim=tuple(out_dims)).values
    else:  # Try to select a slice
        out_slice = {}
        try:
            for key, value in sel_slice.items():
                out_slice[key] = value
        except AttributeError:
            raise ValueError('Slices should be defind with dictionaries,'
                             'of the form of dim:slice/num')
        return dset[out_slice].values


class BuildWidget:

    """Plot widget Factory."""

    cmaps = ['viridis', 'plasma', 'inferno', 'magma',
             'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
             'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
             'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn',
             'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
             'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic'
             'Pastel1', 'Pastel2', 'Paired', 'Accent',
             'Dark2', 'Set1', 'Set2', 'Set3',
             'tab10', 'tab20', 'tab20b', 'tab20c',
             'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
             'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg', 'hsv',
             'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar']

    def __init__(self, *,
                 figsize=None,
                 vmax=[None],
                 vmin=[None],
                 stepsize=None,
                 clim_setter=lambda vmin, vmax, num: None,
                 plot_updater=lambda: None,
                 cmap_setter=lambda num: None,
                 invert_yaxis=False,
                 step_variable='time',
                 link=False,
                 num_dsets=1,
                 **kwargs):
        """Create an instance of a plot widget."""
        self.clim_setter = clim_setter
        self.cmap_setter = cmap_setter
        self.plot_updater = plot_updater
        self.vmin = vmin
        self.vmax = vmax
        self.figsize = figsize or (8, 8)
        self.fig = None
        self.step_variable = step_variable
        self.cbars = []
        self.ax = []
        self.invert_yaxis = invert_yaxis
        self._link = link
        self._num_dsets = num_dsets

        self._tstep = 0
        self.mag = []
        for i in range(len(self.vmax)):
            try:
                max_v = self.vmax[i] + np.fabs(self.vmax[i] - self.vmin[i])
            except TypeError:
                max_v = 1000
            try:
                lg_abs = math.log10(np.fabs(max_v))-2
                self.mag.append(stepsize or 10**(math.floor(lg_abs)))
            except AttributeError:
                self.mag.append(1)

    def _set_widgets(self):
        min_v = []
        max_v = []
        if self._link:
            n_links = 1
        else:
            n_links = self._num_dsets
        for i in range(n_links):
            try:
                min_v.append((self.vmin[i]-np.fabs(self.vmax[i]-self.vmin[i]),
                             self.vmin[i]))
            except TypeError:
                min_v.append((0, -1))
            try:
                max_v.append((self.vmax[i]+np.fabs(self.vmax[i]-self.vmin[i]),
                             self.vmax[i]))
            except TypeError:
                max_v.append((1000, 11000))
        self.val_sliders = [widgets.FloatRangeSlider(
                                value=[min_v[i][-1], max_v[i][-1]],
                                min=min_v[i][0],
                                max=max_v[i][0],
                                step=self.mag[i],
                                description='Range:',
                                disabled=False,
                                continuous_update=False,
                                orientation='horizontal',
                                readout=True,
                                readout_format='0.4f',
                                layout=Layout(width='100%'))
                            for i in range(n_links)]
        self.cmap_sel = [widgets.Dropdown(
                                options=self.cmaps,
                                value=self.cmaps[0],
                                description='CMap:',
                                disabled=False,
                                layout=Layout(width='200px'))
                         for i in range(n_links)]
        self.t_step = widgets.BoundedFloatText(
                                 value=0,
                                 min=0,
                                 max=10000,
                                 step=1,
                                 disabled=False,
                                 description=self.step_variable,
                                 layout=Layout(width='200px',
                                               height='30px'))

        self.t_step.observe(self._tstep_observer, names='value')
        for n in range(n_links):
            self.val_sliders[n].observe(self._clim_observer, names='value')
            self.cmap_sel[n].observe(self._cmap_observer, names='value')
            self.val_sliders[n].num = n
            self.cmap_sel[n].num = n
            if n_links > 1:
                self.val_sliders[n].description = 'Range #{}:'.format(n+1)
                self.cmap_sel[n].description = 'CMap #{}:'.format(n+1)
            if self._link and n > 0:
                _ = widgets.jslink((self.val_sliders[0], 'value'),
                                   (self.val_sliders[n], 'value'))

    @property
    def timestep(self):
        """Get the time step."""
        return self._tstep

    def _tstep_observer(self, value):
        try:
            tstep = int(value['new'])
        except TypeError:
            return
        self._tstep = tstep
        self.plot_updater()

    def _clim_observer(self, plot_range):
        """Update the color limits."""
        num = plot_range['owner'].num
        try:
            vmin, vmax = plot_range['new'][0], plot_range['new'][-1]
        except KeyError:
            return
        self.clim_setter(vmin, vmax, num)

    def _cmap_observer(self, sel):
        num = sel['owner'].num
        """Update the colormap."""
        try:
            cmap_val = str(sel['new'])
        except KeyError:
            return

        try:
            cmap = getattr(cm, cmap_val)
        except AttributeError:
            cmap = getattr(cm2, cmap_val)
        except ValueError:
            return
            cmap = gettr(cm2, cmap_val)
        self.cmap_setter(cmap, num)

    def auto_adjust(self, lables):
        """Do automatic adjustment of the subplots."""
        for label in labels:
            bbox = label.get_window_extent()
            # the figure transform goes from relative coords->pixels and we
            # want the inverse of that
            bboxi = bbox.inverse_transformed(self.fig.transFigure)
            bboxes.append(bboxi)

        # this is the bbox that bounds all the bboxes, again in relative
        # figure coords
        bbox = mtransforms.Bbox.union(bboxes)
        if self.fig.subplotpars.left < bbox.width:
            # we need to move it over
            self.fig.subplots_adjust(left=1.1*bbox.width)  # pad a little

    @staticmethod
    def colorbar(mappable, vmin, vmax, draw=True):
        """Create the colorbar."""
        cbar_ticks = np.linspace(vmin, vmax, 6)
        ax = mappable.axes
        fig = ax.figure
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        colorbar = fig.colorbar(mappable, cax=cax)
        colorbar.update_normal(mappable)
        colorbar.set_ticks([])
        colorbar.update_ticks()
        return colorbar

    def adjust_step(self, datasets, step_variables):
        """Automatic adjusetment of the min/max steps."""
        vmax = []
        for (step_var, dset) in zip(step_variables, datasets):
            try:
                vmax.append(dset[step_var].shape[0] - 1)
            except KeyError:
                vmax.append(0)
        self.t_step.max = min(vmax)
        self.t_step.min = 0
