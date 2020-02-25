"""Module for plotting Nd fields."""

import abc
import math

from ipywidgets import widgets
from IPython.display import display
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import seaborn as sns
import threading

from .setup_plot import (BuildWidget, _check, _read_data)

CBAR_ARGS = {
            'cbar_mode': 'single',
            'cbar_size': '2%',
            'cbar_pad': '2%',
            'cbar_location': 'bottom',
            }

__all__ = ('ProfilePlotter',)


COL_PAL = sns.color_palette("colorblind", 8)


class ProfilePlotter(BuildWidget):

    """Plot factory to create different kinds of Profile Plots."""

    @abc.abstractmethod
    def __init__(self,
                 datasets,
                 varnames,
                 step_dim,
                 sel_slice,
                 avg_dims, *,
                 apply_func=None,
                 figsize=None,
                 vmax=None,
                 vmin=None,
                 stepsize=None,
                 invert_yaxis=False,
                 step_variable='time',
                 cbar_args=None,
                 dims=1,
                 maps=False,
                 **kwargs):
        """Create the Profile Plot object."""
        self.dims = dims
        self.maps = maps
        _link = False
        if not isinstance(datasets, (tuple, list, set)):
            self.datasets = [datasets]
        else:
            self.datasets = list(datasets)
        self.varnames = _check(self.datasets, varnames, warn='variable names')
        self.step_dims = _check(self.datasets, step_dim, accpet_none=False)
        self.apply_func = _check(self.datasets, apply_func, accpet_none=False)
        self.avg_dims = _check(self.datasets, avg_dims)
        self.sel_slice = _check(self.datasets, sel_slice)
        vmin = _check(self.datasets, vmin, accpet_none=False)
        vmax = _check(self.datasets, vmax, accpet_none=False)
        self.dims = dims
        self.cbar_args = {key: value for (key, value) in CBAR_ARGS.items()}
        try:
            for key, value in cbar_args.items():
                self.cbar_args[key] = value
        except AttributeError:
            pass
        self.step_variables = []
        if self.cbar_args['cbar_mode'].lower() != 'each':
            _link = True
        for (dset, varn, step_d) in zip(self.datasets,
                                          self.varnames,
                                          self.step_dims):
            if isinstance(step_d, str):
                self.step_variables.append(step_d)
            elif isinstance(step_d, (int, float)):
                self.step_variables.append(dset[varn].dims[step_d])
            else:
                self.step_variables.append(None)

        self.plots = [None for i in range(len(self.varnames))]

        super().__init__(figsize=figsize,
                         vmax=vmax,
                         vmin=vmin,
                         stepsize=stepsize,
                         clim_setter=self._set_clim,
                         plot_updater=self._update_plot,
                         cmap_setter=self._set_cmap,
                         step_variable=step_variable,
                         invert_yaxis=invert_yaxis,
                         link=_link,
                         num_dsets=len(datasets))

        self.kwargs = kwargs
        self.figsize = figsize
        self.cmap = getattr(cm, self.cmaps[0])
        self._set_widgets()
        self._thread_stop = None
        self._thread = None
        self.adjust_step(self.datasets, self.step_variables)
        self.setup()

    def _update_plot(self, plot_range=None):
        if self.dims == 1:
            self._update_1dplot(plot_range=plot_range)
        else:
            self._update_2dplot(plot_range=plot_range)

    def setup(self):
        """Setup the Profile Plot."""
        if self.dims == 1:
            func = self.setup_1d
        else:
            func = self.setup_2d
        self._thread_stop = threading.Event()
        self._thread = threading.Thread(target=func)
        self._thread.start()

    def _stop(self):
        try:
            self._thread_stop.set()
        except AttributeError:
            pass

    def _set_clim(self, vmin, vmax, num):
        """Set new display limits of images."""
        cbar_ticks = np.linspace(vmin, vmax, 6)
        self.vmin[num], self.vmax[num] = vmin, vmax
        if self._link:
            plots, cbars = self.plots, self.cbars
        else:
            try:
                plots, cbars = [self.plots[num]], [self.cbars[num]]
            except IndexError:
                plots, cbars = [self.plots[num]], self.cbars

        for n, im in enumerate(plots):
            try:
                im.set_clim(vmin, vmax)
            except AttributeError:
                continue
            try:
                cbar = cbars[n]
            except IndexError:
                pass
            try:
                cbar.update_normal(im)
            except AttributeError:
                pass
            try:
                cbar.set_ticks(cbar_ticks)
                cbar.update_ticks()
            except AttributeError:
                pass

    def _set_cmap(self, cmap, num):
        """Set new colormap."""
        if self._link:
            plots = self.plots
        else:
            plots = [self.plots[num]]
        for im in plots:
            try:
                im.set_cmap(cmap)
            except AttributeError:
                return

    def get_data(self):
        """Get the plot data."""
        data = []
        for n, (dset, varn) in enumerate(zip(self.datasets, self.varnames)):
            try:
                data.append(self.apply_func[n](dset,
                                               varn,
                                               self.timestep,
                                               **self.kwargs))
            except TypeError:
                try:
                    sel_slice = self.sel_slice[n]
                except TypeError:
                    sel_slice = self.sel_slice
                try:
                    avg_dims = self.avg_dims[n]
                except TypeError:
                    avg_dims = self.avg_dims
                data.append(_read_data(dset,
                                       varn,
                                       self.step_variables[n],
                                       sel_slice,
                                       avg_dims,
                                       self.timestep,
                                       self.dims))
            if self.vmin[n] is None and self.vmax[n] is None:
                self.vmax[n] = np.nanmax(data[n])
                self.vmin[n] = np.nanmin(data[n])
                max_v = self.vmax[n]+np.fabs(self.vmax[n]-self.vmin[n])
                min_v = self.vmin[n]-np.fabs(self.vmax[n]-self.vmin[n])
                if min_v == max_v:  # Both are prob 0:
                    max_v += 0.03
                    min_v -= 0.03
                mag = 10**(math.floor(math.log10(np.fabs(max_v))-2))
                try:
                    self.val_sliders[n].step = mag
                    self.val_sliders[n].min = min_v
                    self.val_sliders[n].max = max_v
                    self.val_sliders[n].value = [self.vmin[n]/2.,
                                                 self.vmax[n]/2.]
                except IndexError:
                    pass

        return data

    def _update_1dplot(self, **kwargs):
        """Update the plotted image."""
        data = self.get_data()
        if self.fig is None:
            # No plot hasn't been created yet
            self.fig = plt.figure(figsize=self.figsize)
            self.fig.subplots_adjust(bottom=0.1,
                                     top=0.95,
                                     hspace=0.15,
                                     wspace=0.15,
                                     right=.9,
                                     left=0.05)

        for i, _ in enumerate(data):
            if self.data_dim == 'y' or self.data_dim == 'Y':
                y = data[i]
                x = self.get_secondary(self.second_vars[i], i)
            else:
                x = data[i]
                y = self.get_secondary(self.second_vars[i], i)
            if self.plots[i] is None:  # We do not have an image yet
                self.ax.append(self.fig.add_subplot(1, len(data), i+1))
                self.plots[i] = self.ax[i].plot(x, y,
                                                color=COL_PAL[i],
                                                lw=self.linewidth)[0]

                self.set_plot_range(i, x, y)
            else:  # We do have an image, that needs updating
                self.plots[i].set_data(x, y)
                self.set_plot_range(i, x, y)
            if self.invert_yaxis:
                self.ax[i].invert_yaxis()

    def set_plot_range(self, num, x, y):
        """Set the min/max display range."""
        p_rangex = np.fabs(x.max() - y.min())
        p_rangey = np.fabs(y.max() - y.min())
        self.ax[num].set_xlim(x.min() - p_rangex*0.05, x.max() + p_rangex*0.05)
        self.ax[num].set_ylim(y.min() - p_rangey*0.05, y.max() + p_rangey*0.05)

    def setup_1d(self):
        """Setup the widgets."""
        self._update_plot(plot_range=(self.vmin, self.vmax))
        self.sl = widgets.HBox([self.t_step])
        for wdg in (self.sl, ):
            display(wdg)

    def get_secondary(self, var, num):
        """Get the secondary axis."""
        try:
            return self.apply_second(self.datasets[num])
        except TypeError:
            return np.arange(self.datasets[num].dims[var])

    def setup_2d(self):
        """Setup the widgets."""
        self._update_plot()
        _ = self.get_data()

        self.sl = widgets.HBox([self.t_step, *self.cmap_sel])
        for wdg in [self.sl] + self.val_sliders:
            display(wdg)

    def _update_2dplot(self, **kwargs):
        """Update the plotted image."""
        data = self.get_data()
        if self.fig is None:
            # No plot hasn't been created yet
            self.fig = plt.figure(figsize=self.figsize)
            self.fig.subplots_adjust(bottom=0.1,
                                     top=0.95,
                                     hspace=0.15,
                                     wspace=0.15,
                                     right=.9,
                                     left=0.05)
            self.ax = ImageGrid(self.fig, 111,
                                nrows_ncols=(1, len(data)),
                                axes_pad=0.01,
                                direction='row',
                                **self.cbar_args)
            for nn, dset in enumerate(self.datasets):
                try:
                    name = dset[self.varnames[nn]].long_name
                except AttributeError:
                    name = dset[self.varnames[nn]].name
                units = dset[self.varnames[nn]].units
                self.ax[nn].set_title('{} [{}]'.format(name, units))

        for i, _  in enumerate(data):
            if self.plots[i] is None:  # We do not have an image yet
                self.plots[i] = self.ax[i].imshow(data[i],
                                                  vmin=self.vmin[i],
                                                  vmax=self.vmax[i],
                                                  cmap=self.cmap,
                                                  origin='lower')

                if self.cbar_args['cbar_mode'] == 'each' or i == 0:
                    cbar = self.ax.cbar_axes[i].colorbar(self.plots[i])
                else:
                    cbar = None
                self.cbars.append(cbar)
                self.ax[i].axes.grid(b=None)
                self._set_clim(self.vmin[i], self.vmax[i], i)
            else:  # We do have an image, that needs updating
                self.plots[i].set_clim(self.vmin[i], self.vmax[i])
                self.plots[i].set_array(data[i])
        for ax in self.ax:
            if self.invert_yaxis:
                ax.axis.axes.invert_yaxis()
            ax.axis.axes.set_xticks([])
        self.invert_yaxis = False

    @classmethod
    def profile_2d(cls, datasets, varnames, figsize=None, apply_func=None,
                   vmax=None, vmin=None, stepsize=None, invert_yaxis=True,
                   sel_slice=None, cbar_args=None, step_dim=0, avg_dims=(1,),
                   **kwargs):
        """
        Create a 2D Profile (cross section) plot.

        This method takes datasets and creates cross sections by slicing
        or averaging or applying a user given function to input dataset(s).

        Parameters
        ----------

        datasets : collection, xarray dataset
            Input dataset(s)
        varnames : collection, str
            Variable name(s) that are to be displayed.
            Note: this variable has only an effect if no user defined function
            that is applyed to the input data is given.
        apply_func : mappable (default: None)
            User defined function that is applied to data, if None is given
            the build_in function to slice or average data based on
            variable names will be taken.
        sel_slice : collection of dicts (default : None)
            if sel_slice is not None than a slice along a given axis is taken
            from the data. the slice information for each dataset should be
            {dim_name/dim_num : slice/num}. Note: this has only effect if no
            user given apply_func is defined
        avg_dims : collection  (default : (1,))
            dimensions across which an average is applied this can be a tuple
            of strings or integers representing the dimension(s).
            Note: this variable has no effect if a user given apply_func is
            defined or the sel_slice variable is set.
        vmax : int, float (default : None)
            Max value for display range of the colorscale. If None given
            the range will be set by the data
        vmin : int, float
            Min value for display range of the colorscale. If None given
            the range will be set by the data
        stepsize : int, float (default : None)
            stepping size for the range slider that selects the display range
            (vmin, vmax). If None is given the steping size will be 1/100.
            of the magnitude of the input data.
        invert_yaxis : bool (default : True)
            If true the y-axis of the plot is inverted
        cbar_args : dict (default : None)
            Additional color bar arguments that are passed to the
            AxesGrid object of mpl_toolkits.axes_grid1.
        **kwargs:
            additional key word arguments that are passed to any user defined
            apply_func.

        """
        cls.setup = cls.setup_2d
        plot_obj = cls(datasets, varnames, step_dim,
                       sel_slice, avg_dims,
                       figsize=figsize,
                       apply_func=apply_func,
                       vmax=vmax, vmin=vmin,
                       stepsize=stepsize,
                       cbar_args=cbar_args,
                       dims=2,
                       maps=False,
                       invert_yaxis=invert_yaxis)
        return plot_obj

    @classmethod
    def profile_1d(cls, datasets, xvars, yvars,
                   figsize=None, apply_funcs=None, data_dim='y',
                   linewidth=2, stepsize=None, invert_yaxis=True,
                   sel_slice=None, cbar_args=None, step_dim=0, avg_dims=(1,),
                   **kwargs):
        """
        Create a 2D Profile (cross section) plot.

        This method takes datasets and creates cross sections by slicing
        or averaging or applying a user given function to input dataset(s).

        Parameters
        ----------

        datasets : collection, xarray dataset
            Input dataset(s)
        xvars : collection, str
            Variable name(s) that are to be displayed for X-values.
            Note: this variable has only an effect if no user defined function
            that is applyed to the input data is given.
        yvars : collection, str
            Variable name(s) that are to be displayed for Y-values.
            Note: this variable has only an effect if no user defined function
            that is applyed to the input data is given.
        linewidth : int (default : 2)
            linewidth of the plot
        data_dim : str (default : y)
            The dimension along the data is plotted (y or x)
        apply_funcs : collections of mappable (default: None)
            User defined functions that are applied to x and y data,
            if None are given the build_in function to slice or average data
            based on variable names will be taken.
        sel_slice : collection of dicts (default : None)
            if sel_slice is not None than a slice along a given axis is taken
            from the data. the slice information for each dataset should be
            {dim_name/dim_num : slice/num}. Note: this has only effect if no
            user given apply_func is defined
        avg_dims : collection  (default : (1,))
            dimensions across which an average is applied this can be a tuple
            of strings or integers representing the dimension(s).
            Note: this variable has no effect if a user given apply_func is
            defined or the sel_slice variable is set.
        stepsize : int, float (default : None)
            stepping size for the range slider that selects the display range
            (vmin, vmax). If None is given the steping size will be 1/100.
            of the magnitude of the input data.
        invert_yaxis : bool (default : True)
            If true the y-axis of the plot is inverted
        cbar_args : dict (default : None)
            Additional color bar arguments that are passed to the
            AxesGrid object of mpl_toolkits.axes_grid1.
        **kwargs:
            additional key word arguments that are passed to any user defined
            apply_func.

        """
        cls.xvars = _check(datasets, xvars)
        apply_funcs = _check(datasets, apply_funcs, accpet_none=False)
        if data_dim in ('Y', 'y'):
            apply_func = apply_funcs[1]
            cls.apply_second = apply_funcs[0]
            varnames = yvars
            second_vars = xvars
        else:
            apply_func = apply_funcs[0]
            cls.apply_second = apply_funcs[1]
            varnames = xvars
            second_vars = yvars
        cls.second_vars = _check(datasets, second_vars)
        cls.data_dim = data_dim
        cls.linewidth = linewidth
        plot_obj = cls(datasets, varnames, step_dim,
                       sel_slice, avg_dims,
                       figsize=figsize,
                       stepsize=stepsize,
                       apply_func=apply_func,
                       vmin=0, vmax=1000,
                       cbar_args=cbar_args,
                       invert_yaxis=invert_yaxis,
                       dims=1, maps=False)
        return plot_obj
