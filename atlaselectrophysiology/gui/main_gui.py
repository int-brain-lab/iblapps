import os
import platform
from typing import Union, Optional, List, Dict
from dataclasses import asdict
from collections import defaultdict

from iblutil.util import Bunch

if platform.system() == 'Darwin':
    if platform.release().split('.')[0] >= '20':
        os.environ["QT_MAC_WANTS_LAYER"] = "1"


from qtpy import QtWidgets
import pyqtgraph as pg
import pyqtgraph.exporters
import numpy as np

import atlaselectrophysiology.qt_utils.utils as utils
import atlaselectrophysiology.qt_utils.plots as qt_plots
import atlaselectrophysiology.qt_utils.ColorBar as qt_cbar
from atlaselectrophysiology.gui.layout import Setup

from atlaselectrophysiology.loaders.probe_loader import ProbeLoaderLocal, ProbeLoaderCSV, ProbeLoaderONE

from atlaselectrophysiology.plugins.cluster_popup import callback as cluster_callback
from atlaselectrophysiology.plugins.qc_dialog import display as display_qc
from pathlib import Path
from qt_helpers import qt
import matplotlib.pyplot as mpl  # noqa  # This is needed to make qt show properly :/

from functools import wraps
from typing import Any, Callable, Sequence
import time

# TODO
# Removed: top and tip line move and apply offset
# Removed: displaying nearby histology regions
# Removed: changing histology map



def shank_loop(func: Callable) -> Callable:
    """
    Decorator to loop over shanks and call the function with (shank: Union[List[str], Tuple[str, ...]], items: Union[Dict, Bunch]).

    This decorator automatically defaults 'shanks' to self.all_shanks if none are provided.
    It then iterates over each shank, fetching the corresponding items from self.shank_items,
    and calls the decorated function with shank and items as extra arguments.

    Parameters
    ----------
    func : Callable
        The function to decorate. It should have a signature where the first two arguments
        after 'self' are 'shank' (the current shank from the loop) and 'items'
        (self.shank_items[shank]).

    Returns
    -------
    Callable
        The wrapped function.
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs) -> Any:
        # Use all_shanks if none provided

        shanks = kwargs.pop('shanks', self.all_shanks)
        shanks = self.all_shanks if shanks is None else shanks
        configs = kwargs.pop('configs', self.loaddata.configs)
        configs = self.loaddata.configs if configs is None else configs
        data_only = kwargs.pop('data_only', False)

        results = []
        if configs:
            for config in configs:
                self.loaddata.current_config = config
                for shank in shanks:
                    self.loaddata.current_shank = shank
                    if not self.loaddata.get_current_shank().align_exists and not data_only:
                        continue

                    result = func(self, self.shank_items[shank][config], *args, **kwargs, shank=shank, config=config)
                    results.append(result)
        else:
            for shank in shanks:
                self.loaddata.current_shank = shank
                if not self.loaddata.get_current_shank().align_exists and not data_only:
                    continue
                result = func(self, self.shank_items[shank], *args, **kwargs, shank=shank, config=None)
                results.append(result)
        return results
    return wrapper



class MainWindow(QtWidgets.QMainWindow, Setup):

    @staticmethod
    def _instances():
        app = QtWidgets.QApplication.instance()
        return [w for w in app.topLevelWidgets() if isinstance(w, MainWindow)]

    @staticmethod
    def _get_or_create(title='Electrophysiology Atlas', **kwargs):
        av = next(filter(lambda e: e.isVisible() and e.windowTitle() == title,
                         MainWindow._instances()), None)
        if av is None:
            av = MainWindow(**kwargs)
            av.setWindowTitle(title)
        return av

    def __init__(self, offline=False, csv=None):
        super(MainWindow, self).__init__()

        self.csv = False if csv is None else csv
        self.offline = offline
        self.init_variables()
        self.init_layout()

        if offline:
            self.loaddata = ProbeLoaderLocal()
        elif csv:
            self.loaddata = ProbeLoaderCSV(csv)
            utils.populate_lists(self.loaddata.get_subjects(), self.subj_list, self.subj_combobox)
        else:
            self.loaddata = ProbeLoaderONE()
            utils.populate_lists(self.loaddata.get_subjects(), self.subj_list, self.subj_combobox)

    def init_variables(self):
        """
        Initialise variables
        """
        # Variables to do with probe dimension
        self.extend_feature = 1
        # Initialise with linear fit scaling as default
        self.lin_fit = True

        # Keep track of display options
        self.line_status = True
        self.label_status = True
        self.channel_status = True
        self.grid = True

        # Intialise probe dimensions for setting up figures
        self.probe_tip = 0
        self.probe_top = 3840
        self.probe_extra = 100
        self.view_total = [-2000, 6000]
        self.depth = np.arange(self.view_total[0], self.view_total[1], 20)
        # Padding to add to figures to make sure always same size viewbox
        self.pad = 0.05

        # Keep track of the display the mouse is hovering over
        self.hover_line = None
        self.hover_shank = None
        self.hover_idx = None
        self.hover_config = None

        # Keep track of the shanks we have
        self.all_shanks = list()
        self.shank_items = dict()


        # Keep track of histogram lut for slice images
        self.lut_levels = None
        self.lut_status = True

        # TODO make sure this works as expected
        # Keep track of normalization

        self.blockPlugins = False

    @shank_loop
    def init_shank_items(self, items: Union[Dict, Bunch], data_only=True, **kwargs):
        items.img_plots = list()
        items.line_plots = list()
        items.probe_plots = list()
        items.img_cbars = list()
        items.probe_cbars = list()
        items.scale_regions = np.empty((0, 1))
        items.slice_lines = list()
        items.slice_plots = list()
        items.probe_bounds = list()
        items.hist_regions = dict()

        items.traj_line = None
        items.slice_chns = None

        items.yrange = []

        items.probe_tip = 0
        items.probe_top = 3840
        items.probe_extra = 100
        items.view_total = [-2000, 6000]
        items.depth = np.arange(items.view_total[0], items.view_total[1], 20)
        items.pad = 0.05

    @shank_loop
    def init_align_items(self, items: Union[Dict, Bunch], **kwargs):
        items.lines_features = np.empty((0, 3))
        items.lines_tracks = np.empty((0, 1))
        items.points = np.empty((0, 1))

    @shank_loop
    def clear_shank_items(self, items: Union[Dict, Bunch], **kwargs):

        utils.remove_items(items.fig_img, items.img_plots)
        utils.remove_items(items.fig_img, items.img_cbars)
        utils.remove_items(items.fig_line, items.line_plots)
        utils.remove_items(items.fig_probe, items.probe_plots)
        utils.remove_items(items.fig_probe, items.probe_cbars)
        utils.remove_items(items.fig_probe, items.probe_cbars)

        items.fig_hist.clear()
        items.ax_hist.setTicks([])
        items.fig_hist_ref.clear()
        items.ax_hist_ref.setTicks([])
        items.fig_scale.clear()
        items.fit_scatter.setData()
        items.fit_plot.setData()


    @shank_loop
    def set_xaxis_range(self, items: Union[Dict, Bunch], fig, label=True, **kwargs):
        """
        Set the x-axis range for a given figure for specified shank or all shanks.

        Parameters
        ----------
        items : dict
            The dictionary or object containing plot items for the shank.
        fig : str
            Key to access the desired figure within items.
        label : bool, optional
            Whether to label the x-axis (default is True).
        """
        qt_plots.set_xaxis_range(items[fig], items, label=label)

    @shank_loop
    def set_yaxis_range(self, items: Union[Dict, Bunch], fig, **kwargs):
        """
        Set the y-axis range for a given figure for specified shank or all shanks.

        Parameters
        ----------
        shank : Any
            The current shank.
        items : dict
            The dictionary or object containing plot items for the shank.
        fig : str
            Key to access the desired figure within items.
        """
        qt_plots.set_yaxis_range(items[fig], items)

    def add_yrange_to_data(self, items, data):

        if isinstance(data, dict or Bunch):
            data['yrange'] = items.yrange
            data['pad'] = items.pad
        else:
            data.yrange = items.yrange
            data.pad = items.pad
            data = asdict(data)
        return data

    def create_blank_data(self, items, fig, fig_cb=None, img=False):

        data = Bunch()
        data['yrange'] = items.yrange
        data['pad'] = items.pad
        data['xrange'] = [0, 1]

        qt_plots.set_xaxis_range(items[fig], data, label=False)
        qt_plots.set_yaxis_range(items[fig], data)
        utils.set_axis(items[fig], 'bottom', pen='w', label='blank')
        if fig_cb:
            utils.set_axis(items[fig_cb], 'top', pen='w')
        if img:
            items.ephys_plot = None
            items.y_scale = None
            items.xrange = data['xrange']


    def execute_plugins(self, func, *args, **kwargs):

        if self.blockPlugins:
            return

        for _, plug in self.plugins.items():
            plug_func = plug.get(func, None)
            if plug_func is not None:
                plug_func(self, *args, **kwargs)


    # -------------------------------------------------------------------------------------------------
    # Plotting functions
    # -------------------------------------------------------------------------------------------------
    @shank_loop
    def plot_histology_panels(self, items: Union[Dict, Bunch], **kwargs):

        items.fig_hist.clear()
        data = self.add_yrange_to_data(items, items.hist_data)
        hist = qt_plots.plot_histology(items.fig_hist, data)
        items.hist_regions['left'] = hist
        items.tip_pos = pg.InfiniteLine(pos=items.probe_tip, angle=0, pen=utils.kpen_dot)
        items.top_pos = pg.InfiniteLine(pos=items.probe_top, angle=0, pen=utils.kpen_dot)

        items.fig_hist.addItem(items.tip_pos)
        items.fig_hist.addItem(items.top_pos)

        self.hover_region = items.hist_regions['left'][-2]

    @shank_loop
    def plot_histology_ref_panels(self, items: Union[Dict, Bunch], **kwargs):

        items.fig_hist_ref.clear()
        data = self.add_yrange_to_data(items, items.hist_data_ref)
        hist = qt_plots.plot_histology(items.fig_hist_ref, data, ax='right')
        items.hist_regions['right'] = hist
        items.tip_pos = pg.InfiniteLine(pos=items.probe_tip, angle=0, pen=utils.kpen_dot)
        items.top_pos = pg.InfiniteLine(pos=items.probe_top, angle=0, pen=utils.kpen_dot)

        items.fig_hist_ref.addItem(items.tip_pos)
        items.fig_hist_ref.addItem(items.top_pos)

    def plot_scale_factor_panels(self, shanks=None):
        if self.loaddata.configs is not None and self.loaddata.selected_config == 'both':
            results = self._plot_scale_factor_panels(shanks=shanks)
            for res in results:
                cbar = res['cbar']
                cbar.setAxis(cbar.ticks, cbar.label, loc='top', extent=20)
                cbar.setAxis([], loc='bottom', extent=20)
        else:
            self._plot_scale_factor_panels(shanks=shanks)


    @shank_loop
    def _plot_scale_factor_panels(self, items: Union[Dict, Bunch], **kwargs):
        shank = kwargs.get('shank')
        config = kwargs.get('config', None)

        items.scale_data['scale_factor'] = items.scale_data['scale'] - 0.5
        items.fig_scale.clear()
        data = self.add_yrange_to_data(items, items.scale_data)
        scale, cbar = qt_plots.plot_scale_factor(items.fig_scale, items.fig_scale_cb, data)
        items.scale_regions = scale

        return {'shank': shank, 'config': config, 'cbar': cbar}

    @shank_loop
    def plot_fit_panels(self,  items: Union[Dict, Bunch], **kwargs):

        data = {}
        data['x'] = self.loaddata.feature * 1e6
        data['y'] = self.loaddata.track * 1e6
        data['depth_lin'] = self.loaddata.feature2track_lin(items.depth, data['x'], data['y'])
        data['depth'] = items.depth
        qt_plots.plot_fit(items.fit_plot, items.fit_plot_lin, items.fit_scatter, data)

    @shank_loop
    def remove_fit_panels(self, items: Union[Dict, Bunch], **kwargs):
        # TODO pass these in as arguments instead when we call this method
        shank = kwargs.get('shank')
        if shank != self.selected_shank:
            items.fit_plot.setData()
            items.fit_scatter.setData()
            items.fit_plot_lin.setData()

    @shank_loop
    def _plot_slice_panels(self, items, plot_type, **kwargs):
        shank = kwargs.get('shank')
        data = self.loaddata.slice_plots.get(plot_type, None)
        self.slice_init = plot_type

        if data is None:
            return None, None, items.fig_slice, shank

        data_chns = {}
        data_chns['x'] = self.loaddata.xyz_track[:, 0]
        data_chns['y'] = self.loaddata.xyz_track[:, 2]

        items.slice_plots = utils.remove_items(items.fig_slice, items.slice_plots)
        if items.traj_line:
            items.fig_slice.removeItem(items.traj_line)
        img, cbar, traj = qt_plots.plot_slice(items.fig_slice, data, data_chns)
        items.slice_plots.append(img)
        items.traj_line = traj

        return img, cbar, items.fig_slice, shank

    def plot_slice_panels(self, plot_type, data_only=True):

        if plot_type != self.slice_init:
            if not self.channel_status:
                self.toggle_channels()
            self.lut_levels = None
        self.slice_figs = Bunch()
        self.imgs = []

        if self.loaddata.configs is not None:
            if self.loaddata.selected_config == 'both':
                results = self._plot_slice_panels(plot_type, data_only=data_only, configs=['quarter'])
                self.slice_figs = {res[3]: res[2] for res in results}
                self.plot_channel_panels()
            else:
                results = self._plot_slice_panels(plot_type, data_only=data_only, configs=[self.loaddata.selected_config])
                self.slice_figs = {res[3]: res[2] for res in results}
                self.plot_channel_panels(configs=[self.loaddata.selected_config])
        else:
            results = self._plot_slice_panels(plot_type)
            self.slice_figs = {res[3]: res[2] for res in results}
            self.plot_channel_panels()

        self.imgs = [res[0] for res in results]

        cb = results[0][1]

        if self.slice_init != 'Annotation':
            if not self.lut_status:
                self.lut_layout.addItem(self.slice_LUT)
                self.lut_status = True
            self.slice_LUT.blockSignals(True)
            self.slice_LUT.setImageItem(self.imgs[0])
            self.slice_LUT.gradient.setColorMap(cb.map)
            self.slice_LUT.autoHistogramRange()
            hist_levels = self.slice_LUT.getLevels()
            hist_val, hist_count = self.imgs[0].getHistogram()
            upper_idx = np.where(hist_count > 10)[0][-1]
            upper_val = hist_val[upper_idx]
            if hist_levels[0] != 0:
                self.set_lut_levels(levels=[hist_levels[0], upper_val])
            else:
                self.set_lut_levels()
            self.slice_LUT.blockSignals(False)
        else:
            if self.lut_status:
                self.lut_layout.removeItem(self.slice_LUT)
                self.lut_status = False

    def set_lut_levels(self, levels=None):
        levels = levels or self.lut_levels
        if levels is None:
            return

        for img in self.imgs:
            img.setLevels(levels)
        self.slice_LUT.setLevels(min=levels[0], max=levels[1])

    def update_lut_levels(self):
        self.lut_levels = self.slice_LUT.getLevels()

        for img in self.imgs:
            img.setLevels(self.lut_levels)

    @shank_loop
    def plot_channel_panels(self, items: Union[Dict, Bunch], **kwargs):
        shank = kwargs.get('shank')

        self.channel_status = True
        data = {}
        data['xyz_channels'] = self.loaddata.xyz_channels
        data['track_lines'] = self.loaddata.track_lines

        c = 'g' if items.config == 'dense' else 'r'

        items.slice_lines = utils.remove_items(self.slice_figs[shank], items.slice_lines)
        if items.slice_chns:
            self.slice_figs[shank].removeItem(items.slice_chns)
        lines, chns = qt_plots.plot_channels(self.slice_figs[shank], data, colour=c)
        items.slice_chns = chns
        items.slice_lines += lines

    def plot_scatter_panels(self, plot_type, data_only=True, **kwargs):

        self.img_init = plot_type
        if self.loaddata.configs is not None and self.loaddata.selected_config == 'both':
            self.scatter_levels = self.get_normalised_levels('scatter_plots', plot_type)
            results = self._plot_scatter_panels(plot_type, data_only=data_only, **kwargs)
            self.plot_dual_colorbar(results, 'fig_dual_img_cb')

        else:
            self.scatter_levels = Bunch.fromkeys(self.all_shanks, None)
            self._plot_scatter_panels(plot_type, data_only=data_only, **kwargs)

        self.execute_plugins('plot_scatter_panels', plot_type)


    @shank_loop
    def _plot_scatter_panels(self,  items: Union[Dict, Bunch], plot_type, data_only=True, **kwargs):
        shank = kwargs.get('shank')
        config = kwargs.get('config', None)
        data = self.loaddata.scatter_plots.get(plot_type, None)
        items.img_plots = utils.remove_items(items.fig_img, items.img_plots)
        items.img_cbars = utils.remove_items(items.fig_img_cb, items.img_cbars)

        if data is None:
            self.create_blank_data(items, 'fig_img', fig_cb='fig_img_cb', img=True)
            return {'shank': shank, 'config': config, 'cbar': None}

        data = self.add_yrange_to_data(items, data)
        scat, cbar = qt_plots.plot_scatter(items.fig_img, items.fig_img_cb, data, levels=self.scatter_levels[shank])
        items.img_plots.append(scat)
        items.img_cbars.append(cbar)
        items.ephys_plot = scat
        if data['cluster']:
            items.cluster_data = data['x']
            items.ephys_plot.sigClicked.connect(lambda plot, points: cluster_callback(self, items, plot, points))
        items.y_scale = 1
        items.xrange = data['xrange']

        return {'shank': shank, 'config': config, 'cbar': cbar}

    @shank_loop
    def plot_line_panels(self, items: Union[Dict, Bunch], plot_type, data_only=True, **kwargs):

        data = self.loaddata.line_plots.get(plot_type, None)
        items.line_plots = utils.remove_items(items.fig_line, items.line_plots)
        self.line_init = plot_type

        if data is None:
            self.create_blank_data(items,'fig_line')
            return

        data = self.add_yrange_to_data(items, data)
        items.line_plots = utils.remove_items(items.fig_line, items.line_plots)
        lin = qt_plots.plot_line(items.fig_line, data)
        items.line_plots.append(lin)

    def plot_probe_panels(self,  plot_type, data_only=True, **kwargs):

        self.probe_init = plot_type
        if self.loaddata.configs is not None and self.loaddata.selected_config == 'both':
            self.probe_levels = self.get_normalised_levels('probe_plots', plot_type)
            results = self._plot_probe_panels(plot_type, data_only=data_only, **kwargs)
            self.plot_dual_colorbar(results, 'fig_dual_probe_cb')

        else:
            self.probe_levels = Bunch.fromkeys(self.all_shanks, None)
            self._plot_probe_panels(plot_type, data_only=data_only, **kwargs)

        self.execute_plugins('plot_probe_panels', plot_type)

    @shank_loop
    def _plot_probe_panels(self, items: Union[Dict, Bunch], plot_type, data_only=True, **kwargs):

        shank = kwargs.get('shank')
        config = kwargs.get('config', None)

        items.probe_plots = utils.remove_items(items.fig_probe, items.probe_plots)
        items.probe_cbars = utils.remove_items(items.fig_probe_cb, items.probe_cbars)
        items.probe_bounds = utils.remove_items(items.fig_probe, items.probe_bounds)

        data = self.loaddata.probe_plots.get(plot_type, None)
        if data is None:
            self.create_blank_data(items, 'fig_probe', fig_cb='fig_probe_cb')
            return {'shank': shank, 'config': config, 'cbar': None}

        data = self.add_yrange_to_data(items, data)
        prbs, cbar, bnds = qt_plots.plot_probe(items.fig_probe, items.fig_probe_cb, data, levels=self.probe_levels[shank])
        items.probe_plots += prbs
        items.probe_bounds += bnds
        items.probe_cbars.append(cbar)

        return {'shank': shank, 'config': config, 'cbar': cbar}


    def plot_image_panels(self, plot_type, data_only=True, **kwargs):

        self.img_init = plot_type
        if self.loaddata.configs is not None and self.loaddata.selected_config == 'both':
            self.img_levels = self.get_normalised_levels('image_plots', plot_type)
            results = self._plot_image_panels(plot_type, data_only=data_only, **kwargs)
            self.plot_dual_colorbar(results, 'fig_dual_img_cb')
        else:
            self.img_levels = Bunch.fromkeys(self.all_shanks, None)
            self._plot_image_panels(plot_type, data_only=data_only, **kwargs)

    @shank_loop
    def _plot_image_panels(self, items: Union[Dict, Bunch], plot_type, data_only=True, **kwargs):

        shank = kwargs.get('shank')
        config = kwargs.get('config', None)

        data = self.loaddata.image_plots.get(plot_type, None)
        items.img_plots = utils.remove_items(items.fig_img, items.img_plots)
        items.img_cbars = utils.remove_items(items.fig_img_cb, items.img_cbars)

        if data is None:
            self.create_blank_data(items, 'fig_img', fig_cb='fig_img_cb', img=True)
            return {'shank': shank, 'config': config, 'cbar': None}

        data = self.add_yrange_to_data(items, data)

        img, cbar = qt_plots.plot_image(items.fig_img, items.fig_img_cb, data, levels=self.img_levels[shank])
        items.img_plots.append(img)
        items.img_cbars.append(cbar)
        items.ephys_plot = img
        items.y_scale = data['scale'][1]
        items.xrange = data['xrange']

        return {'shank': shank, 'config': config, 'cbar': cbar}


    def get_normalised_levels(self, plot_group, plot_type):
        if self.normalise_levels == 'both' or self.normalise_levels is None:
            return Bunch.fromkeys(self.all_shanks, None)
        levels = Bunch()
        for shank in self.all_shanks:
            # Try to get data from the primary config
            data = self.loaddata.get_plot(shank, plot_group, plot_type, self.normalise_levels)
            # If no data found, fallback to the other config
            if not data:
                other_config = next(c for c in self.loaddata.configs if c != self.normalise_levels)
                data = self.loaddata.get_plot(shank, plot_group, plot_type, other_config)

            levels[shank] = data.levels if data else None

        return levels

    def plot_dual_colorbar(self, results, fig):
        cbs = defaultdict(dict)
        for res in results:
            cbs[res['shank']][res['config']] = res['cbar']

        for shank in cbs.keys():
            cb_dense = cbs[shank].get('dense')
            cb_quarter = cbs[shank].get('quarter')

            cmap = cb_quarter.cmap_name if cb_quarter else (cb_dense.cmap_name if cb_dense else None)
            if not cmap:
                continue

            fig_cb = self.shank_items[shank]['quarter'][fig]
            cbar = qt_cbar.ColorBar(cmap, plot_item=fig_cb)
            if cb_quarter:
                cbar.setAxis(cb_quarter.ticks, cb_quarter.label, loc='top', extent=20)
            else:
                cbar.setAxis([], cb_dense.label, loc='top', extent=20)

            if cb_dense:
                cbar.setAxis(cb_dense.ticks, loc='bottom', extent=20)


    def update_plots(self, shanks=()) -> None:
        """
        Refresh all plots to reflect the current alignment state.
        """

        self.get_scaled_histology(shanks=shanks)
        self.plot_histology_panels(shanks=shanks)
        self.plot_scale_factor_panels(shanks=shanks)
        self.plot_fit_panels(shanks=shanks)
        if self.loaddata.selected_config == 'both' or not self.loaddata.selected_config:
            self.plot_channel_panels(shanks=shanks)
        else:
            self.plot_channel_panels(shanks=shanks, configs=[self.loaddata.selected_config])
        self.remove_reference_lines_from_display(shanks=shanks)
        self.add_reference_lines_to_display(shanks=shanks)
        self.align_reference_lines(shanks=shanks)
        self.set_yaxis_range('fig_hist', shanks=shanks)
        self.update_string()

        self.execute_plugins('update_plots')

        # for hook in self.plot_hooks:
        #     hook(self)
        # def register_hook(self, func):
        #     self.plot_hooks.append(func)


    # -------------------------------------------------------------------------------------------------
    # Session selection
    # -------------------------------------------------------------------------------------------------
    def on_subject_selected(self, idx):
        """
        Triggered when subject is selected from drop down list options
        :param idx: index chosen subject (item) in drop down list
        :type idx: int
        """
        self.loaded = None
        self.sess_list.clear()
        sessions = self.loaddata.get_sessions(idx)
        utils.populate_lists(sessions, self.sess_list, self.sess_combobox)
        self.on_session_selected(0)

        self.data_button.setStyleSheet(utils.button_style['activated'])

    def on_session_selected(self, idx):
        """
        Triggered when session is selected from drop down list options
        :param idx: index of chosen session (item) in drop down list
        :type idx: int
        """
        self.loaded = None
        self.shank_list.clear()
        self.config_list.clear()
        self.loaddata.get_config(0)
        shanks = self.loaddata.get_shanks(idx)
        # if len(shanks) > 1:
        utils.populate_lists(shanks, self.shank_list, self.shank_combobox)
        self.on_shank_selected(0)
        self.data_button.setStyleSheet(utils.button_style['activated'])

    def on_shank_selected(self, idx):
        """
        Online version
        """
        self.align_list.clear()
        self.loaddata.get_info(idx)
        # Update prev_alignments
        self.prev_alignments = self.loaddata.get_previous_alignments()
        utils.populate_lists(self.prev_alignments, self.align_list, self.align_combobox)
        self.loaddata.get_starting_alignment(0)

        if self.loaded is not None:
            if not self.shank_tabs.grid_layout:
                self.set_current_tab(idx)
            self.shank_button_pressed()

    def on_alignment_selected(self, idx):
        self.loaddata.get_starting_alignment(idx)
        if self.loaded is not None:
            self.align_button_pressed()

    def on_config_selected(self, idx, init=False):

        self.loaddata.get_config(idx)
        self.normalise_idx = 0
        self.normalise_levels = self.loaddata.selected_config
        self.setup(init=init)

        if not init:
            self.execute_plugins('on_config_selected')
            self.setFocus()
            self.raise_()
            self.activateWindow()

    def on_folder_selected(self):
        """
        Triggered in offline mode when folder button is clicked
        """
        self.loaded = None
        self.align_list.clear()
        self.shank_list.clear()
        folder_path = Path(QtWidgets.QFileDialog.getExistingDirectory(None, "Select Folder"))
        self.folder_line.setText(str(folder_path))
        shank_options = self.loaddata.get_shanks(folder_path)
        utils.populate_lists(shank_options, self.shank_list, self.shank_combobox)
        self.on_shank_selected(0)

    def data_button_pressed(self):
        """
        Triggered when Get Data button pressed, uses subject and session info to find eid and
        downloads and computes data needed for GUI display
        """
        if self.loaded:
            return

        start = time.time()
        self.all_shanks = list(self.loaddata.shanks.keys())
        self.loaddata.load_data()
        self.loaded = True
        self.selected_shank = self.loaddata.selected_shank
        self.selected_idx = self.loaddata.shank_idx
        self.populate_menu_bar()
        if self.csv:
            utils.populate_lists(self.loaddata.possible_configs, self.config_list, self.config_combobox)

        self.fix_reference_colours = True if len(self.all_shanks) > 1 else False

        self.on_config_selected(0, init=True)


        self.execute_plugins('data_button_pressed')

        self.init_display()
        self.data_button.setStyleSheet(utils.button_style['deactivated'])

        # self.subj_line.clearFocus()
        # self.subj_combobox.clearFocus()

        self.setFocus()
        self.raise_()
        self.activateWindow()

        print(time.time() - start)


    def shank_button_pressed(self):

        self.loaddata.set_init_alignment()
        self.selected_shank = self.loaddata.selected_shank
        self.selected_idx = self.loaddata.shank_idx

        self.remove_points_from_display()
        self.add_points_to_display()

        if not self.shank_tabs.grid_layout:
            self.remove_fit_panels()
            self.plot_fit_panels(shanks=[self.selected_shank])
        else:
            self.select_background()

    def align_button_pressed(self):

        self.remove_reference_lines_from_display(shanks=[self.selected_shank])
        self.init_align_items(shanks=[self.selected_shank])
        self.set_init_reference_lines(shanks=[self.selected_shank])
        self.update_plots(shanks=[self.selected_shank])


    def setup(self, init=True):
        if not init:
            self.remove_reference_lines_from_display()

        self.fig_fit.clear()
        self.clear_tabs()

        if init:
            self.init_shanks()
            self.init_align_items()

        self.init_shank_items(data_only=True)
        self.init_tabs()

        self.set_probe_lims(data_only=True)
        self.set_yaxis_lims(self.loaddata.y_min, self.loaddata.y_max, data_only=True)

        # TODO is this generic enough?
        self.blockPlugins = True
        utils.find_actions(self.img_init, self.img_options_group).trigger() if self.img_init else None
        utils.find_actions(self.line_init, self.line_options_group).trigger() if self.line_init else None
        utils.find_actions(self.probe_init, self.probe_options_group).trigger() if self.probe_init else None
        utils.find_actions(self.slice_init, self.slice_options_group).trigger()
        self.blockPlugins = False
        # TODO check
        # utils.find_actions(self.unit_init, self.filter_options_group).trigger()

        # Initialise histology plots
        self.get_scaled_histology()
        self.plot_histology_ref_panels()
        self.plot_histology_panels()
        self.plot_scale_factor_panels()
        self.label_status = False
        self.toggle_labels()
        self.update_string()

        if init:
            self.set_init_reference_lines()
        else:
            self.add_reference_lines_to_display()
            self.set_lut_levels()

        # Add reference points for selected shank
        self.remove_points_from_display()
        self.add_points_to_display()

        # Plot fits
        self.plot_fit_panels()

        # Select highlighted shank
        self.select_background()


    def filter_unit_pressed(self, filter_type, data_only=True):
        if filter_type == self.filter_init:
            return

        self.blockPlugins = True
        self._filter_units(filter_type, data_only=data_only)
        utils.find_actions(self.img_init, self.img_options_group).trigger() if self.img_init else None
        utils.find_actions(self.line_init, self.line_options_group).trigger() if self.line_init else None
        utils.find_actions(self.probe_init, self.probe_options_group).trigger() if self.probe_init else None
        self.filter_init = filter_type
        self.blockPlugins = False

        self.execute_plugins('filter_unit_pressed')


    @shank_loop
    def _filter_units(self, items, filter_type, data_only=True, **kwargs):
        self.loaddata.get_current_shank().filter_plots(filter_type)



    # -------------------------------------------------------------------------------------------------
    # Upload/ save data
    # -------------------------------------------------------------------------------------------------

    def complete_button_pressed(self):
        """
        Triggered when complete button or Shift+F key pressed. Uploads final channel locations to
        Alyx
        """

        if not self.offline:
            display_qc(self)

        upload = QtWidgets.QMessageBox.question(
            self, '', "Upload alignment?", QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)

        if upload == QtWidgets.QMessageBox.Yes:

            info = self.loaddata.upload_data()
            self.prev_alignments = self.loaddata.load_previous_alignments()
            utils.populate_lists(self.prev_alignments, self.align_list, self.align_combobox)
            self.loaddata.get_starting_alignment(0)

            QtWidgets.QMessageBox.information(self, 'Status', info)
        else:
            QtWidgets.QMessageBox.information(self, 'Status', "Channels not saved")


    # -------------------------------------------------------------------------------------------------
    # Fitting functions
    # -------------------------------------------------------------------------------------------------
    @shank_loop
    def offset_hist_data(self, items, val: Optional[float] = None, **kwargs) -> None:
        """
        Apply an offset to brain regions based on probe tip position.

        Parameters
        ----------
        val : float, optional
            Offset value in meters. If None, uses current value of  self.tip_pos
        """

        val = val or self.tip_pos.value() / 1e6
        self.loaddata.offset_hist_data(val)

    @shank_loop
    def scale_hist_data(self, items, **kwargs) -> None:
        """
        Scale brain regions along the probe track based on reference lines.
        """

        line_track = np.array([line[0].pos().y() for line in items.lines_tracks]) / 1e6
        line_feature = np.array([line[0].pos().y() for line in items.lines_features]) / 1e6

        # We loop here over the configs to set for both
        self.loaddata.scale_hist_data(line_track, line_feature, extend_feature=self.extend_feature, lin_fit=self.lin_fit)

    @shank_loop
    def get_scaled_histology(self, items: Union[Dict, Bunch], **kwargs) -> None:
        """
        Retrieve scaled histological data after alignment operations.

        """
        items.hist_data, items.hist_data_ref, items.scale_data = self.loaddata.get_scaled_histology()


    def apply_fit(self, fit_function: Callable, **kwargs) -> None:
        """
        Apply a given fitting function to histology data and update all relevant plots.

        Parameters
        ----------
        fit_function : Callable
            A function that modifies the alignment
        **kwargs :
            Additional arguments passed to `fit_function`.
        """

        fit_function(**kwargs)

        self.update_plots(shanks=[self.selected_shank])


    def offset_button_pressed(self) -> None:
        """
        Called when the offset button or 'O' key is pressed.
        Applies offset based on location of the probe tip line and refreshes plots.
        """
        self.apply_fit(self.offset_hist_data, shanks=[self.selected_shank])


    def movedown_button_pressed(self) -> None:
        """
        Called when Shift+Down is pressed. Offsets probe tip 100µm down.
        """
        self.apply_fit(self.offset_hist_data, shanks=[self.selected_shank], val=-100/1e6)


    def moveup_button_pressed(self) -> None:
        """
        Called when Shift+Up is pressed. Offsets probe tip 100µm up.
        """
        self.apply_fit(self.offset_hist_data, shanks=[self.selected_shank], val=100/1e6)


    def fit_button_pressed(self) -> None:
        """
        Called when the fit button or Enter is pressed.
        Scales regions using reference lines and refreshes plots.
        """
        self.apply_fit(self.scale_hist_data, shanks=[self.selected_shank])

    def next_button_pressed(self) -> None:
        """
        Called when right key pressed.
        Updates all plots and indices with next alignment.
        Ensures user cannot go past latest move

        Note self.selected_shank is passed in already in the setup
        """

        if self.loaddata.next_idx():
            # TODO should the loop be here or in probe_loader?
            self.update_plots(shanks=[self.selected_shank])

    #@shank_loop
    def prev_button_pressed(self) -> None:
        """
        Called when left key pressed.
        Updates all plots and indices with previous alignment.
        Note self.selected_shank is passed in already in the setup
        """

        if self.loaddata.prev_idx():
            # TODO should the loop be here or in probe_loader?
            self.update_plots(shanks=[self.selected_shank])

    def reset_button_pressed(self) -> None:
        """
        Called when Reset button or Shift+R is pressed.
        Resets feature and track alignment to original alignment and updates plots.
        """

        self.remove_reference_lines_from_display(shanks=[self.selected_shank])

        self.init_align_items(shanks=[self.selected_shank])

        self.reset_feature_and_tracks(shanks=[self.selected_shank])

        self.set_init_reference_lines(shanks=[self.selected_shank])

        self.update_plots(shanks=[self.selected_shank])

    @shank_loop
    def reset_feature_and_tracks(self, items, **kwargs):
        self.loaddata.reset_features_and_tracks()

    def lin_fit_option_changed(self, state: int) -> None:
        """
        Toggle the use of linear fit for scaling histology data.

        Parameters
        ----------
        state : int
            0 disables linear fit, any other value enables it.
        """
        self.lin_fit = bool(state)
        self.fit_button_pressed()

    def update_string(self) -> None:
        """
        Updates on-screen text showing current and total alignment steps.
        """
        self.idx_string.setText(f"Current Index = {self.loaddata.current_idx}")
        self.tot_idx_string.setText(f"Total Index = {self.loaddata.total_idx}")

    # -------------------------------------------------------------------------------------------------
    # Mouse interactions
    # -------------------------------------------------------------------------------------------------

    def on_mouse_double_clicked(self, event, idx) -> None:
        """
        Handles a double-click event on the ephys or histology plots.

        Adds a movable reference line on the ephys and histology plots

        Parameters
        ----------
        event : pyqtgraph.GraphicsScene.mouseEvents.MouseClickEvent
            The mouse double-click event.
        """

        if event.double():
            if idx != self.selected_idx:
                self.shank_combobox.setCurrentIndex(idx)
                self.on_shank_selected(idx)

            if self.loaddata.configs is not None:
                config = 'quarter' if self.loaddata.selected_config == 'both' else self.loaddata.selected_config
                items = self.shank_items[self.selected_shank][config]
                pos = items.ephys_plot.mapFromScene(event.scenePos())
                y_scale = items.y_scale
                for config in self.loaddata.configs:
                    items = self.shank_items[self.selected_shank][config]
                    self.create_reference_line(pos.y() * y_scale, items)
            else:
                items = self.shank_items[self.selected_shank]
                pos = items.ephys_plot.mapFromScene(event.scenePos())
                self.create_reference_line(pos.y() * items.y_scale, items)



    def on_mouse_hover(self, hover_items: List[pg.GraphicsObject], name: str, idx: int, config: str) -> None:
        """
        Handles mouse hover events over pyqtgraph plot items.

        Identifies reference lines or linear regions the mouse is hovering over
        to allow interactive operations like deletion or displaying additional info.

        Parameters
        ----------
        items : list of pyqtgraph.GraphicsObject
            List of items under the mouse cursor.
        """
        self.hover_idx = idx
        self.hover_shank = name
        self.hover_config = config

        if len(hover_items) > 1:
            self.hover_line = []
            hover_item0, hover_item1 = hover_items[0], hover_items[1]
            if isinstance(hover_item0, pg.InfiniteLine):
                self.hover_line = hover_item0
            elif isinstance(hover_item1, pg.LinearRegionItem):
                if self.loaddata.configs is not None:
                    items = self.shank_items[name][config]
                else:
                    items = self.shank_items[name]
                # Check if we are on the fig_scale plot
                if hover_item0 == items.fig_scale:
                    idx = np.where(items.scale_regions == hover_item1)[0][0]
                    items.fig_scale_ax.setLabel('Scale = ' + str(np.around(items.scale_data['scale'][idx], 2)))
                    return
                # Check if we are on the histology plot
                if hover_item0 == items.fig_hist:
                    self.hover_region = hover_item1
                    return
                if hover_item0 == items.fig_hist_ref:
                    self.hover_region = hover_item1
                    return


    # -------------------------------------------------------------------------------------------------
    # Display options
    # -------------------------------------------------------------------------------------------------

    @shank_loop
    def select_background(self, items: Union[Dict, Bunch], **kwargs):
        shank = kwargs.get('shank')
        if shank == self.selected_shank:
            items.header.setStyleSheet(utils.tab_style['selected'])
        else:
            items.header.setStyleSheet(utils.tab_style['deselected'])


    def toggle_labels(self) -> None:
        """
        Toggle visibility of brain region labels on histology plots.

        Triggered by pressing Shift+A. Updates the pens for axis items on both the main
        and reference histology plots to show or hide Allen atlas region labels.
        """
        self.label_status = not self.label_status
        self.toggle_label()

    @shank_loop
    def toggle_label(self,  items: Union[Dict, Bunch], **kwargs) -> None:

        """
        Toggle visibility of reference lines on electrophysiology and histology plots.

        Triggered by pressing Shift+L.

        Returns
        -------
        None
        """

        pen = 'k' if self.label_status else None
        items.ax_hist_ref.setPen(pen)
        items.ax_hist_ref.setTextPen(pen)
        items.ax_hist.setPen(pen)
        items.ax_hist.setTextPen(pen)

        items.fig_hist_ref.update()
        items.fig_hist.update()

    def toggle_reference_lines(self) -> None:
        """
        Toggle visibility of reference lines on electrophysiology and histology plots.

        Triggered by pressing Shift+L.

        Returns
        -------
        None
        """
        self.line_status = not self.line_status
        if not self.line_status:
            self.remove_reference_lines_from_display()
        else:
            self.add_reference_lines_to_display()

    def toggle_channels(self) -> None:
        """
        Toggle visibility of channels and trajectory lines on the histology slice image.

        Triggered by pressing Shift+C. Adds or removes the visual indicators of the probe
        trajectory and channels on the slice view.

        Returns
        -------
        None
        """

        self.channel_status = not self.channel_status
        self.toggle_channel()

    @shank_loop
    def toggle_channel(self, items: Union[Dict, Bunch], **kwargs) -> None:
        """
        Toggle between displaying histology boundaries or block regions

        Triggered by pressing Shift+N. If nearby boundaries are not yet computed, this will
        call the boundary computation method

        Returns
        -------
        None
        """
        shank = kwargs.get('shank')

        toggleItem = self.slice_figs[shank].addItem if self.channel_status else self.slice_figs[shank].removeItem

        if items.traj_line:
            toggleItem(items.traj_line)
        toggleItem(items.slice_chns)
        for line in items.slice_lines:
            toggleItem(line)

    def reset_axis_button_pressed(self) -> None:
        """
        Reset zoomable plot axis to default

        Triggered by pressing Shift+A.
        Returns
        -------
        None
        """

        self.set_yaxis_range('fig_hist')
        self.set_yaxis_range('fig_hist_ref')
        self.set_yaxis_range('fig_img')
        self.set_xaxis_range('fig_img', label=False)
        if self.loaddata.configs is not None:
            if self.loaddata.selected_config == 'both':
                configs = ['quarter']
            else:
                configs = [self.loaddata.selected_config]
            self.reset_slice_axis(configs=configs)
        else:
            self.reset_slice_axis()

    @shank_loop
    def reset_slice_axis(self, items: Union[Dict, Bunch], **kwargs) -> None:
        items.fig_slice.autoRange()


    def loop_shanks(self, direction:int):
        idx = np.mod(self.selected_idx + direction, len(self.all_shanks))
        self.shank_combobox.setCurrentIndex(idx)
        self.on_shank_selected(idx)


    def layout_changed(self) -> None:
        """

        Returns
        -------
        None
        """
        self.shank_combobox.setCurrentIndex(self.selected_idx)
        self.on_shank_selected(self.selected_idx)

    def shank_tab_changed(self, idx, from_hist=False):
        """

        Returns
        -------
        None
        """
        self.shank_combobox.setCurrentIndex(idx)
        self.on_shank_selected(idx)
        if not from_hist:
            self.hist_tabs.tab_widget.setCurrentIndex(idx)
        self.remove_fit_panels()
        self.plot_fit_panels(shanks=[self.selected_shank])

    def hist_tab_changed(self, idx: int) -> None:
        """

        Returns
        -------
        None
        """
        self.shank_tabs.tab_widget.setCurrentIndex(idx)

    def toggle_layout(self) -> None:
        """

        Returns
        -------
        None
        """
        # Change the display of the histology slices
        self.hist_tabs.toggle_layout()
        self.hist_tabs.tab_widget.setCurrentIndex(self.selected_idx)
        # Change the display of the shank displays
        self.shank_tabs.toggle_layout()
        self.shank_tabs.tab_widget.setCurrentIndex(self.selected_idx)

        if self.shank_tabs.grid_layout:
            self.plot_fit_panels()
            self.select_background()
        else:
            self.remove_fit_panels()

    def set_current_tab(self, idx: int) -> None:
        """

        Returns
        -------
        None
        """
        self.shank_tabs.tab_widget.blockSignals(True)
        self.shank_tabs.tab_widget.setCurrentIndex(idx)
        self.hist_tabs.tab_widget.setCurrentIndex(idx)
        self.shank_tabs.tab_widget.blockSignals(False)


    def on_normalise_levels(self):
        if len(self.loaddata.possible_configs) == 0:
            return
        self.normalise_idx += 1
        idx = np.mod(self.normalise_idx, len(self.loaddata.possible_configs))
        self.normalise_levels = self.loaddata.possible_configs[idx]
        utils.find_actions(self.img_init, self.img_options_group).trigger() if self.img_init else None
        utils.find_actions(self.probe_init, self.probe_options_group).trigger() if self.probe_init else None

    def on_fig_size_changed(self) -> None:
        """

        Returns
        -------
        None
        """
        self.lin_fit_option.move(70, 10)
    # -------------------------------------------------------------------------------------------------
    # Probe top and tip lines
    # -------------------------------------------------------------------------------------------------

    @shank_loop
    def set_probe_lims(self, items: Union[Dict, Bunch], data_only=True, **kwargs) -> None:
        """
        Set the limits for the probe tip and probe top, and update the associated lines accordingly.

        Parameters
        ----------
        shank : list of str or tuple of str
            Identifiers for the shank(s) whose reference lines and points are to be removed.
        items : dict or Bunch
            A structure containing plot elements for a shank
        Returns
        -------
        None
        """

        items.probe_tip = self.loaddata.chn_min
        items.probe_top = self.loaddata.chn_max

        for top_line in items.probe_top_lines:
            top_line.setY(items.probe_top)
        for tip_line in items.probe_tip_lines:
            tip_line.setY(items.probe_tip)

    @shank_loop
    def set_yaxis_lims(self, items: Union[Dict, Bunch], min_val, max_val, data_only=True, **kwargs) -> None:
        """
        Set the limits for the probe tip and probe top, and update the associated lines accordingly.

        Parameters
        ----------
        shank : list of str or tuple of str
            Identifiers for the shank(s) whose reference lines and points are to be removed.
        items : dict or Bunch
            A structure containing plot elements for a shank
        Returns
        -------
        None
        """

        min_val = self.loaddata.chn_min
        max_val = self.loaddata.chn_max

        items.yrange = [min_val - items.probe_extra, max_val + items.probe_extra]

    # -------------------------------------------------------------------------------------------------
    # Reference lines
    # -------------------------------------------------------------------------------------------------

    @shank_loop
    def set_init_reference_lines(self, items: Union[Dict, Bunch], **kwargs) -> None:
        """
        Finds the initial alignment for specified shanks and creates reference lines
        Parameters
        ----------
        shank : list of str or tuple of str
            Identifiers for the shank(s) whose reference lines and points are to be removed.

        items : dict or Bunch
            A structure containing plot elements for a shank
        Returns
        -------
        None
        """
        self.loaddata.set_init_alignment()
        feature_prev = self.loaddata.feature_prev
        if np.any(feature_prev):
            self.create_reference_lines(feature_prev[1:-1] * 1e6, items)

    def create_reference_line(self, pos: float, items: Union[Dict, Bunch]) -> None:
        """
        Create a single movable horizontal reference line and corresponding scatter point

        This includes:
        - A track line in the histology figure
        - Feature lines in the image, line, and probe figures that are synchronized
        - A scatter point in the fit figure indicating the correspondence

        Parameters
        ----------
        pos : float
            Y-axis position at which to draw the horizontal line.
        items : dict or Bunch
            A structure containing plot elements for a shank

        Returns
        -------
        None
        """
        colour = items.colour if self.fix_reference_colours else None
        pen, brush = utils.create_line_style(colour=colour)

        # Reference line on histology figure (track)
        line_track = pg.InfiniteLine(pos=pos, angle=0, pen=pen, movable=True)
        line_track.sigPositionChanged.connect(lambda track, i=items.idx, c=items.config: self.update_track_reference_line(track, i, c))
        line_track.setZValue(100)
        items.fig_hist.addItem(line_track)

        # Reference lines on electrophysiology figures (feature)
        line_features = []
        for fig in [items.fig_img, items.fig_line, items.fig_probe]:
            line_feature = pg.InfiniteLine(pos=pos, angle=0, pen=pen, movable=True)
            line_feature.setZValue(100)
            line_feature.sigPositionChanged.connect(lambda feature, i=items.idx, c=items.config: self.update_feature_reference_line(feature, i, c))
            fig.addItem(line_feature)
            line_features.append(line_feature)

        items.lines_features = np.vstack([items.lines_features, line_features])
        items.lines_tracks = np.vstack([items.lines_tracks, line_track])

        # Add marker to fit figure
        point = pg.PlotDataItem()
        point.setData(x=[line_track.pos().y()], y=[line_features[0].pos().y()],
                      symbolBrush=brush, symbol='o', symbolSize=10)
        self.fig_fit.addItem(point)
        items.points = np.vstack([items.points, point])

    def create_reference_lines(self, positions: Union[np.ndarray, List[float]], items: Union[Dict, Bunch]) -> None:
        """
        Create movable horizontal reference lines across electrophysiology and histology figures at multiple positions

        For each y-position in `positions`, this method creates:
        - A movable horizontal line in the histology figure (track line)
        - Synchronized lines in the image, line, and probe figures (feature lines)
        - A corresponding scatter point in the fit figure

        Parameters
        ----------
        positions : array-like of float
            List or array of y-axis positions at which to draw horizontal lines across histology, image, line,
            probe, and fit figures.
        items : dict or Bunch
            A structure containing plot elements for a shank

        Returns
        -------
        None
        """

        for pos in positions:
            self.create_reference_line(pos, items)

    def delete_reference_line(self) -> None:
        """
        Delete the currently selected reference line from all plots.

        Triggered when the user hovers over a reference line and presses Shift + D key.

        Removes:
        - Reference lines from image, line, probe, and histology figures
        - Associated scatter point from the fit figure

        Returns
        -------
        None
        """

        if not self.hover_line:
            return

        if self.loaddata.configs is not None:
            if self.hover_config == 'both':
                for config in self.loaddata.configs:
                    items = self.shank_items[self.hover_shank][config]
                    line_idx = np.where(items.lines_features == self.hover_line)[0]
                    if line_idx.size != 0:
                        break
                line_idx = line_idx[0]
                self._delete_reference_line(line_idx, shanks=[self.hover_shank])
                return
            else:
                items = self.shank_items[self.hover_shank][self.hover_config]
        else:
            items = self.shank_items[self.hover_shank]

        # Attempt to find selected line in feature lines
        line_idx = np.where(items.lines_features == self.hover_line)[0]
        if line_idx.size == 0:
            # If not found, try in track lines
            line_idx = np.where(items.lines_tracks == self.hover_line)[0]
            if line_idx.size == 0:
                return  # Couldn't find either

        line_idx = line_idx[0]

        self._delete_reference_line(line_idx, shanks=[self.hover_shank])



    @shank_loop
    def _delete_reference_line(self, items, line_idx,  **kwargs) -> None:

        # Remove line items from plots
        items.fig_img.removeItem(items.lines_features[line_idx][0])
        items.fig_line.removeItem(items.lines_features[line_idx][1])
        items.fig_probe.removeItem(items.lines_features[line_idx][2])
        items.fig_hist.removeItem(items.lines_tracks[line_idx, 0])
        self.fig_fit.removeItem(items.points[line_idx, 0])

        # Remove from tracking arrays
        items.lines_features = np.delete(items.lines_features, line_idx, axis=0)
        items.lines_tracks = np.delete(items.lines_tracks, line_idx, axis=0)
        items.points = np.delete(items.points, line_idx, axis=0)


    def update_feature_reference_line(self, feature_line: pg.InfiniteLine, idx: int, config: str) -> None:
        """
        Callback triggered when a reference line is moved in one of the electrophysiology plots
        (image, line, or probe). This function ensures the line's new position is synchronized
        across the other ephys plots, and updates the corresponding scatter point in the fit plot.

        Parameters
        ----------
        feature_line : pyqtgraph.InfiniteLine
            The line instance that was moved by the user.
        idx: int
            The panel number that the line instance belongs to, used to update the selected_shank

        Returns
        -------
        None
        """

        if idx != self.selected_idx:
            self.shank_combobox.setCurrentIndex(idx)
            self.on_shank_selected(idx)

        if config is None:
            items = self.shank_items[self.selected_shank]
        else:
            items = self.shank_items[self.selected_shank][config]

        idx = np.where(items.lines_features == feature_line)
        line_idx = idx[0][0]
        fig_idx = np.setdiff1d(np.arange(0, 3), idx[1][0])  # Indices of two other plots

        self._update_feature_reference_line(feature_line, line_idx, fig_idx, shanks=[self.selected_shank])

    @shank_loop
    def _update_feature_reference_line(self, items, feature_line, line_idx, fig_idx, **kwargs) -> None:

        items.lines_features[line_idx][fig_idx[0]].setPos(feature_line.value())
        items.lines_features[line_idx][fig_idx[1]].setPos(feature_line.value())

        # Update scatter point on the fit figure
        items.points[line_idx][0].setData(x=[items.lines_features[line_idx][0].pos().y()],
                                          y=[items.lines_tracks[line_idx][0].pos().y()])

    def update_track_reference_line(self, track_line: pg.InfiniteLine, idx: int, config: str) -> None:
        """
        Callback triggered when a reference line in the histology plot is moved.
        This updates the corresponding scatter point in the fit plot.

        Parameters
        ----------
        track_line : pyqtgraph.InfiniteLine
            The line instance that was moved by the user.
        idx: int
            The panel number that the line instance belongs to, used to update the selected_shank

        Returns
        -------
        None
        """

        if idx != self.selected_idx:
            self.shank_combobox.setCurrentIndex(idx)
            self.on_shank_selected(idx)

        if config is None:
            items = self.shank_items[self.selected_shank]
        else:
            items = self.shank_items[self.selected_shank][config]

        line_idx = np.where(items.lines_tracks == track_line)[0][0]

        self._update_track_reference_line(track_line, line_idx, shanks=[self.selected_shank])

    @shank_loop
    def _update_track_reference_line(self, items, track_line, line_idx, **kwargs) -> None:
        items.lines_tracks[line_idx][0].setPos(track_line.value())
        items.points[line_idx][0].setData(x=[items.lines_features[line_idx][0].pos().y()],
                                          y=[items.lines_tracks[line_idx][0].pos().y()])

    @shank_loop
    def align_reference_lines(self, items, **kwargs) -> None:
        """
        Align the positions of all track reference lines and scatter points based on the new positions
        of their corresponding feature reference lines.

        Returns
        -------
        None
        """

        for line_feature, line_track, point in zip(items.lines_features, items.lines_tracks, items.points):
            line_track[0].setPos(line_feature[0].getYPos())
            point[0].setData(x=[line_feature[0].pos().y()], y=[line_feature[0].pos().y()])

    @shank_loop
    def remove_points_from_display(self, items: Union[Dict, Bunch], **kwargs):
        """
        Remove all reference points from the fit plot.

        Parameters
        ----------
        shank : list of str or tuple of str
            Identifiers for the shank(s) whose reference lines and points are to be removed.

        items : dict or Bunch
            A structure containing plot elements for a shank
        Returns
        -------
        None
        """
        for point in items.points:
            self.fig_fit.removeItem(point[0])

    def add_points_to_display(self) -> None:
        """
        Add reference points to the fit plot for the selected shank

        Returns
        -------
        None
        """
        if self.loaddata.configs is None:
            items = self.shank_items[self.selected_shank]
        else:
            config = 'quarter' if self.loaddata.selected_config == 'both' else self.loaddata.selected_config
            items = self.shank_items[self.selected_shank][config]
        for point in items.points:
            self.fig_fit.addItem(point[0])

    @shank_loop
    def remove_reference_lines_from_display(self, items: Union[Dict, Bunch], **kwargs) -> None:
        """
        Remove all reference lines and scatter points from the electrophysiology, histology, and fit plots.

        Parameters
        ----------
        shank : list of str or tuple of str
            Identifiers for the shank(s) whose reference lines and points are to be removed.

        items : dict or Bunch
            A structure containing plot elements for a shank
        Returns
        -------
        None
        """

        for line_feature, line_track, point in zip(items.lines_features, items.lines_tracks, items.points):
            items.fig_img.removeItem(line_feature[0])
            items.fig_line.removeItem(line_feature[1])
            items.fig_probe.removeItem(line_feature[2])
            items.fig_hist.removeItem(line_track[0])
            self.fig_fit.removeItem(point[0])

    @shank_loop
    def add_reference_lines_to_display(self, items: Union[Dict, Bunch], **kwargs) -> None:
        """
        Add previously created reference lines and scatter points to their respective plots.

        Parameters
        ----------
        shank : list of str or tuple of str
            Identifiers for the shank(s) whose reference lines and points are to be removed.

        items : dict or Bunch
            A structure containing the plot elements for a shank
        Returns
        -------
        None
        """
        shank = kwargs.get('shank')

        for line_feature, line_track, point in zip(items.lines_features, items.lines_tracks, items.points):
            items.fig_img.addItem(line_feature[0])
            items.fig_line.addItem(line_feature[1])
            items.fig_probe.addItem(line_feature[2])
            items.fig_hist.addItem(line_track[0])
            if shank == self.selected_shank:
                self.fig_fit.addItem(point[0])




def viewer(probe_id, one=None, histology=False, spike_collection=None, title=None):
    """
    """
    qt.create_app()
    av = MainWindow._get_or_create(probe_id=probe_id, one=one, histology=histology,
                                   spike_collection=spike_collection, title=title)
    av.show()
    return av


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='Offline vs online mode')
    parser.add_argument(
        '-o', '--offline',
        required=False,
        default=False,
        help='Run in offline mode'
    )

    parser.add_argument(
        '-c', '--csv',
        required=False,
        type=str,
        help='Path to the CSV file'
    )

    args = parser.parse_args()

    app = QtWidgets.QApplication([])
    mainapp = MainWindow(offline=args.offline, csv=args.csv)
    mainapp.show()
    app.exec_()
