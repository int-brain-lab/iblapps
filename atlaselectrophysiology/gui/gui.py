import os
import platform
from typing import Union, Optional, List, Callable, Tuple, Dict, Any

from iblutil.util import Bunch

if platform.system() == 'Darwin':
    if platform.release().split('.')[0] >= '20':
        os.environ["QT_MAC_WANTS_LAYER"] = "1"

from PyQt5 import QtWidgets
import pyqtgraph as pg
import pyqtgraph.exporters
import numpy as np

import atlaselectrophysiology.qt_utils.utils as utils
import atlaselectrophysiology.qt_utils.plots as qt_plots
from atlaselectrophysiology.gui.layout import Setup

from atlaselectrophysiology.loaders.probe_loader import ProbeLoaderLocal, ProbeLoaderCSV

from atlaselectrophysiology.plugins.cluster_popup import callback as cluster_callback
from atlaselectrophysiology.plugins.qc_dialog import display as display_qc
from pathlib import Path
from qt_helpers import qt
import matplotlib.pyplot as mpl  # noqa  # This is needed to make qt show properly :/

from functools import wraps
from typing import Any, Callable, Sequence


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
    def wrapper(self, *args, shanks: Sequence = (), **kwargs) -> Any:
        # Use all_shanks if none provided
        shanks = shanks or self.all_shanks
        results = []
        for shank in shanks:
            result = func(self, shank, self.shank_items[shank], *args, **kwargs)
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

    def __init__(self, offline=False, probe_id=None, one=None, histology=True,
                 spike_collection=None, unity=False, loaddata=None):
        super(MainWindow, self).__init__()
        self.lala = True
        self.test = None
        self.fix_reference_colours = True

        self.init_variables()
        self.init_layout(offline=offline)
        self.configure = True

        if loaddata is not None:
            self.loaddata = loaddata
            self.loaddata.get_info(probe_id)
            self.loaddata.get_starting_alignment(0)
            self.data_button_pressed()
            self.offline = False


        if not offline and probe_id is None:
            self.loaddata = ProbeLoaderCSV()
            utils.populate_lists(self.loaddata.get_subjects(), self.subj_list, self.subj_combobox)
            self.offline = False
        # elif not offline and probe_id is not None and loaddata is not None:
        #     self.loaddata = LoadData(probe_id=probe_id, one=one, load_histology=histology,
        #                              spike_collection=spike_collection)
        #     self.current_shank_idx = 0
        #     _, self.histology_exists = self.loaddata.get_info(0)
        #     self.loaddata.get_starting_alignment(0)
        #     self.data_status = False
        #     self.data_button_pressed()
        #     self.offline = False
        elif loaddata is None:
            self.loaddata = ProbeLoaderLocal()
            self.offline = True
            self.histology_exists = True


    def init_variables(self):
        """
        Initialise variables
        """

        # Variables to do with probe dimension
        self.extend_feature = 1

        # Initialise with linear fit scaling as default
        self.lin_fit = True

        # Variables to keep track of reference lines and points added
        self.line_status = True
        self.label_status = True
        self.channel_status = True
        self.hist_bound_status = True

        self.y_scale = 1
        self.x_scale = 1

        self.probe_tip = 0
        self.probe_top = 3840
        self.probe_extra = 100
        self.view_total = [-2000, 6000]
        self.depth = np.arange(self.view_total[0], self.view_total[1], 20)
        # Padding to add to figures to make sure always same size viewbox
        self.pad = 0.05


        # Variables to keep track of popup plots
        self.cluster_popups = []
        self.label_popup = []
        self.popup_status = True
        self.subj_win = None

        self.nearby = None

        # Filter by different types of units
        self.filter_type = 'all'


        self.all_shanks = list()
        self.shank_items = dict()
        self.docks = []
        self.selected_idx = 0
        self.grid = True

    @shank_loop
    def init_shank_items(self, shank: Union[List[str], Tuple[str, ...]], items: Union[Dict, Bunch]):

        items.img_plots = list()
        items.line_plots = list()
        items.probe_plots = list()
        items.img_cbars = list()
        items.probe_cbars = list()
        items.scale_regions = np.empty((0, 1))
        items.slice_lines = list()
        items.slice_items = list()
        items.probe_bounds = list()
        items.hist_regions = dict()

        items.lines_features = np.empty((0, 3))
        items.lines_tracks = np.empty((0, 1))
        items.points = np.empty((0, 1))

        items.probe_tip = 0
        items.probe_top = 3840
        items.probe_extra = 100
        items.view_total = [-2000, 6000]
        items.depth = np.arange(items.view_total[0], items.view_total[1], 20)
        items.pad = 0.05

    @shank_loop
    def clear_shank_items(self, shank: Union[List[str], Tuple[str, ...]], items: Union[Dict, Bunch]):

        utils.remove_items(items.fig_img, items.img_plots)
        utils.remove_items(items.fig_img, items.img_cbars)
        utils.remove_items(items.fig_line, items.line_plots)
        utils.remove_items(items.fig_probe, items.probe_plots)
        utils.remove_items(items.fig_probe, items.probe_cbars)
        items.fig_slice.clear()
        items.fig_hist.clear()
        items.ax_hist.setTicks([])
        items.fig_hist_ref.clear()
        items.ax_hist_ref.setTicks([])
        items.fig_scale.clear()
        items.fit_scatter.setData()
        items.fit_plot.setData()


    @shank_loop
    def set_xaxis_range(self, shank: Union[List[str], Tuple[str, ...]], items: Union[Dict, Bunch], fig, data, label=True):
        """
        Set the x-axis range for a given figure for specified shank or all shanks.

        Parameters
        ----------
        shank : Any
            The current shank (e.g., 0, 1, 2).
        items : dict
            The dictionary or object containing plot items for the shank.
        fig : str
            Key to access the desired figure within items.
        data : Any
            Data to define the axis range.
        label : bool, optional
            Whether to label the x-axis (default is True).
        """
        qt_plots.set_xaxis_range(items[fig], data, label=label)

    @shank_loop
    def set_yaxis_range(self, shank: Union[List[str], Tuple[str, ...]], items: Union[Dict, Bunch], fig):
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
        data['yrange'] = [items.probe_tip - items.probe_extra, items.probe_top + items.probe_extra]
        data['pad'] = items.pad
        return data

    # -------------------------------------------------------------------------------------------------
    # Plotting functions
    # -------------------------------------------------------------------------------------------------
    @shank_loop
    def plot_histology_panels(self, shank: Union[List[str], Tuple[str, ...]], items: Union[Dict, Bunch]):
        items.fig_hist.clear()
        data = self.add_yrange_to_data(items, items.hist_data)
        hist = qt_plots.plot_histology(items.fig_hist, data)
        items.hist_regions['left'] = hist
        items.tip_pos = pg.InfiniteLine(pos=items.probe_tip, angle=0, pen=utils.kpen_dot, movable=True)
        items.top_pos = pg.InfiniteLine(pos=items.probe_top, angle=0, pen=utils.kpen_dot, movable=True)

        offset = 1
        items.tip_pos.setBounds((self.loaddata.probes[shank].loaders['align'].track[0] * 1e6 + offset,
                                 self.loaddata.probes[shank].loaders['align'].track[-1] * 1e6 - (
                                                   items.probe_top + offset)))
        items.top_pos.setBounds(
            (self.loaddata.probes[shank].loaders['align'].track[0] * 1e6 + (items.probe_top + offset),
             self.loaddata.probes[shank].loaders['align'].track[-1] * 1e6 - offset))
        items.tip_pos.sigPositionChanged.connect(self.tip_line_moved)
        items.top_pos.sigPositionChanged.connect(self.top_line_moved)

        items.fig_hist.addItem(items.tip_pos)
        items.fig_hist.addItem(items.top_pos)

        self.selected_region = items.hist_regions['left'][-2]

    @shank_loop
    def plot_histology_ref_panels(self, shank: Union[List[str], Tuple[str, ...]], items: Union[Dict, Bunch]):
        items.fig_hist_ref.clear()
        data = self.add_yrange_to_data(items, items.hist_data_ref)
        hist = qt_plots.plot_histology(items.fig_hist_ref, data, ax='right')
        items.hist_regions['right'] = hist
        items.tip_pos = pg.InfiniteLine(pos=items.probe_tip, angle=0, pen=utils.kpen_dot)
        items.top_pos = pg.InfiniteLine(pos=items.probe_top, angle=0, pen=utils.kpen_dot)

        items.fig_hist_ref.addItem(items.tip_pos)
        items.fig_hist_ref.addItem(items.top_pos)

    @shank_loop
    def plot_scale_factor_panels(self, shank: Union[List[str], Tuple[str, ...]], items: Union[Dict, Bunch]):

        items.scale_data['scale_factor'] = items.scale_data['scale'] - 0.5
        items.fig_scale.clear()
        data = self.add_yrange_to_data(items, items.scale_data)
        scale = qt_plots.plot_scale_factor(items.fig_scale, items.fig_scale_cb, data)
        items.scale_regions = scale

    @shank_loop
    def plot_fit_panels(self, shank: Union[List[str], Tuple[str, ...]], items: Union[Dict, Bunch]):
        data = {}
        data['x'] = self.loaddata.probes[shank].loaders['align'].feature * 1e6
        data['y'] = self.loaddata.probes[shank].loaders['align'].track * 1e6
        data['depth_lin'] = self.loaddata.probes[shank].loaders['align'].align.ephysalign.feature2track_lin(
            items.depth, data['x'], data['y'])
        data['depth'] = items.depth
        qt_plots.plot_fit(items.fit_plot, items.fit_plot_lin, items.fit_scatter, data)

    @shank_loop
    def remove_fit_panels(self, shank: Union[List[str], Tuple[str, ...]], items: Union[Dict, Bunch]):
        if shank != self.selected_shank:
            items.fit_plot.setData()
            items.fit_scatter.setData()
            items.fit_plot_lin.setData()

    def plot_slice_panels(self, plot_type, shanks=()):
        self.imgs = []
        results = self.plot_slice_panel(plot_type, shanks=shanks)
        self.imgs = [img for img, _ in results]
        img, cb = results[0]
        self.channel_status = True
        if cb is not None:
            self.slice_LUT.axis.hide()
            self.slice_LUT.setImageItem(img)
            self.slice_LUT.gradient.setColorMap(cb.map)
            self.slice_LUT.autoHistogramRange()
            self.slice_LUT.sigLevelsChanged.connect(self.update_levels)

            hist_levels = self.slice_LUT.getLevels()
            hist_val, hist_count = img.getHistogram()
            upper_idx = np.where(hist_count > 10)[0][-1]
            upper_val = hist_val[upper_idx]
            if hist_levels[0] != 0:
                self.slice_LUT.setLevels(min=hist_levels[0], max=upper_val)


    @shank_loop
    def plot_slice_panel(self, shank: Union[List[str], Tuple[str, ...]], items: Union[Dict, Bunch], plot_type):

        # TODO can I handle this better wihtout the fig.clear like we do the other ones
        items.fig_slice.clear()
        items.slice_chns = pg.ScatterPlotItem()
        items.slice_lines = []

        data_chns = {}
        data_chns['xyz_channels'] = self.loaddata.probes[shank].loaders['align'].xyz_channels
        data_chns['track_lines'] = self.loaddata.probes[shank].loaders['align'].track_lines
        data_chns['x'] = self.loaddata.probes[shank].loaders['align'].xyz_track[:, 0]
        data_chns['y'] = self.loaddata.probes[shank].loaders['align'].xyz_track[:, 2]

        data = self.loaddata.probes[shank].slice_plots[plot_type]

        img, cbar, traj = qt_plots.plot_slice(items.fig_slice, items.slice_chns, data, data_chns)
        items.traj_line = traj

        return img, cbar


    def update_levels(self):
        levels = self.slice_LUT.getLevels()

        for img in self.imgs:
            img.setLevels(levels)

    @shank_loop
    def plot_channel_panels(self, shank: Union[List[str], Tuple[str, ...]], items: Union[Dict, Bunch]):
        self.channel_status = True
        data = {}
        data['xyz_channels'] = self.loaddata.probes[shank].loaders['align'].xyz_channels
        data['track_lines'] = self.loaddata.probes[shank].loaders['align'].track_lines

        items.slice_lines = utils.remove_items(items.fig_slice, items.slice_lines)
        lines = qt_plots.plot_channels(items.fig_slice, items.slice_chns, data)
        items.slice_lines += lines


    @shank_loop
    def plot_scatter_panels(self, shank: Union[List[str], Tuple[str, ...]], items: Union[Dict, Bunch], plot_type):
        data = self.add_yrange_to_data(items, self.loaddata.probes[shank].scatter_plots[plot_type])
        items.img_plots = utils.remove_items(items.fig_img, items.img_plots)
        items.img_cbars = utils.remove_items(items.fig_img_cb, items.img_cbars)
        scat, cbar = qt_plots.plot_line(items.fig_img, items.fig_img_cb, data)
        items.img_plots.append(scat)
        items.img_cbars.append(cbar)
        if data['cluster']:
            items.cluster_data = data['x']
            items.ephys_plot.sigClicked.connect(lambda plot, points: cluster_callback(self, plot, points))
        items.ephys_plot = scat
        self.y_scale = 1
        self.xrange = data['xrange']

    @shank_loop
    def plot_line_panels(self, shank: Union[List[str], Tuple[str, ...]], items: Union[Dict, Bunch], plot_type):

        data = self.add_yrange_to_data(items, self.loaddata.probes[shank].line_plots[plot_type])
        items.line_plots = utils.remove_items(items.fig_line, items.line_plots)
        lin = qt_plots.plot_line(items.fig_line, data)
        items.line_plots.append(lin)

    @shank_loop
    def plot_probe_panels(self, shank: Union[List[str], Tuple[str, ...]], items: Union[Dict, Bunch], plot_type):

        data = self.add_yrange_to_data(items, self.loaddata.probes[shank].probe_plots[plot_type])
        items.probe_plots = utils.remove_items(items.fig_probe, items.probe_plots)
        items.probe_cbars = utils.remove_items(items.fig_probe_cb, items.probe_cbars)
        items.probe_bounds = utils.remove_items(items.fig_probe, items.probe_bounds)
        prbs, cbar, bnds = qt_plots.plot_probe(items.fig_probe, items.fig_probe_cb, data)
        items.probe_plots += prbs
        items.probe_bounds += bnds
        items.probe_cbars.append(cbar)

    @shank_loop
    def plot_image_panels(self, shank: Union[List[str], Tuple[str, ...]], items: Union[Dict, Bunch], plot_type):

        data = self.add_yrange_to_data(items, self.loaddata.probes[shank].img_plots[plot_type])
        items.img_plots = utils.remove_items(items.fig_img, items.img_plots)
        items.img_cbars = utils.remove_items(items.fig_img_cb, items.img_cbars)
        img, cbar = qt_plots.plot_image(items.fig_img, items.fig_img_cb, data)
        items.img_plots.append(img)
        items.img_cbars.append(cbar)
        items.ephys_plot = img
        # TODO should be just once but oh well
        self.y_scale = data['scale'][1]
        self.x_scale = data['scale'][0]
        self.xrange = data['xrange']


    def update_plots(self, shanks=()) -> None:
        """
        Refresh all plots to reflect the current alignment state.
        """

        self.get_scaled_histology(shanks=shanks)
        self.plot_histology_panels(shanks=shanks)
        self.plot_scale_factor_panels(shanks=shanks)
        self.plot_fit_panels(shanks=shanks)
        self.plot_channel_panels(shanks=shanks)
        self.remove_reference_lines_from_display(shanks=shanks)
        self.add_reference_lines_to_display(shanks=shanks)
        self.align_reference_lines()
        self.set_yaxis_range('fig_hist', shanks=shanks)
        self.update_string()

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
        self.shank = None
        self.sess_list.clear()
        sessions = self.loaddata.get_sessions(idx)
        utils.populate_lists(sessions, self.sess_list, self.sess_combobox)

        self.on_session_selected(0)

    def on_session_selected(self, idx):
        """
        Triggered when session is selected from drop down list options
        :param idx: index of chosen session (item) in drop down list
        :type idx: int
        """
        self.shank = None
        self.shank_list.clear()
        shanks = self.loaddata.get_shanks(idx)
        if len(shanks) > 1:
            utils.populate_lists(shanks, self.shank_list, self.shank_combobox)

        self.on_shank_selected(0)

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

        if self.shank is not None:
            if not self.shank_tabs.grid_layout:
                self.set_current_tab(idx)
            self.shank_button_pressed()

    def on_folder_selected(self):
        """
        Triggered in offline mode when folder button is clicked
        """
        self.shank = None
        self.align_list.clear()
        self.shank_list.clear()
        folder_path = Path(QtWidgets.QFileDialog.getExistingDirectory(None, "Select Folder"))
        self.folder_line.setText(str(folder_path))
        shank_options = self.loaddata.get_shanks(folder_path)
        utils.populate_lists(shank_options, self.shank_list, self.shank_combobox)
        self.on_shank_selected(0)

    def on_alignment_selected(self, idx):
        self.loaddata.get_starting_alignment(idx)
        if self.shank is not None:
            self.align_button_pressed()

    def add_alignment_pressed(self):
        file_path = Path(QtWidgets.QFileDialog.getOpenFileName()[0])
        if file_path.name != 'prev_alignments.json':
            print("Wrong file selected, must be of format prev_alignments.json")
            return
        else:
            self.prev_alignments = self.loaddata.add_extra_alignments(file_path)
            utils.populate_lists(self.prev_alignments, self.align_list, self.align_combobox)
            self.loaddata.get_starting_alignment(0)


    def shank_button_pressed(self):

        self.shank = self.loaddata.get_selected_probe()
        self.shank.loaders['align'].set_init_alignment()
        self.selected_shank = self.loaddata.probe_label
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
        items = self.shank_items[self.selected_shank]

        items.lines_features = np.empty((0, 3))
        items.lines_tracks = np.empty((0, 1))
        items.points = np.empty((0, 1))

        self.set_init_reference_lines(shanks=[self.selected_shank])
        self.update_plots(shanks=[self.selected_shank])

    def data_button_pressed(self):
        """
        Triggered when Get Data button pressed, uses subject and session info to find eid and
        downloads and computes data needed for GUI display
        """

        self.fig_fit.clear()
        # Remove previous shanks displays
        self.clear_tabs()
        # Get the new list of shanks
        self.all_shanks = self.loaddata.probes
        # Make new set of shank displays
        self.init_tabs()
        # Add additional variables
        self.init_shank_items()

        # Load data of new selection
        self.loaddata.load_data()
        self.shank = self.loaddata.get_selected_probe()
        self.selected_shank = self.loaddata.probe_label
        self.selected_idx = self.loaddata.shank_idx
        self.set_probe_lims(np.min([0, self.shank.plotdata.chn_min]), self.shank.plotdata.chn_max)

        # Populate the menu bar with the plot options
        self.populate_menu_bar()

        # Initialise checked plots
        self.img_init.setChecked(True)
        self.line_init.setChecked(True)
        self.probe_init.setChecked(True)
        self.unit_init.setChecked(True)
        self.slice_init.setChecked(True)

        # Initialise ephys plots
        self.plot_image_panels(self.img_init.text())
        self.plot_probe_panels(self.probe_init.text())
        self.plot_line_panels(self.line_init.text())
        self.plot_slice_panels(self.slice_init.text())

        # Initialise histology plots
        self.get_scaled_histology()
        self.plot_histology_ref_panels()
        self.plot_histology_panels()
        self.plot_scale_factor_panels()
        self.label_status = False
        self.toggle_labels()
        self.update_string()

        # Initialise reference lines
        self.set_init_reference_lines()

        # Add refernce points for selected shank
        self.remove_points_from_display()
        self.add_points_to_display()

        # Plot fits
        self.plot_fit_panels()

        self.select_background()
        self.init_display()


    def filter_unit_pressed(self, filter_type):
        #TODO FIX
        self.filter_type = filter_type
        self.shank.plotdata.filter_units(self.filter_type)
        self.shank.filter_plots()

        self.img_init.setChecked(True)
        self.line_init.setChecked(True)
        self.probe_init.setChecked(True)
        self.plot_image(self.shank.img_plots['Firing Rate'])
        self.plot_probe(self.shank.probe_plots['rms AP'])
        self.plot_line(self.shank.line_plots['Firing Rate'])


    # -------------------------------------------------------------------------------------------------
    # Upload/ save data
    # -------------------------------------------------------------------------------------------------

    def complete_button_pressed(self):
        """
        Triggered when complete button or Shift+F key pressed. Uploads final channel locations to
        Alyx
        """
        # If no histology we can't upload alignment
        if not self.histology_exists:
            return

        if hasattr(self.shank.loaders['align'], 'get_qc_string'):
            display_qc(self)

        upload = QtWidgets.QMessageBox.question(self, '', "Upload alignment?",
                                                QtWidgets.QMessageBox.Yes |
                                                QtWidgets.QMessageBox.No)

        if upload == QtWidgets.QMessageBox.Yes:

            info = self.shank.upload_data()
            self.prev_alignments = self.shank.get_previous_alignments()
            utils.populate_lists(self.prev_alignments, self.align_list, self.align_combobox)
            self.shank.get_starting_alignment(0)

            QtWidgets.QMessageBox.information(self, 'Status', info)
        else:
            QtWidgets.QMessageBox.information(self, 'Status', "Channels not saved")


    # -------------------------------------------------------------------------------------------------
    # Fitting functions
    # -------------------------------------------------------------------------------------------------
    def offset_hist_data(self, val: Optional[float] = None) -> None:
        """
        Apply an offset to brain regions based on probe tip position.

        Parameters
        ----------
        val : float, optional
            Offset value in meters. If None, uses current value of  self.tip_pos
        """
        # If no histology we can't do alignment
        if not self.histology_exists:
            return

        val = val or self.tip_pos.value() / 1e6
        self.shank.loaders['align'].offset_hist_data(val)

    def scale_hist_data(self) -> None:
        """
        Scale brain regions along the probe track based on reference lines.
        """
        # If no histology we can't do alignment
        if not self.histology_exists:
            return


        items = self.shank_items[self.selected_shank]
        line_track = np.array([line[0].pos().y() for line in items.lines_tracks]) / 1e6
        line_feature = np.array([line[0].pos().y() for line in items.lines_features]) / 1e6

        self.shank.loaders['align'].scale_hist_data(line_track, line_feature,
                                         extend_feature=self.extend_feature, lin_fit=self.lin_fit)

    @shank_loop
    def get_scaled_histology(self, shank: Union[List[str], Tuple[str, ...]], items: Union[Dict, Bunch]) -> None:
        """
        Retrieve scaled histological data after alignment operations.
        """

        items.hist_data, items.hist_data_ref, items.scale_data = self.loaddata.probes[shank].loaders['align'].get_scaled_histology()


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

        # If no histology we can't plot histology
        if not self.histology_exists:
            return

        fit_function(**kwargs)

        self.update_plots(shanks=[self.selected_shank])


    def offset_button_pressed(self) -> None:
        """
        Called when the offset button or 'O' key is pressed.
        Applies offset based on location of the probe tip line and refreshes plots.
        """
        self.apply_fit(self.offset_hist_data)


    def movedown_button_pressed(self) -> None:
        """
        Called when Shift+Down is pressed. Offsets probe tip 100µm down.
        """
        self.apply_fit(self.offset_hist_data, val=-100/1e6)


    def moveup_button_pressed(self) -> None:
        """
        Called when Shift+Up is pressed. Offsets probe tip 100µm up.
        """
        self.apply_fit(self.offset_hist_data, val=100/1e6)


    def fit_button_pressed(self) -> None:
        """
        Called when the fit button or Enter is pressed.
        Scales regions using reference lines and refreshes plots.
        """
        self.apply_fit(self.scale_hist_data)


    def next_button_pressed(self) -> None:
        """
        Called when right key pressed.
        Updates all plots and indices with next alignment.
        Ensures user cannot go past latest move
        """
        # If no histology we can't plot histology
        if not self.histology_exists:
            return

        if self.shank.loaders['align'].next_idx():
            self.update_plots(shanks=[self.selected_shank])


    def prev_button_pressed(self) -> None:
        """
        Called when left key pressed.
        Updates all plots and indices with previous alignment.
        """

        # If no histology we can't plot histology
        if not self.histology_exists:
            return

        if self.shank.loaders['align'].prev_idx():
            self.update_plots(shanks=[self.selected_shank])


    def reset_button_pressed(self) -> None:
        """
        Called when Reset button or Shift+R is pressed.
        Resets feature and track alignment to original alignment and updates plots.
        """
        # If no histology we can't plot histology
        if not self.histology_exists:
            return

        items= self.shank_items[self.selected_shank]
        self.remove_reference_lines_from_display(shanks=[self.selected_shank])
        items.lines_features = np.empty((0, 3))
        items.lines_tracks = np.empty((0, 1))
        items.points = np.empty((0, 1))

        self.shank.loaders['align'].reset_features_and_tracks()

        self.set_init_reference_lines(shanks=[self.selected_shank])

        self.update_plots(shanks=[self.selected_shank])


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
        self.idx_string.setText(f"Current Index = {self.shank.loaders['align'].current_idx}")
        self.tot_idx_string.setText(f"Total Index = {self.shank.loaders['align'].total_idx}")

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

        # If no histology no point adding lines
        if not self.histology_exists:
            return

        if event.double():
            if idx != self.selected_idx:
                self.shank_combobox.setCurrentIndex(idx)
                self.on_shank_selected(idx)

            items = self.shank_items[self.selected_shank]
            pos = items.ephys_plot.mapFromScene(event.scenePos())
            self.create_reference_line(pos.y() * self.y_scale, items)

    def on_mouse_hover(self, hover_items: List[pg.GraphicsObject]) -> None:
        """
        Handles mouse hover events over pyqtgraph plot items.

        Identifies reference lines or linear regions the mouse is hovering over
        to allow interactive operations like deletion or displaying additional info.

        Parameters
        ----------
        items : list of pyqtgraph.GraphicsObject
            List of items under the mouse cursor.
        """

        if len(hover_items) > 1:
            self.selected_line = []
            item0, item1 = hover_items[0], hover_items[1]
            # TODO think about adding names to keep track of things
            if isinstance(item0, pg.InfiniteLine):
                self.selected_line = item0
            elif isinstance(item1, pg.LinearRegionItem):

                # Check if we are on the fig_scale plot
                items = self._get_figure(item0, 'fig_scale')
                if items is not None:
                    idx = np.where(items.scale_regions == item1)[0][0]
                    items.fig_scale_ax.setLabel('Scale Factor = ' + str(np.around(items.scale_data['scale_factor'][idx], 2)))
                    return
                # Check if we are on the histology plot
                items = self._get_figure(item0, 'fig_hist')
                if items is not None:
                    self.selected_region = item1
                    return
                items = self._get_figure(item0, 'fig_hist_ref')
                if items is not None:
                    self.selected_region = item1
                    return

    def _get_figure(self, item, fig):
        return next((items for _, items in self.shank_items.items() if item == items[fig]), None)


    # -------------------------------------------------------------------------------------------------
    # Display options
    # -------------------------------------------------------------------------------------------------

    @shank_loop
    def select_background(self, shank: Union[List[str], Tuple[str, ...]], items: Union[Dict, Bunch]):
        if shank == self.selected_shank:
            items.header.setStyleSheet(utils.selected_tab_style)
        else:
            items.header.setStyleSheet(utils.deselected_tab_style)


    def toggle_labels(self, shanks=()) -> None:
        """
        Toggle visibility of brain region labels on histology plots.

        Triggered by pressing Shift+A. Updates the pens for axis items on both the main
        and reference histology plots to show or hide Allen atlas region labels.
        """
        self.label_status = not self.label_status
        self.toggle_label(shanks=shanks)

    @shank_loop
    def toggle_label(self, shank: Union[List[str], Tuple[str, ...]], items: Union[Dict, Bunch]):
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
        """
        self.line_status = not self.line_status
        if not self.line_status:
            self.remove_reference_lines_from_display()
        else:
            self.add_reference_lines_to_display()

    def toggle_channels(self, shanks=()) -> None:
        """
        Toggle visibility of channels and trajectory lines on the histology slice image.

        Triggered by pressing Shift+C. Adds or removes the visual indicators of the probe
        trajectory and channels on the slice view.
        """

        self.channel_status = not self.channel_status
        self.toggle_channel(shanks=shanks)

    @shank_loop
    def toggle_channel(self, shank: Union[List[str], Tuple[str, ...]], items: Union[Dict, Bunch]):
        toggleItem = items.fig_slice.addItem if self.channel_status else items.fig_slice.removeItem

        toggleItem(items.traj_line)
        toggleItem(items.slice_chns)
        for line in items.slice_lines:
            toggleItem(line)

    def toggle_histology(self) -> None:
        """
        Toggle between displaying histology boundaries or block regions

        Triggered by pressing Shift+N. If nearby boundaries are not yet computed, this will
        call the boundary computation method
        """

        self.hist_bound_status = not self.hist_bound_status

        if not self.hist_bound_status:
            if self.hist_nearby_x is None:
                #TODO fix
                self.compute_nearby_boundaries()
            self.plot_histology_nearby(self.fig_hist_ref)
        else:
            self.plot_histology(self.fig_hist_ref, self.hist_data_ref, ax='right')

    def toggle_histology_map(self) -> None:
        """
        Toggle between different histology mapping sources ('Allen' and 'FP').

        Triggered by pressing Shift+M. Re-scales and re-renders histology data using the new mapping.
        """

        self.hist_mapping = 'FP' if self.hist_mapping == 'Allen' else 'Allen'

        self.get_scaled_histology()
        self.plot_histology(self.fig_hist, self.hist_data)
        self.plot_histology(self.fig_hist_ref, self.hist_data_ref, ax='right')

        self.remove_reference_lines_from_display()
        self.add_reference_lines_to_display()

    def reset_axis_button_pressed(self) -> None:
        """
        Reset zoomable plot axis to default

        Triggered by pressing Shift+A.
        """
        self.set_yaxis_range('fig_hist')
        self.set_yaxis_range('fig_hist_ref')
        self.set_yaxis_range('fig_img')
        self.set_xaxis_range('fig_img', {'xrange': [self.xrange[0], self.xrange[1]]}, label=False)

    def layout_changed(self):
        # TODO figure out
        self.shank_combobox.setCurrentIndex(self.selected_idx)
        self.on_shank_selected(self.selected_idx)

    def shank_tab_changed(self, index, from_hist=False):
        self.shank_combobox.setCurrentIndex(index)
        self.on_shank_selected(index)
        if not from_hist:
            self.hist_tabs.tab_widget.setCurrentIndex(index)
        self.remove_fit_panels()
        self.plot_fit_panels(shanks=[self.selected_shank])

    def hist_tab_changed(self, index):
        self.shank_tabs.tab_widget.setCurrentIndex(index)

    def toggle_layout(self):
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

    def set_current_tab(self, idx):
        self.shank_tabs.tab_widget.blockSignals(True)
        self.shank_tabs.tab_widget.setCurrentIndex(idx)
        self.hist_tabs.tab_widget.setCurrentIndex(idx)
        self.shank_tabs.tab_widget.blockSignals(False)

    def on_fig_size_changed(self):
        self.lin_fit_option.move(70, 10)
    # -------------------------------------------------------------------------------------------------
    # Probe top and tip lines
    # -------------------------------------------------------------------------------------------------

    @shank_loop
    def set_probe_lims(self, shank: Union[List[str], Tuple[str, ...]], items: Union[Dict, Bunch], min_val: Union[int, float], max_val: Union[int, float]) -> None:
        """
        Set the limits for the probe tip and probe top, and update the associated lines accordingly.

        Parameters
        ----------
        shank : list of str or tuple of str
            Identifiers for the shank(s) whose reference lines and points are to be removed.

        items : dict or Bunch
            A structure containing plot elements for a shank
        min_val : int or float
            The new minimum limit representing the probe tip position.
        max_val : int or float
            The new maximum limit representing the probe top position.
        Returns
        -------
        None
        """

        items.probe_tip = min_val
        items.probe_top = max_val

        for top_line in items.probe_top_lines:
            top_line.setY(items.probe_top)
        for tip_line in items.probe_tip_lines:
            tip_line.setY(items.probe_tip)

    def tip_line_moved(self) -> None:
        """
        Callback triggered when the probe tip line (dotted line) on the histology figure is moved.

        This function updates the probe top line's vertical position to maintain a fixed
        vertical distance between tip and top (i.e., the probe length).
        """
        items = self.shank_items[self.selected_shank]
        items.top_pos.setPos(items.tip_pos.value() + items.probe_top)

    def top_line_moved(self) -> None:
        """
        Callback triggered when the probe top line (dotted line) on the histology figure is moved.

        This function updates the probe tip line's vertical position to maintain a fixed
        vertical distance between tip and top (i.e., the probe length).
        """
        items = self.shank_items[self.selected_shank]
        items.tip_pos.setPos(items.top_pos.value() - items.probe_top)


    # -------------------------------------------------------------------------------------------------
    # Reference lines
    # -------------------------------------------------------------------------------------------------

    @shank_loop
    def set_init_reference_lines(self, shank: Union[List[str], Tuple[str, ...]], items: Union[Dict, Bunch]) -> None:
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

        self.loaddata.probes[shank].loaders['align'].set_init_alignment()
        feature_prev = self.loaddata.probes[shank].loaders['align'].feature_prev
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
        line_track.sigPositionChanged.connect(lambda track, i=items.idx: self.update_track_reference_line(track, i))
        line_track.setZValue(100)
        items.fig_hist.addItem(line_track)

        # Reference lines on electrophysiology figures (feature)
        line_features = []
        for fig in [items.fig_img, items.fig_line, items.fig_probe]:
            line_feature = pg.InfiniteLine(pos=pos, angle=0, pen=pen, movable=True)
            line_feature.setZValue(100)
            line_feature.sigPositionChanged.connect(lambda feature, i=items.idx: self.update_feature_reference_line(feature, i))
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
        if not self.selected_line:
            return

        items = self.shank_items[self.selected_shank]
        # Attempt to find selected line in feature lines
        line_idx = np.where(items.lines_features == self.selected_line)[0]
        if line_idx.size == 0:
            # If not found, try in track lines
            line_idx = np.where(items.lines_tracks == self.selected_line)[0]
            if line_idx.size == 0:
                return  # Couldn't find either

        line_idx = line_idx[0]

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

    def update_feature_reference_line(self, feature_line: pg.InfiniteLine, idx: int) -> None:
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

        items = self.shank_items[self.selected_shank]
        idx = np.where(items.lines_features == feature_line)

        line_idx = idx[0][0]
        fig_idx = np.setdiff1d(np.arange(0, 3), idx[1][0]) #  Indices of two other plots

        # Update the other two lines to the new y-position
        items.lines_features[line_idx][fig_idx[0]].setPos(feature_line.value())
        items.lines_features[line_idx][fig_idx[1]].setPos(feature_line.value())

        # Update scatter point on the fit figure
        items.points[line_idx][0].setData(x=[items.lines_features[line_idx][0].pos().y()],
                                         y=[items.lines_tracks[line_idx][0].pos().y()])

    def update_track_reference_line(self, track_line: pg.InfiniteLine, idx: int) -> None:
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

        items = self.shank_items[self.selected_shank]
        line_idx = np.where(items.lines_tracks == track_line)[0][0]

        items.points[line_idx][0].setData(x=[items.lines_features[line_idx][0].pos().y()],
                                         y=[items.lines_tracks[line_idx][0].pos().y()])

    def align_reference_lines(self) -> None:
        """
        Align the positions of all track reference lines and scatter points based on the new positions
        of their corresponding feature reference lines.

        Returns
        -------
        None
        """

        items = self.shank_items[self.selected_shank]
        for line_feature, line_track, point in zip(items.lines_features, items.lines_tracks, items.points):
            line_track[0].setPos(line_feature[0].getYPos())
            point[0].setData(x=[line_feature[0].pos().y()], y=[line_feature[0].pos().y()])

    @shank_loop
    def remove_points_from_display(self, shank: Union[List[str], Tuple[str, ...]], items: Union[Dict, Bunch]):
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
        items = self.shank_items[self.selected_shank]
        for point in items.points:
            self.fig_fit.addItem(point[0])

    @shank_loop
    def remove_reference_lines_from_display(self, shank: Union[List[str], Tuple[str, ...]], items: Union[Dict, Bunch]) -> None:
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
    def add_reference_lines_to_display(self, shank: Union[List[str], Tuple[str, ...]], items: Union[Dict, Bunch]) -> None:
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


def launch_offline():

    app_off = QtWidgets.QApplication([])
    mainapp_off = MainWindow(offline=True)
    mainapp_off.show()
    app_off.exec_()


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='Offline vs online mode')
    parser.add_argument('-o', '--offline', default=False, required=False, help='Offline mode')
    parser.add_argument('-i', '--insertion', default=None, required=False, help='Insertion mode')
    parser.add_argument('-u', '--unity', default=False, required=False, help='Unity mode')
    args = parser.parse_args()

    app = QtWidgets.QApplication([])
    mainapp = MainWindow(offline=args.offline, probe_id=args.insertion, unity=args.unity)
    # mainapp = MainWindow(offline=True)
    mainapp.show()
    app.exec_()
