import os
import platform
from typing import Any, Union, Optional, List, Dict, Callable

if platform.system() == 'Darwin':
    if platform.release().split('.')[0] >= '20':
        os.environ["QT_MAC_WANTS_LAYER"] = "1"

from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
import pyqtgraph.exporters
import numpy as np

import atlaselectrophysiology.qt_utils.utils as utils
import atlaselectrophysiology.qt_utils.plots as qt_plots
from atlaselectrophysiology.qt_utils.layouts import Setup

from atlaselectrophysiology.load_data import LoadData
from atlaselectrophysiology.loaders.probe_loader import ProbeLoaderONE, ProbeLoaderLocal, ProbeLoaderCSV
import atlaselectrophysiology.qt_utils.ColorBar as cb
import atlaselectrophysiology.ephys_gui_setup as ephys_gui

from atlaselectrophysiology.plugins.cluster_popup import callback as cluster_callback
from atlaselectrophysiology.plugins.qc_dialog import display as display_qc
from pathlib import Path
from qt_helpers import qt
import matplotlib.pyplot as mpl  # noqa  # This is needed to make qt show properly :/


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
        if unity:
            from atlaselectrophysiology.plugins.unity_data import UnityData
            self.unitydata = UnityData()
            self.unity = True
        else:
            self.unity = False

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
        #     self.feature_prev, self.track_prev = self.loaddata.get_starting_alignment(0)
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
        # Line styles and fonts
        self.kpen_dot = pg.mkPen(color='k', style=QtCore.Qt.DotLine, width=2)
        self.rpen_dot = pg.mkPen(color='r', style=QtCore.Qt.DotLine, width=2)
        self.kpen_solid = pg.mkPen(color='k', style=QtCore.Qt.SolidLine, width=2)
        self.bpen_solid = pg.mkPen(color='b', style=QtCore.Qt.SolidLine, width=3)
        self.bar_colour = QtGui.QColor(160, 160, 160)

        # Padding to add to figures to make sure always same size viewbox
        self.pad = 0.05

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
        self.pad = 0.05


        # Variables to keep track of popup plots
        self.cluster_popups = []
        self.label_popup = []
        self.popup_status = True
        self.subj_win = None


        self.hist_nearby_x = None
        self.hist_nearby_y = None
        self.hist_nearby_col = None
        self.hist_nearby_parent_x = None
        self.hist_nearby_parent_y = None
        self.hist_nearby_parent_col = None
        self.hist_mapping = 'Allen'


        self.nearby = None

        # keep track of unity display
        self.unity_plot = None
        self.unity_region_status = True
        self.point_size = 0.05

        # Filter by different types of units
        self.filter_type = 'all'


        self.all_shanks = list()
        self.dock_items = dict()
        self.docks = []
        self.selected_idx = 0
        self.grid = True

    def init_shank_items(self, shank):

        shank_items = self.dock_items[shank]
        shank_items.img_plots = []
        shank_items.line_plots = []
        shank_items.probe_plots = []
        shank_items.img_cbars = []
        shank_items.probe_cbars = []
        shank_items.scale_regions = np.empty((0, 1))
        shank_items.slice_lines = []
        shank_items.slice_shank_items = []
        shank_items.probe_bounds = []
        shank_items.hist_regions = dict()

        shank_items.lines_features = np.empty((0, 3))
        shank_items.lines_tracks = np.empty((0, 1))
        shank_items.points = np.empty((0, 1))

        shank_items.probe_tip = 0
        shank_items.probe_top = 3840
        shank_items.probe_extra = 100
        shank_items.view_total = [-2000, 6000]
        shank_items.depth = np.arange(shank_items.view_total[0], shank_items.view_total[1], 20)
        shank_items.pad = 0.05


    # TODO move to setup
    def set_view(
            self,
            view: int,
            configure: bool = False
    ) -> None:
        """
        Update the layout and visual configuration of figure panels based on the selected view mode.

        Parameters
        ----------
        view : int
            The layout mode to use. Supported modes are:
                1 - img | line | probe
                2 - img | probe | line
                3 - probe | line | img
        configure : bool
            If True, update stored figure width and height dimensions before applying the layout.
        """

        if configure:
            self.fig_ax_width = self.fig_data_ax.width()
            self.fig_img_width = self.fig_img.width() - self.fig_ax_width
            self.fig_line_width = self.fig_line.width()
            self.fig_probe_width = self.fig_probe.width()
            self.slice_width = self.fig_slice.width()
            self.slice_height = self.fig_slice.height()
            self.slice_rect = self.fig_slice.viewRect()

        # Remove all existing layout shank_items to start fresh
        for item in [self.fig_img_cb, self.fig_probe_cb, self.fig_img, self.fig_line, self.fig_probe]:
            self.fig_data_layout.removeItem(item)

        # Define configurations for each view mode
        layout_configs = {
            1: {
                'shank_items': [
                    (self.fig_img_cb, 0, 0),
                    (self.fig_probe_cb, 0, 1, 1, 2),
                    (self.fig_img, 1, 0),
                    (self.fig_line, 1, 1),
                    (self.fig_probe, 1, 2),
                ],
                'col_stretch': [(0, 6), (1, 2), (2, 1)],
                'row_stretch': [(0, 1), (1, 10)],
                'axis_target': self.fig_img,
                'sizes': lambda: (
                    self.fig_img.setPreferredWidth(self.fig_img_width + self.fig_ax_width),
                    self.fig_line.setPreferredWidth(self.fig_line_width),
                    self.fig_probe.setFixedWidth(self.fig_probe_width)
                )
            },
            2: {
                'shank_items': [
                    (self.fig_img_cb, 0, 0),
                    (self.fig_probe_cb, 0, 1, 1, 2),
                    (self.fig_img, 1, 0),
                    (self.fig_probe, 1, 1),
                    (self.fig_line, 1, 2),
                ],
                'col_stretch': [(0, 6), (1, 1), (2, 2)],
                'row_stretch': [(0, 1), (1, 10)],
                'axis_target': self.fig_img,
                'sizes': lambda: (
                    self.fig_img.setPreferredWidth(self.fig_img_width + self.fig_ax_width),
                    self.fig_line.setPreferredWidth(self.fig_line_width),
                    self.fig_probe.setFixedWidth(self.fig_probe_width)
                )
            },
            3: {
                'shank_items': [
                    (self.fig_probe_cb, 0, 0, 1, 2),
                    (self.fig_img_cb, 0, 2),
                    (self.fig_probe, 1, 0),
                    (self.fig_line, 1, 1),
                    (self.fig_img, 1, 2),
                ],
                'col_stretch': [(0, 1), (1, 2), (2, 6)],
                'row_stretch': [(0, 1), (1, 10)],
                'axis_target': self.fig_probe,
                'sizes': lambda: (
                    self.fig_probe.setFixedWidth(self.fig_probe_width + self.fig_ax_width),
                    self.fig_img.setPreferredWidth(self.fig_img_width),
                    self.fig_line.setPreferredWidth(self.fig_line_width)
                )
            }
        }

        # Validate view and retrieve layout config
        config = layout_configs.get(view)
        if not config:
            raise ValueError(f"Unknown view mode: {view}")

        # Add layout shank_items
        for item_args in config['shank_items']:
            self.fig_data_layout.addItem(*item_args)

        # Apply column and row stretch factors
        for col, factor in config['col_stretch']:
            self.fig_data_layout.layout.setColumnStretchFactor(col, factor)
        for row, factor in config['row_stretch']:
            self.fig_data_layout.layout.setRowStretchFactor(row, factor)

        # Configure axes: only one figure shows the axis label
        for fig in [self.fig_img, self.fig_line, self.fig_probe]:
            if fig == config['axis_target']:
                utils.set_axis(fig, 'left', label='Distance from probe tip (um)')
            else:
                utils.set_axis(fig, 'left', show=False)

        # Apply size adjustments specific to view
        config['sizes']()

        # Force updates and axis correction
        for fig in [self.fig_img, self.fig_line, self.fig_probe]:
            fig.update()
        self.fig_img.setXRange(min=self.xrange[0] - 10, max=self.xrange[1] + 10, padding=0)
        self.reset_axis_button_pressed()


    # TODO move to setup
    @staticmethod
    def toggle_plots(
            options_group: QtWidgets.QActionGroup
    ) -> None:
        """
        Cycle through image, line, probe and slice plots using keyboard shortcuts (Alt+1, Alt+2, Alt+3, Alt+4)
        Parameters
        ----------
        options_group : QActionGroup
            The group of QAction shank_items representing plots to toggle through
        """

        current_act = options_group.checkedAction()
        actions = options_group.actions()
        current_idx = next(i for i, act in enumerate(actions) if act == current_act)
        next_idx = np.mod(current_idx + 1, len(actions))
        actions[next_idx].setChecked(True)
        actions[next_idx].trigger()

    def set_xaxis_range(self, fig, data, shanks=(), label=True):
        shanks = shanks or self.all_shanks
        for shank in shanks:
            shank_items = self.dock_items[shank]
            qt_plots.set_xaxis_range(shank_items[fig], data, label=label)

    def set_yaxis_range(self, fig, shanks=()):
        shanks = shanks or self.all_shanks
        for shank in shanks:
            shank_items = self.dock_items[shank]
            qt_plots.set_yaxis_range(shank_items[fig], shank_items)

    # -------------------------------------------------------------------------------------------------
    # Plotting functions
    # -------------------------------------------------------------------------------------------------
    def plot_histology_panels(self, shanks=()):
        shanks = shanks or self.all_shanks
        for shank in shanks:
            shank_items = self.dock_items[shank]
            qt_plots.plot_histology(self.dock_items[shank].fig_hist, self.dock_items[shank].hist_data,
                                shank_items)
            # TODO this is ugly, also better handling of tip_pos and top_pos for the fig_hist vs fig_hist_ref
            shank_items.tip_pos = pg.InfiniteLine(pos=shank_items.probe_tip, angle=0, pen=utils.kpen_dot,
                                                  movable=True)
            shank_items.top_pos = pg.InfiniteLine(pos=shank_items.probe_top, angle=0, pen=utils.kpen_dot,
                                                  movable=True)

            offset = 1
            shank_items.tip_pos.setBounds((self.loaddata.probes[shank].loaders['align'].track[0] * 1e6 + offset,
                                           self.loaddata.probes[shank].loaders['align'].track[-1] * 1e6 - (
                                                       shank_items.probe_top + offset)))
            shank_items.top_pos.setBounds(
                (self.loaddata.probes[shank].loaders['align'].track[0] * 1e6 + (shank_items.probe_top + offset),
                 self.loaddata.probes[shank].loaders['align'].track[-1] * 1e6 - offset))
            shank_items.tip_pos.sigPositionChanged.connect(self.tip_line_moved)
            shank_items.top_pos.sigPositionChanged.connect(self.top_line_moved)

            self.dock_items[shank].fig_hist.addItem(shank_items.tip_pos)
            self.dock_items[shank].fig_hist.addItem(shank_items.top_pos)


            self.selected_region = shank_items.hist_regions['left'][-2]



    def plot_histology_ref_panels(self, shanks=()):
        shanks = shanks or self.all_shanks
        for shank in shanks:
            shank_items = self.dock_items[shank]
            qt_plots.plot_histology(self.dock_items[shank].fig_hist_ref,  self.dock_items[shank].hist_data_ref,
                                shank_items, ax='right')
            shank_items.tip_pos = pg.InfiniteLine(pos=shank_items.probe_tip, angle=0, pen=utils.kpen_dot)
            shank_items.top_pos = pg.InfiniteLine(pos=shank_items.probe_top, angle=0, pen=utils.kpen_dot)

            self.dock_items[shank].fig_hist_ref.addItem(shank_items.tip_pos)
            self.dock_items[shank].fig_hist_ref.addItem(shank_items.top_pos)


    def plot_scale_factor_panels(self, shanks=()):
        shanks = shanks or self.all_shanks
        for shank in shanks:
            qt_plots.plot_scale_factor(self.dock_items[shank])

    def plot_fit_panels(self, shanks=()):
        shanks = shanks or self.all_shanks
        for shank in shanks:
            self.plot_fit(self.dock_items[shank], self.loaddata.probes[shank])

    def remove_fit_panels(self, shanks=()):
        shanks = shanks or self.all_shanks
        for shank in shanks:
            shank_items = self.dock_items[shank]
            if shank != self.selected_shank:
                shank_items.fit_plot.setData()
                shank_items.fit_scatter.setData()
                shank_items.fit_plot_lin.setData()

    def plot_fit(self, shank_items, shank_data) -> None:
        """
        Plot the scale factor and offset applied to channels along the depth of the probe track,
        relative to the original positions of the channels.
        """

        # If no histology we can't do alignment
        if not self.histology_exists:
            return

        x = shank_data.loaders['align'].feature * 1e6
        y = shank_data.loaders['align'].track * 1e6

        if len(x) > 2:
            shank_items.fit_plot.setData(x=x, y=y)
            shank_items.fit_scatter.setData(x=x,y=y)

            depth_lin = shank_data.loaders['align'].align.ephysalign.feature2track_lin(self.depth, x, y)

            if np.any(depth_lin):
                shank_items.fit_plot_lin.setData(x=self.depth, y=depth_lin)
            else:
                shank_items.fit_plot_lin.setData()
        else:
            shank_items.fit_plot.setData()
            shank_items.fit_scatter.setData()
            shank_items.fit_plot_lin.setData()

    def plot_slice_panels(self, plot_type, shanks=()):
        shanks = shanks or self.all_shanks
        self.imgs = []
        for shank in shanks:
            img, cb = qt_plots.plot_slice(self.dock_items[shank], self.loaddata.probes[shank],
                            self.loaddata.probes[shank].slice_plots[plot_type])
            self.imgs.append(img)
        self.channel_status=True

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

    def update_levels(self):
        levels = self.slice_LUT.getLevels()
        #lut = self.slice_LUT.getLookupTable()

        for img in self.imgs:
            img.setLevels(levels)
            #img.setLookupTable(lut)


    def plot_channel_panels(self, shanks=()):
        self.channel_status=True
        shanks = shanks or self.all_shanks
        for shank in shanks:
            qt_plots.plot_channels(self.dock_items[shank], self.loaddata.probes[shank])


    def plot_scatter_panels(self, plot_type, shanks=()):
        shanks = shanks or self.all_shanks
        for shank in shanks:
            shank_items = self.dock_items[shank]
            data = self.loaddata.probes[shank].scatter_plots[plot_type]
            qt_plots.plot_scatter(shank_items, data)
            if data['cluster']:
                shank_items.cluster_data = data['x']
                shank_items.ephys_plot.sigClicked.connect(lambda plot, points: cluster_callback(self, plot, points))

        self.y_scale = 1
        self.xrange = data['xrange']


    def plot_line_panels(self, plot_type, shanks=()):
        shanks = shanks or self.all_shanks
        for shank in shanks:
            qt_plots.plot_line(self.dock_items[shank], self.loaddata.probes[shank].line_plots[plot_type])


    def plot_probe_panels(self, plot_type, shanks=()):
        shanks = shanks or self.all_shanks
        for shank in shanks:
            qt_plots.plot_probe(self.dock_items[shank], self.loaddata.probes[shank].probe_plots[plot_type])

    def plot_image_panels(self, plot_type, shanks=()):
        shanks = shanks or self.all_shanks
        for shank in shanks:
            data = self.loaddata.probes[shank].img_plots[plot_type]
            qt_plots.plot_image(self.dock_items[shank], data)
        self.y_scale = data['scale'][1]
        self.x_scale = data['scale'][0]
        self.xrange = data['xrange']



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
        self.shank_list.clear()
        self.align_list.clear()
        sessions = self.loaddata.get_sessions(idx)
        utils.populate_lists(sessions, self.sess_list, self.sess_combobox)
        shanks = self.loaddata.get_shanks(0)
        if len(shanks) > 1:
            utils.populate_lists(shanks, self.shank_list, self.shank_combobox)

        self.loaddata.get_info(0)
        self.prev_alignments = self.loaddata.get_previous_alignments()
        utils.populate_lists(self.prev_alignments, self.align_list, self.align_combobox)
        self.loaddata.get_starting_alignment(0)
        # For IBL case at the moment we only using single shank
        #self.current_shank_idx = 0

    def on_session_selected(self, idx):
        """
        Triggered when session is selected from drop down list options
        :param idx: index of chosen session (item) in drop down list
        :type idx: int
        """
        self.shank = None
        self.shank_list.clear()
        self.align_list.clear()
        shanks = self.loaddata.get_shanks(idx)
        if len(shanks) > 1:
            utils.populate_lists(shanks, self.shank_list, self.shank_combobox)

        self.loaddata.get_info(0)
        self.prev_alignments = self.loaddata.get_previous_alignments()
        utils.populate_lists(self.prev_alignments, self.align_list, self.align_combobox)
        self.loaddata.get_starting_alignment(0)

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

        # TODO clean up
        if self.shank is not None:
            if not self.grid_widget.grid_layout:
                self.grid_widget.tab_widget.blockSignals(True)
                self.grid_widget.tab_widget.setCurrentIndex(idx)
                self.hist_grid_widget.tab_widget.setCurrentIndex(idx)
                self.grid_widget.tab_widget.blockSignals(False)

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
            self.feature_prev, self.track_prev = self.loaddata.get_starting_alignment(0)

    def clear_shank(self, shank):

        shank_items = self.dock_items[shank]
        utils.remove_items(shank_items.fig_img, shank_items.img_plots)
        utils.remove_items(shank_items.fig_img, shank_items.img_cbars)
        utils.remove_items(shank_items.fig_line, shank_items.line_plots)
        utils.remove_items(shank_items.fig_probe, shank_items.probe_plots)
        utils.remove_items(shank_items.fig_probe, shank_items.probe_cbars)
        shank_items.fig_slice.clear()
        shank_items.fig_hist.clear()
        shank_items.ax_hist.setTicks([])
        shank_items.fig_hist_ref.clear()
        shank_items.ax_hist_ref.setTicks([])
        shank_items.fig_scale.clear()
        shank_items.fit_scatter.setData()
        shank_items.fit_plot.setData()

    def shank_button_pressed(self):

        self.shank = self.loaddata.get_selected_probe()
        self.shank.loaders['align'].set_init_alignment()
        self.selected_shank = self.loaddata.probe_label
        self.selected_idx = self.loaddata.shank_idx


        self.remove_points_from_display()
        self.add_points_to_display()

        if not self.grid_widget.grid_layout:
            self.remove_fit_panels()
            self.plot_fit_panels(shanks=[self.selected_shank])
        else:
            self.select_background()

    def select_background(self):
        for shank in self.all_shanks:
            shank_items = self.dock_items[shank]
            if shank == self.selected_shank:
                shank_items.header.setStyleSheet("""
                QLabel {
                    background-color: #2c3e50;
                    color: white;
                    padding: 6px;
                    font-weight: bold;
                }
            """)
            else:
                shank_items.header.setStyleSheet("""
                QLabel {
                    background-color: rgb(240, 240, 240);
                    color: black;
                    padding: 6px;
                    font-weight: bold;
                }
            """)
            # shank_items.header.style().unpolish(shank_items.header)
            # shank_items.header.style().polish(shank_items.header)
            # shank_items.header.update()


    def align_button_pressed(self):
        # remove reference lines
        # remove reference lines
        # add reference lines

        self.remove_reference_lines_from_display(shanks=[self.selected_shank])
        shank_items= self.dock_items[self.selected_shank]

        shank_items.lines_features = np.empty((0, 3))
        shank_items.lines_tracks = np.empty((0, 1))
        shank_items.points = np.empty((0, 1))

        self.loaddata.probes[self.selected_shank].loaders['align'].set_init_alignment()
        feature_prev = self.loaddata.probes[self.selected_shank].loaders['align'].feature_prev
        if np.any(feature_prev):
            self.create_reference_lines(feature_prev[1:-1] * 1e6, self.dock_items[self.selected_shank])

        self.update_plots(shanks=[self.selected_shank])






    def data_button_pressed(self):
        """
        Triggered when Get Data button pressed, uses subject and session info to find eid and
        downloads and computes data needed for GUI display
        """

        # if self.shank is None: # when subject or probe is configured
        self.fig_fit.clear()
        self.clear_tabs()
        self.all_shanks = self.loaddata.probes
        self.init_tabs()
        for shank in self.all_shanks:
            self.init_shank_items(shank)
                #self.clear_shank(shank)

            #self.remove_reference_lines_from_display()

        self.loaddata.load_data()
        self.shank = self.loaddata.get_selected_probe()
        self.shank.loaders['align'].set_init_alignment()
        self.selected_shank = self.loaddata.probe_label
        # self.clear_shank(self.selected_shank)
        # self.init_shank_items(self.selected_shank)
        # self.remove_reference_lines_from_display()

        self.set_probe_lims(np.min([0, self.shank.plotdata.chn_min]), self.shank.plotdata.chn_max)
        self.init_menubar()
        self.get_scaled_histology()
        self.histology_exists = self.shank.histology

        # Initialise checked plots
        self.img_init.setChecked(True)
        self.line_init.setChecked(True)
        self.probe_init.setChecked(True)
        self.unit_init.setChecked(True)
        self.slice_init.setChecked(True)

        # Initialise ephys plots
        self.plot_image_panels('Firing Rate')
        self.plot_probe_panels('rms AP')
        self.plot_line_panels('Firing Rate')

        # Initialise histology plots
        self.plot_histology_ref_panels()
        self.plot_histology_panels()
        self.label_status = False
        self.toggle_labels()
        self.plot_scale_factor_panels()

        for shank in self.all_shanks:
            self.loaddata.probes[shank].loaders['align'].set_init_alignment()
            feature_prev = self.loaddata.probes[shank].loaders['align'].feature_prev
            if np.any(feature_prev):
                self.create_reference_lines(feature_prev[1:-1] * 1e6, self.dock_items[shank])

        self.remove_points_from_display()
        self.add_points_to_display()

        self.plot_fit_panels()
        self.plot_slice_panels('CCF')
        self.update_string()

        self.select_background()

        self.init_display()

        # Initialise unity plot
        # if self.unity:
        #     self.unitydata.add_regions(np.unique(self.hist_data['axis_label'][:, 1]))
        #     self.set_unity_xyz()
        #     self.plot_unity('probe')

        # # Only configure the view the first time the GUI is launched
        # self.set_view(view=1, configure=self.configure)
        # self.configure = False



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

        # if self.unity:
        #     self.unity_plot = 'probe'
        #     self.set_unity_xyz()
        #     self.plot_unity()

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


        shank_items = self.dock_items[self.selected_shank]
        line_track = np.array([line[0].pos().y() for line in shank_items.lines_tracks]) / 1e6
        line_feature = np.array([line[0].pos().y() for line in shank_items.lines_features]) / 1e6

        self.shank.loaders['align'].scale_hist_data(line_track, line_feature,
                                         extend_feature=self.extend_feature, lin_fit=self.lin_fit)

    def get_scaled_histology(self, shanks=()) -> None:
        """
        Retrieve scaled histological data after alignment operations.
        """
        shanks = shanks or self.all_shanks
        for shank in shanks:
            self.dock_items[shank].hist_data, self.dock_items[shank].hist_data_ref, self.dock_items[shank].scale_data \
                = self.loaddata.probes[shank].loaders['align'].get_scaled_histology()


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

    def update_plots(self, shanks=()) -> None:
        """
        Refresh all plots to reflect the current alignment state.
        """

        self.get_scaled_histology(shanks=shanks)
        self.plot_histology_panels(shanks=shanks)
        self.plot_scale_factor_panels(shanks=shanks)
        self.plot_fit_panels(shanks=shanks)
        self.plot_channel_panels(shanks=shanks)
        if self.unity:
            self.set_unity_xyz()
            self.plot_unity()
        self.remove_reference_lines_from_display(shanks=shanks)
        self.add_reference_lines_to_display(shanks=shanks)
        self.align_reference_lines()
        self.set_yaxis_range('fig_hist', shanks=shanks)
        self.update_string()

        # for hook in self.plot_hooks:
        #     hook(self)
        # def register_hook(self, func):
        #     self.plot_hooks.append(func)

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
        Updates all plots and indices with next alignment. Ensures user cannot go past latest move
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
        Ensures user cannot go back more than self.max_idx moves
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

        shank_items= self.dock_items[self.selected_shank]
        self.remove_reference_lines_from_display(shanks=[self.selected_shank])
        shank_items.lines_features = np.empty((0, 3))
        shank_items.lines_tracks = np.empty((0, 1))
        shank_items.points = np.empty((0, 1))

        self.shank.loaders['align'].reset_features_and_tracks()

        self.loaddata.probes[self.selected_shank].loaders['align'].set_init_alignment()
        feature_prev = self.loaddata.probes[self.selected_shank].loaders['align'].feature_prev
        if np.any(feature_prev):
            self.create_reference_lines(feature_prev[1:-1] * 1e6, self.dock_items[self.selected_shank])

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


    def on_mouse_double_clicked(self, event, i) -> None:
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
            if i != self.selected_idx:
                self.shank_combobox.setCurrentIndex(i)
                self.on_shank_selected(i)

            shank_items = self.dock_items[self.selected_shank]
            pos = shank_items.ephys_plot.mapFromScene(event.scenePos())
            self.create_reference_line(pos.y() * self.y_scale, shank_items)

    def on_mouse_hover(self, hover_items: List[pg.GraphicsObject]) -> None:
        """
        Handles mouse hover events over pyqtgraph plot shank_items.

        Identifies reference lines or linear regions the mouse is hovering over
        to allow interactive operations like deletion or displaying additional info.

        Parameters
        ----------
        shank_items : list of pyqtgraph.GraphicsObject
            List of shank_items under the mouse cursor.
        """

        if len(hover_items) > 1:
            self.selected_line = []
            item0, item1 = hover_items[0], hover_items[1]
            # TODO think about adding names to keep track of things
            if isinstance(item0, pg.InfiniteLine):
                self.selected_line = item0
            elif isinstance(item1, pg.LinearRegionItem):

                # Check if we are on the fig_scale plot
                shank_items = self._get_figure(item0, 'fig_scale')
                if shank_items is not None:
                    idx = np.where(shank_items.scale_regions == item1)[0][0]
                    shank_items.fig_scale_ax.setLabel('Scale Factor = ' + str(np.around(shank_items.scale_factor[idx], 2)))
                    return
                # Check if we are on the histology plot
                shank_items = self._get_figure(item0, 'fig_hist')
                if shank_items is not None:
                    self.selected_region = item1
                    return
                shank_items = self._get_figure(item0, 'fig_hist_ref')
                if shank_items is not None:
                    self.selected_region = item1
                    return

    def _get_figure(self, item, fig):
        return next((shank_items for _, shank_items in self.dock_items.items() if item == shank_items[fig]), None)



    # -------------------------------------------------------------------------------------------------
    # Display options
    # -------------------------------------------------------------------------------------------------
    def toggle_labels(self, shanks=()) -> None:
        """
        Toggle visibility of brain region labels on histology plots.

        Triggered by pressing Shift+A. Updates the pens for axis shank_items on both the main
        and reference histology plots to show or hide Allen atlas region labels.
        """
        self.label_status = not self.label_status
        pen = 'k' if self.label_status else None
        shanks = shanks or self.all_shanks
        for shank in shanks:
            shank_items = self.dock_items[shank]
            shank_items.ax_hist_ref.setPen(pen)
            shank_items.ax_hist_ref.setTextPen(pen)
            shank_items.ax_hist.setPen(pen)
            shank_items.ax_hist.setTextPen(pen)

            shank_items.fig_hist_ref.update()
            shank_items.fig_hist.update()

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
        # If no histology we can't plot histology
        if not self.histology_exists:
            return

        self.channel_status = not self.channel_status

        shanks = shanks or self.all_shanks
        for shank in shanks:
            shank_items = self.dock_items[shank]
            # Choose the appropriate method (addItem or removeItem) based on toggle state
            toggleItem = shank_items.fig_slice.addItem if self.channel_status else shank_items.fig_slice.removeItem

            toggleItem(shank_items.traj_line)
            toggleItem(shank_items.slice_chns)
            for line in shank_items.slice_lines:
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

    def tab_changed(self, index, from_hist=False):
        self.shank_combobox.setCurrentIndex(index)
        self.on_shank_selected(index)
        if not from_hist:
            self.hist_tabs.tab_widget.setCurrentIndex(index)
        self.remove_fit_panels()
        self.plot_fit_panels(shanks=[self.selected_shank])

    def hist_tab_changed(self, index):
        self.shank_tabs.tab_widget.setCurrentIndex(index)

    def toggle_layout(self):
        self.hist_tabs.toggle_layout()
        self.hist_tabs.tab_widget.setCurrentIndex(self.selected_idx)
        self.shank_tabs.toggle_layout()
        self.shank_tabs.tab_widget.setCurrentIndex(self.selected_idx)
        if self.shank_tabs.grid_layout:
            self.plot_fit_panels()
        else:
            self.remove_fit_panels()

    # -------------------------------------------------------------------------------------------------
    # Probe top and tip lines
    # -------------------------------------------------------------------------------------------------

    def set_probe_lims(
            self,
            min_val: Union[int, float],
            max_val: Union[int, float],
            shanks=()
        ) -> None:
        """
        Set the limits for the probe tip and probe top, and update the associated lines accordingly.

        Parameters
        ----------
        min_val : int or float
            The new minimum limit representing the probe tip position.
        max_val : int or float
            The new maximum limit representing the probe top position.
        """

        shanks = shanks or self.all_shanks
        for shank in shanks:
            shank_items = self.dock_items[shank]
            shank_items.probe_tip = min_val
            shank_items.probe_top = max_val
            shank_items = self.dock_items[shank]
            for top_line in shank_items.probe_top_lines:
                top_line.setY(shank_items.probe_top)

            for tip_line in shank_items.probe_tip_lines:
                tip_line.setY(shank_items.probe_tip)

    def tip_line_moved(self) -> None:
        """
        Callback triggered when the probe tip line (dotted line) on the histology figure is moved.

        This function updates the probe top line's vertical position to maintain a fixed
        vertical distance between tip and top (i.e., the probe length).
        """
        shank_items = self.dock_items[self.selected_shank]
        shank_items.top_pos.setPos(shank_items.tip_pos.value() + shank_items.probe_top)

    def top_line_moved(self) -> None:
        """
        Callback triggered when the probe top line (dotted line) on the histology figure is moved.

        This function updates the probe tip line's vertical position to maintain a fixed
        vertical distance between tip and top (i.e., the probe length).
        """
        shank_items = self.dock_items[self.selected_shank]
        shank_items.tip_pos.setPos(shank_items.top_pos.value() - shank_items.probe_top)


    # -------------------------------------------------------------------------------------------------
    # Reference lines
    # -------------------------------------------------------------------------------------------------

    def create_reference_line(
            self,
            pos: float,
            shank_items
        ) -> None:
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
        """
        colour = shank_items.colour if self.fix_reference_colours else None
        pen, brush = utils.create_line_style(colour=colour)

        # Reference line on histology figure (track)
        line_track = pg.InfiniteLine(pos=pos, angle=0, pen=pen, movable=True)
        line_track.sigPositionChanged.connect(lambda track, i=shank_items.idx: self.update_track_reference_line(track, i))
        line_track.setZValue(100)
        shank_items.fig_hist.addItem(line_track)

        # Reference lines on electrophysiology figures (feature)
        line_features = []
        for fig in [shank_items.fig_img, shank_items.fig_line, shank_items.fig_probe]:
            line_feature = pg.InfiniteLine(pos=pos, angle=0, pen=pen, movable=True)
            line_feature.setZValue(100)
            line_feature.sigPositionChanged.connect(lambda feature, i=shank_items.idx: self.update_feature_reference_line(feature, i))
            fig.addItem(line_feature)
            line_features.append(line_feature)

        shank_items.lines_features = np.vstack([shank_items.lines_features, line_features])
        shank_items.lines_tracks = np.vstack([shank_items.lines_tracks, line_track])

        # Add marker to fit figure
        point = pg.PlotDataItem()
        point.setData(x=[line_track.pos().y()], y=[line_features[0].pos().y()],
                      symbolBrush=brush, symbol='o', symbolSize=10)
        self.fig_fit.addItem(point)
        shank_items.points = np.vstack([shank_items.points, point])

    def create_reference_lines(
            self,
            positions: Union[np.ndarray, List[float]],
            shank_items
        ) -> None:

        """
        Create movable horizontal reference lines across electrophysiology and histology figures at multiple positions

        For each y-position in `positions`, this method creates:
        - A movable horizontal line in the histology figure (track line)
        - Synchronized lines in the image, line, and probe figures (feature lines)
        - A corresponding scatter point in the fit figure

        Parameters
        ----------
        positions : array-like of float
            List or array of y-axis positions at which to draw horizontal lines across
            histology, image, line, probe, and fit figures.
        """

        for pos in positions:
            self.create_reference_line(pos, shank_items)

    def delete_reference_line(self) -> None:
        """
        Delete the currently selected reference line from all plots.

        Triggered when the user hovers over a reference line and presses the Delete key.
        Removes:
        - Reference lines from image, line, probe, and histology figures
        - Associated scatter point from the fit figure
        """
        if not self.selected_line:
            return

        shank_items = self.dock_items[self.selected_shank]
        # Attempt to find selected line in feature lines
        line_idx = np.where(shank_items.lines_features == self.selected_line)[0]
        if line_idx.size == 0:
            # If not found, try in track lines
            line_idx = np.where(shank_items.lines_tracks == self.selected_line)[0]
            if line_idx.size == 0:
                return  # Couldn't find either

        line_idx = line_idx[0]

        # Remove line shank_items from plots
        shank_items.fig_img.removeItem(shank_items.lines_features[line_idx][0])
        shank_items.fig_line.removeItem(shank_items.lines_features[line_idx][1])
        shank_items.fig_probe.removeItem(shank_items.lines_features[line_idx][2])
        shank_items.fig_hist.removeItem(shank_items.lines_tracks[line_idx, 0])
        self.fig_fit.removeItem(shank_items.points[line_idx, 0])

        # Remove from tracking arrays
        shank_items.lines_features = np.delete(shank_items.lines_features, line_idx, axis=0)
        shank_items.lines_tracks = np.delete(shank_items.lines_tracks, line_idx, axis=0)
        shank_items.points = np.delete(shank_items.points, line_idx, axis=0)

    def update_feature_reference_line(
            self,
            feature_line: pg.InfiniteLine,
            idx
        ) -> None:
        """
        Callback triggered when a reference line is moved in one of the electrophysiology plots
        (image, line, or probe). This function ensures the line's new position is synchronized
        across the other ephys plots, and updates the corresponding scatter point in the fit plot.

        Parameters
        ----------
        feature_line : pyqtgraph.InfiniteLine
            The line instance that was moved by the user.
        """
        if idx != self.selected_idx:
            self.shank_combobox.setCurrentIndex(idx)
            self.on_shank_selected(idx)

        shank_items = self.dock_items[self.selected_shank]
        idx = np.where(shank_items.lines_features == feature_line)

        line_idx = idx[0][0]
        fig_idx = np.setdiff1d(np.arange(0, 3), idx[1][0]) #  Indices of two other plots

        # Update the other two lines to the new y-position
        shank_items.lines_features[line_idx][fig_idx[0]].setPos(feature_line.value())
        shank_items.lines_features[line_idx][fig_idx[1]].setPos(feature_line.value())

        # Update scatter point on the fit figure
        shank_items.points[line_idx][0].setData(x=[shank_items.lines_features[line_idx][0].pos().y()],
                                         y=[shank_items.lines_tracks[line_idx][0].pos().y()])

    def update_track_reference_line(
            self,
            track_line: pg.InfiniteLine,
            idx
        ) -> None:
        """
        Callback triggered when a reference line in the histology plot is moved.
        This updates the corresponding scatter point in the fit plot.

        Parameters
        ----------
        track_line : pyqtgraph.InfiniteLine
            The line instance that was moved by the user.
        """

        if idx != self.selected_idx:
            self.shank_combobox.setCurrentIndex(idx)
            self.on_shank_selected(idx)

        shank_items = self.dock_items[self.selected_shank]
        line_idx = np.where(shank_items.lines_tracks == track_line)[0][0]

        shank_items.points[line_idx][0].setData(x=[shank_items.lines_features[line_idx][0].pos().y()],
                                         y=[shank_items.lines_tracks[line_idx][0].pos().y()])

    def align_reference_lines(self) -> None:
        """
        Align the positions of all track reference lines and scatter points based on the new positions
        of their corresponding feature reference lines.
        """

        shank_items = self.dock_items[self.selected_shank]
        for line_feature, line_track, point in zip(shank_items.lines_features, shank_items.lines_tracks, shank_items.points):
            line_track[0].setPos(line_feature[0].getYPos())
            point[0].setData(x=[line_feature[0].pos().y()], y=[line_feature[0].pos().y()])

    def remove_points_from_display(self, shanks=()):
        shanks = shanks or self.all_shanks
        for shank in shanks:
            shank_items = self.dock_items[shank]
            for point in shank_items.points:
                self.fig_fit.removeItem(point[0])

    def add_points_to_display(self):

        shank_items = self.dock_items[self.selected_shank]
        for point in shank_items.points:
            self.fig_fit.addItem(point[0])

    def remove_reference_lines_from_display(self, shanks=()) -> None:
        """
        Remove all reference lines and scatter points from the electrophysiology, histology, and fit plots.
        """
        shanks = shanks or self.all_shanks
        for shank in shanks:
            shank_items = self.dock_items[shank]
            for line_feature, line_track, point in zip(shank_items.lines_features, shank_items.lines_tracks, shank_items.points):
                shank_items.fig_img.removeItem(line_feature[0])
                shank_items.fig_line.removeItem(line_feature[1])
                shank_items.fig_probe.removeItem(line_feature[2])
                shank_items.fig_hist.removeItem(line_track[0])
                self.fig_fit.removeItem(point[0])

    def add_reference_lines_to_display(self, shanks=()) -> None:
        """
        Add previously created reference lines and scatter points to their respective plots.
        """
        shanks = shanks or self.all_shanks
        for shank in shanks:
            shank_items = self.dock_items[shank]
            for line_feature, line_track, point in zip(shank_items.lines_features, shank_items.lines_tracks, shank_items.points):
                shank_items.fig_img.addItem(line_feature[0])
                shank_items.fig_line.addItem(line_feature[1])
                shank_items.fig_probe.addItem(line_feature[2])
                shank_items.fig_hist.addItem(line_track[0])
                if shank == self.selected_shank:
                    self.fig_fit.addItem(point[0])

    # -------------------------------------------------------------------------------------------------
    # Plugins
    # -------------------------------------------------------------------------------------------------





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
