import json
import os
import platform
from pathlib import Path

if platform.system() == 'Darwin':
    if platform.release().split('.')[0] >= '20':
        os.environ["QT_MAC_WANTS_LAYER"] = "1"

from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import QThread, QObject
from ephys_alignment_gui.thread_worker import Worker
import pyqtgraph as pg

import numpy as np
import numpy.typing as npt
from random import randrange

import pandas
import ants

import ephys_alignment_gui.plot_data as pd
from ephys_alignment_gui.plot_elements import ColorBar
import ephys_alignment_gui.ephys_gui_setup as ephys_gui

from ephys_alignment_gui.load_data_local import LoadDataLocal
from ephys_alignment_gui.windows.subject_scaling import ScalingWindow
from ephys_alignment_gui.create_overview_plots import make_overview_plot

from ephys_alignment_gui.windows.features_across_region import RegionFeatureWindow
from ephys_alignment_gui.ephys_alignment import EphysAlignment

import matplotlib.pyplot as mpl  # noqa  # This is needed to make qt show properly :/

from ephys_alignment_gui.docdb import write_output_to_docdb

ANTS_DIMENSION = 3

class MainWindow(QtWidgets.QMainWindow, ephys_gui.Setup):

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

    def __init__(self, offline=True, probe_id=None, one=None, histology=True,
                 spike_collection=None, remote=False):
        super(MainWindow, self).__init__()

        self.init_variables()
        self.init_layout(self, offline=offline)
        self.configure = True
        one_mode = 'remote' if remote else 'auto'

        self.loaddata = LoadDataLocal()
        self.offline = True
        self.histology_exists = True
        self.data_status = False
        self.output_directory = None
        self.data_root = None

        self.allen = self.loaddata.get_allen_csv()
        self.init_region_lookup(self.allen)

    def init_variables(self):
        """
        Initialise variables
        """
        # Line styles and fonts
        self.kpen_dot = pg.mkPen(color='k', style=QtCore.Qt.DotLine, width=2)
        self.reference_line_kpen = pg.mkPen(color='k', style=QtCore.Qt.DotLine, width=10)
        self.rpen_dot = pg.mkPen(color='r', style=QtCore.Qt.DotLine, width=2)
        self.kpen_solid = pg.mkPen(color='k', style=QtCore.Qt.SolidLine, width=2)
        self.bpen_solid = pg.mkPen(color='b', style=QtCore.Qt.SolidLine, width=3)
        self.bar_colour = QtGui.QColor(160, 160, 160)

        # Padding to add to figures to make sure always same size viewbox
        self.pad = 0.05

        # Variables to do with probe dimension
        self.probe_tip = 0
        self.probe_top = 3840
        self.probe_extra = 100
        self.view_total = [-2000, 6000]
        self.depth = np.arange(self.view_total[0], self.view_total[1], 20)
        self.extend_feature = 1

        # Initialise with linear fit scaling as default
        self.lin_fit = True

        # Variables to keep track of number of fits (max 10)
        self.max_idx = 10
        self.idx = 0
        self.current_idx = 0
        self.total_idx = 0
        self.last_idx = 0
        self.diff_idx = 0

        # Variables to keep track of reference lines and points added
        self.line_status = True
        self.label_status = True
        self.channel_status = True
        self.hist_bound_status = True
        self.lines_features = np.empty((0, 3))
        self.lines_tracks = np.empty((0, 1))
        self.points = np.empty((0, 1))
        self.y_scale = 1
        self.x_scale = 1

        # Variables to keep track of plots and colorbars
        self.img_plots = []
        self.line_plots = []
        self.probe_plots = []
        self.img_cbars = []
        self.probe_cbars = []
        self.scale_regions = np.empty((0, 1))
        self.slice_lines = []
        self.slice_items = []
        self.probe_bounds = []

        # Variables to keep track of popup plots
        self.cluster_popups = []
        self.label_popup = []
        self.popup_status = True
        self.subj_win = None

        self.hist_data = {
            'region': [],
            'axis_label': [],
            'colour': []
        }

        self.hist_data_ref = {
            'region': [],
            'axis_label': [],
            'colour': []
        }

        self.scale_data = {
            'region': [],
            'scale': []
        }

        self.hist_nearby_x = None
        self.hist_nearby_y = None
        self.hist_nearby_col = None
        self.hist_nearby_parent_x = None
        self.hist_nearby_parent_y = None
        self.hist_nearby_parent_col = None
        self.hist_mapping = 'Allen'

        self.track = [0] * (self.max_idx + 1)
        self.features = [0] * (self.max_idx + 1)

        self.nearby = None

    def set_axis(self, fig, ax, show=True, label=None, pen='k', ticks=True):
        """
        Show/hide and configure axis of figure
        :param fig: figure associated with axis
        :type fig: pyqtgraph PlotWidget
        :param ax: orientation of axis, must be one of 'left', 'right', 'top' or 'bottom'
        :type ax: string
        :param show: 'True' to show axis, 'False' to hide axis
        :type show: bool
        :param label: axis label
        :type label: string
        :parm pen: colour on axis
        :type pen: string
        :param ticks: 'True' to show axis ticks, 'False' to hide axis ticks
        :param ticks: bool
        :return axis: axis object
        :type axis: pyqtgraph AxisItem
        """
        if not label:
            label = ''
        if type(fig) == pg.PlotItem:
            axis = fig.getAxis(ax)
        else:
            axis = fig.plotItem.getAxis(ax)
        if show:
            axis.show()
            axis.setPen(pen)
            axis.setTextPen(pen)
            axis.setLabel(label)
            if not ticks:
                axis.setTicks([[(0, ''), (0.5, ''), (1, '')]])
        else:
            axis.hide()

        return axis

    def set_font(self, fig, ax, ptsize=8, width=None, height=None):

        if type(fig) == pg.PlotItem:
            axis = fig.getAxis(ax)
        else:
            axis = fig.plotItem.getAxis(ax)

        font = QtGui.QFont()
        font.setPointSize(ptsize)
        axis.setStyle(tickFont=font)
        labelStyle = {'font-size': f'{ptsize}pt'}
        axis.setLabel(**labelStyle)

        if width:
            axis.setWidth(width)
        if height:
            axis.setHeight(height)

    def set_lims(self, min, max):
        self.probe_tip = min
        self.probe_top = max

        [top_line.setY(self.probe_top) for top_line in self.probe_top_lines]
        [tip_line.setY(self.probe_tip) for tip_line in self.probe_tip_lines]

    def populate_lists(self, data, list_name, combobox):
        """
        Populate drop down lists with subject/session/alignment options
        :param data: list of options to add to widget
        :type data: 1D array of strings
        :param list_name: widget object to which to add data to
        :type list_name: QtGui.QStandardItemModel
        :param combobox: combobox object to which to add data to
        :type combobox: QtWidgets.QComboBox
        """
        list_name.clear()
        for dat in data:
            item = QtGui.QStandardItem(dat)
            item.setEditable(False)
            list_name.appendRow(item)

        # This makes sure the drop down menu is wide enough to showw full length of string
        min_width = combobox.fontMetrics().width(max(data, key=len))
        min_width += combobox.view().autoScrollMargin()
        min_width += combobox.style().pixelMetric(QtWidgets.QStyle.PM_ScrollBarExtent)
        combobox.view().setMinimumWidth(min_width)

        # Set the default to be the first option
        combobox.setCurrentIndex(0)

    def set_view(self, view=1, configure=False):
        """
        Layout of ephys data figures, can be changed using Shift+1, Shift+2, Shift+3
        :param view: from left to right
            1: img plot, line plot, probe plot
            2: img plot, probe plot, line plot
            3: probe plot, line plot, img_plot
        :type view: int
        :param configure: Returns the width of each image, set to 'True' once during the setup to
                          ensure figures are always the same width
        :type configure: bool
        """
        if configure:
            self.fig_ax_width = self.fig_data_ax.width()
            self.fig_img_width = self.fig_img.width() - self.fig_ax_width
            self.fig_line_width = self.fig_line.width()
            self.fig_probe_width = self.fig_probe.width()
            self.slice_width = self.fig_slice.width()
            self.slice_height = self.fig_slice.height()
            self.slice_rect = self.fig_slice.viewRect()

        if view == 1:
            self.fig_data_layout.removeItem(self.fig_img_cb)
            self.fig_data_layout.removeItem(self.fig_probe_cb)
            self.fig_data_layout.removeItem(self.fig_img)
            self.fig_data_layout.removeItem(self.fig_line)
            self.fig_data_layout.removeItem(self.fig_probe)
            self.fig_data_layout.addItem(self.fig_img_cb, 0, 0)
            self.fig_data_layout.addItem(self.fig_probe_cb, 0, 1, 1, 2)
            self.fig_data_layout.addItem(self.fig_img, 1, 0)
            self.fig_data_layout.addItem(self.fig_line, 1, 1)
            self.fig_data_layout.addItem(self.fig_probe, 1, 2)

            self.set_axis(self.fig_img_cb, 'left', pen='w')
            self.set_axis(self.fig_probe_cb, 'left', show=False)
            self.set_axis(self.fig_img, 'left', label='Distance from probe tip (um)')
            self.set_axis(self.fig_probe, 'left', show=False)
            self.set_axis(self.fig_line, 'left', show=False)

            self.fig_img.setPreferredWidth(self.fig_img_width + self.fig_ax_width)
            self.fig_line.setPreferredWidth(self.fig_line_width)
            self.fig_probe.setFixedWidth(self.fig_probe_width)

            self.fig_data_layout.layout.setColumnStretchFactor(0, 6)
            self.fig_data_layout.layout.setColumnStretchFactor(1, 2)
            self.fig_data_layout.layout.setColumnStretchFactor(2, 1)
            self.fig_data_layout.layout.setRowStretchFactor(0, 1)
            self.fig_data_layout.layout.setRowStretchFactor(1, 10)

            self.fig_img.update()
            # Manually force the axis to shift and then reset axis as axis not always correct
            # TO DO: find a better way!
            self.fig_img.setXRange(min=self.xrange[0] - 10, max=self.xrange[1] + 10, padding=0)
            self.reset_axis_button_pressed()
            self.fig_line.update()
            self.fig_probe.update()

        if view == 2:
            self.fig_data_layout.removeItem(self.fig_img_cb)
            self.fig_data_layout.removeItem(self.fig_probe_cb)
            self.fig_data_layout.removeItem(self.fig_img)
            self.fig_data_layout.removeItem(self.fig_line)
            self.fig_data_layout.removeItem(self.fig_probe)
            self.fig_data_layout.addItem(self.fig_img_cb, 0, 0)

            self.fig_data_layout.addItem(self.fig_probe_cb, 0, 1, 1, 2)
            self.fig_data_layout.addItem(self.fig_img, 1, 0)
            self.fig_data_layout.addItem(self.fig_probe, 1, 1)
            self.fig_data_layout.addItem(self.fig_line, 1, 2)

            self.set_axis(self.fig_img_cb, 'left', pen='w')
            self.set_axis(self.fig_probe_cb, 'left', show=False)
            self.set_axis(self.fig_img, 'left', label='Distance from probe tip (um)')
            self.set_axis(self.fig_probe, 'left', show=False)
            self.set_axis(self.fig_line, 'left', show=False)

            self.fig_img.setPreferredWidth(self.fig_img_width + self.fig_ax_width)
            self.fig_line.setPreferredWidth(self.fig_line_width)
            self.fig_probe.setFixedWidth(self.fig_probe_width)

            self.fig_data_layout.layout.setColumnStretchFactor(0, 6)
            self.fig_data_layout.layout.setColumnStretchFactor(1, 1)
            self.fig_data_layout.layout.setColumnStretchFactor(2, 2)
            self.fig_data_layout.layout.setRowStretchFactor(0, 1)
            self.fig_data_layout.layout.setRowStretchFactor(1, 10)

            self.fig_img.update()
            self.fig_img.setXRange(min=self.xrange[0] - 10, max=self.xrange[1] + 10, padding=0)
            self.reset_axis_button_pressed()
            self.fig_line.update()
            self.fig_probe.update()

        if view == 3:
            self.fig_data_layout.removeItem(self.fig_img_cb)
            self.fig_data_layout.removeItem(self.fig_probe_cb)
            self.fig_data_layout.removeItem(self.fig_img)
            self.fig_data_layout.removeItem(self.fig_line)
            self.fig_data_layout.removeItem(self.fig_probe)
            self.fig_data_layout.addItem(self.fig_probe_cb, 0, 0, 1, 2)
            self.fig_data_layout.addItem(self.fig_img_cb, 0, 2)
            self.fig_data_layout.addItem(self.fig_probe, 1, 0)
            self.fig_data_layout.addItem(self.fig_line, 1, 1)
            self.fig_data_layout.addItem(self.fig_img, 1, 2)

            self.set_axis(self.fig_probe_cb, 'left', pen='w')
            self.set_axis(self.fig_img_cb, 'left', show=False)
            self.set_axis(self.fig_line, 'left', show=False)
            self.set_axis(self.fig_img, 'left', pen='w')
            self.set_axis(self.fig_img, 'left', show=False)
            self.set_axis(self.fig_probe, 'left', label='Distance from probe tip (um)')

            self.fig_data_layout.layout.setColumnStretchFactor(0, 1)
            self.fig_data_layout.layout.setColumnStretchFactor(1, 2)
            self.fig_data_layout.layout.setColumnStretchFactor(2, 6)
            self.fig_data_layout.layout.setRowStretchFactor(0, 1)
            self.fig_data_layout.layout.setRowStretchFactor(1, 10)

            self.fig_probe.setFixedWidth(self.fig_probe_width + self.fig_ax_width)
            self.fig_img.setPreferredWidth(self.fig_img_width)
            self.fig_line.setPreferredWidth(self.fig_line_width)

            self.fig_img.update()
            self.fig_img.setXRange(min=self.xrange[0] - 10, max=self.xrange[1] + 10, padding=0)
            self.reset_axis_button_pressed()
            self.fig_line.update()
            self.fig_probe.update()

    def save_plots(self, save_path=None):
        """
        Saves all plots from the GUI into folder
        """
        # make folder to save plots to
        sess_info = ''

        if save_path:
            image_path_overview = Path(save_path)
        else:
            if self.loaddata.output_directory == None:
                self.on_output_folder_selected()
            image_path_overview = Path(self.loaddata.output_directory / f"Plots_Shank_{self.current_shank_idx + 1}")

        os.makedirs(image_path_overview, exist_ok=True)
        os.makedirs(image_path_overview, exist_ok=True)
        # Reset all axis, put view back to 1 and remove any reference lines
        self.reset_axis_button_pressed()
        self.set_view(view=1, configure=False)
        # self.remove_lines_points()

        xlabel_img = self.fig_img.getAxis('bottom').label.toPlainText()
        xlabel_line = self.fig_line.getAxis('bottom').label.toPlainText()

        # First go through all the image plots
        self.fig_data_layout.removeItem(self.fig_probe)
        self.fig_data_layout.removeItem(self.fig_probe_cb)
        self.fig_data_layout.removeItem(self.fig_line)

        width1 = self.fig_data_area.width()
        height1 = self.fig_data_area.height()
        ax_width = self.fig_img.getAxis('left').width()
        ax_height = self.fig_img_cb.getAxis('top').height()

        self.set_font(self.fig_img, 'left', ptsize=15, width=ax_width + 20)
        self.set_font(self.fig_img, 'bottom', ptsize=15)
        self.set_font(self.fig_img_cb, 'top', ptsize=15, height=ax_height + 15)

        self.fig_data_area.resize(700, height1)

        plot = None
        start_plot = self.img_options_group.checkedAction()

        while plot != start_plot:
            self.set_font(self.fig_img_cb, 'top', ptsize=15, height=ax_height + 15)
            exporter = pg.exporters.ImageExporter(self.fig_data_layout.scene())
            exporter.export(str(image_path_overview.joinpath(sess_info + 'img_' +
                                                    self.img_options_group.checkedAction()
                                                    .text() + '.png')))
            self.add_lines_points()  # Add reference lines
            self.toggle_plots(self.img_options_group)
            # self.remove_lines_points()
            plot = self.img_options_group.checkedAction()

        self.set_font(self.fig_img, 'left', ptsize=8, width=ax_width)
        self.set_font(self.fig_img, 'bottom', ptsize=8)
        self.set_font(self.fig_img_cb, 'top', ptsize=8, height=ax_height)
        self.set_axis(self.fig_img, 'bottom', label=xlabel_img)
        self.fig_data_layout.removeItem(self.fig_img)
        self.fig_data_layout.removeItem(self.fig_img_cb)

        # Next go over probe plots
        self.fig_data_layout.addItem(self.fig_probe_cb, 0, 0, 1, 2)
        self.fig_data_layout.addItem(self.fig_probe, 1, 0)
        self.set_axis(self.fig_probe, 'left', label='Distance from probe tip (uV)')
        self.fig_probe.setFixedWidth(self.fig_probe_width + self.fig_ax_width + 20)
        self.set_font(self.fig_probe, 'left', ptsize=15, width=ax_width + 20)
        self.set_font(self.fig_probe_cb, 'top', ptsize=15, height=ax_height + 15)
        self.fig_data_area.resize(250, height1)

        plot = None
        start_plot = self.probe_options_group.checkedAction()

        while plot != start_plot:
            self.set_font(self.fig_probe_cb, 'top', ptsize=15, height=ax_height + 15)
            exporter = pg.exporters.ImageExporter(self.fig_data_layout.scene())
            exporter.export(str(image_path_overview.joinpath(sess_info + 'probe_' +
                                                    self.probe_options_group.checkedAction().
                                                    text() + '.png')))
            self.add_lines_points()  # Add reference line
            self.toggle_plots(self.probe_options_group)
            plot = self.probe_options_group.checkedAction()

        self.fig_probe.setFixedWidth(self.fig_probe_width + self.fig_ax_width)
        self.set_font(self.fig_probe, 'left', ptsize=8, width=ax_width)
        self.set_font(self.fig_probe_cb, 'top', ptsize=8, height=ax_height)
        self.set_axis(self.fig_probe, 'bottom', pen='w', label='blank')
        self.fig_data_layout.removeItem(self.fig_probe)
        self.fig_data_layout.removeItem(self.fig_probe_cb)

        # Next go through the line plots
        self.fig_data_layout.addItem(self.fig_probe_cb, 0, 0, 1, 2)
        self.fig_probe_cb.clear()
        text = self.fig_probe_cb.getAxis('top').label.toPlainText()
        self.set_axis(self.fig_probe_cb, 'top', pen='w')
        self.fig_data_layout.addItem(self.fig_line, 1, 0)

        self.set_axis(self.fig_line, 'left', label='Distance from probe tip (um)')
        self.set_font(self.fig_line, 'left', ptsize=15, width=ax_width + 20)
        self.set_font(self.fig_line, 'bottom', ptsize=15)
        self.fig_data_area.resize(200, height1)

        plot = None
        start_plot = self.line_options_group.checkedAction()
        while plot != start_plot:
            exporter = pg.exporters.ImageExporter(self.fig_data_layout.scene())
            exporter.export(str(image_path_overview.joinpath(sess_info + 'line_' +
                                                    self.line_options_group.checkedAction().
                                                    text() + '.png')))
            self.add_lines_points()  # Add reference line
            self.toggle_plots(self.line_options_group)
            plot = self.line_options_group.checkedAction()

        [self.fig_probe_cb.addItem(cbar) for cbar in self.probe_cbars]
        self.set_axis(self.fig_probe_cb, 'top', pen='k', label=text)
        self.set_font(self.fig_line, 'left', ptsize=8, width=ax_width)
        self.set_font(self.fig_line, 'bottom', ptsize=8)
        self.set_axis(self.fig_line, 'bottom', label=xlabel_line)
        self.fig_data_layout.removeItem(self.fig_line)
        self.fig_data_layout.removeItem(self.fig_probe_cb)
        self.fig_data_area.resize(width1, height1)
        self.fig_data_layout.addItem(self.fig_probe_cb, 0, 0, 1, 2)
        self.fig_data_layout.addItem(self.fig_img_cb, 0, 2)
        self.fig_data_layout.addItem(self.fig_probe, 1, 0)
        self.fig_data_layout.addItem(self.fig_line, 1, 1)
        self.fig_data_layout.addItem(self.fig_img, 1, 2)

        self.set_view(view=1, configure=False)

        # Save slice images
        plot = None
        start_plot = self.slice_options_group.checkedAction()
        while plot != start_plot:
            self.toggle_channel_button_pressed()
            self.traj_line.setData(x=self.xyz_channels[:, 0], y=self.xyz_channels[:, 2],
                                   pen=self.rpen_dot)
            self.fig_slice.addItem(self.traj_line)
            self.plot_channels()

            slice_name = self.slice_options_group.checkedAction().text()
            exporter = pg.exporters.ImageExporter(self.fig_slice)
            exporter.export(str(image_path_overview.joinpath(sess_info + 'slice_' + slice_name + '.png')))
            self.toggle_plots(self.slice_options_group)
            plot = self.slice_options_group.checkedAction()

        plot = None
        start_plot = self.slice_options_group.checkedAction()
        while plot != start_plot:
            self.toggle_channel_button_pressed()
            self.traj_line.setData(x=self.xyz_channels[:, 0], y=self.xyz_channels[:, 2],
                                   pen=self.rpen_dot)
            self.fig_slice.addItem(self.traj_line)
            self.plot_channels()

            slice_name = self.slice_options_group.checkedAction().text()
            self.fig_slice.setXRange(min=np.min(self.xyz_channels[:, 0]) - 200 / 1e6,
                                     max=np.max(self.xyz_channels[:, 0]) + 200 / 1e6)
            self.fig_slice.setYRange(min=np.min(self.xyz_channels[:, 2]) - 500 / 1e6,
                                     max=np.max(self.xyz_channels[:, 2]) + 500 / 1e6)
            self.fig_slice.resize(50, self.slice_height)
            exporter = pg.exporters.ImageExporter(self.fig_slice)
            exporter.export(
                str(image_path_overview.joinpath(sess_info + 'slice_zoom_' + slice_name + '.png')))
            self.fig_slice.resize(self.slice_width, self.slice_height)
            self.fig_slice.setRange(rect=self.slice_rect)
            self.toggle_plots(self.slice_options_group)
            plot = self.slice_options_group.checkedAction()

        # Save the brain regions image
        self.set_axis(self.fig_hist_extra_yaxis, 'left')
        # Add labels to show which ones are aligned
        self.set_axis(self.fig_hist, 'bottom', label='aligned')
        self.set_font(self.fig_hist, 'bottom', ptsize=12)
        self.set_axis(self.fig_hist_ref, 'bottom', label='original')
        self.set_font(self.fig_hist_ref, 'bottom', ptsize=12)
        exporter = pg.exporters.ImageExporter(self.fig_hist_layout.scene())
        exporter.export(str(image_path_overview.joinpath(sess_info + 'hist.png')))
        self.set_axis(self.fig_hist_extra_yaxis, 'left', pen=None)
        self.set_font(self.fig_hist, 'bottom', ptsize=8)
        self.set_axis(self.fig_hist, 'bottom', pen='w', label='blank')
        self.set_font(self.fig_hist_ref, 'bottom', ptsize=8)
        self.set_axis(self.fig_hist_ref, 'bottom', pen='w', label='blank')

        make_overview_plot(image_path_overview, sess_info, save_folder=image_path_overview)

        self.add_lines_points()

    def toggle_plots(self, options_group, reverse=False):
        """
        Allows user to toggle through image, line, probe and slice plots using keyboard shortcuts
        Alt+1, Alt+2, Alt+3 and Alt+4 respectively
        :param options_group: Set of plots to toggle through
        :param reverse: if True, goes backward
        :type options_group: QtGui.QActionGroup
        """

        current_act = options_group.checkedAction()
        actions = options_group.actions()
        current_idx = [iA for iA, act in enumerate(actions) if act == current_act][0]
        next_idx = np.mod(current_idx + (-1 if reverse else 1), len(actions))
        actions[next_idx].setChecked(True)
        actions[next_idx].trigger()

    """
    Plot functions
    """
    def plot_histology(self, fig, ax='left', movable=True):
        """
        Plots histology figure - brain regions that intersect with probe track
        :param fig: figure on which to plot
        :type fig: pyqtgraph PlotWidget
        :param ax: orientation of axis, must be one of 'left' (fig_hist) or 'right' (fig_hist_ref)
        :type ax: string
        :param movable: whether probe reference lines can be moved, True for fig_hist, False for
                        fig_hist_ref
        :type movable: Bool
        """

        # If no histology we can't plot histology
        if not self.histology_exists:
            return
        fig.clear()
        self.hist_regions = np.empty((0, 1))
        axis = fig.getAxis(ax)
        axis.setTicks([self.hist_data['axis_label']])
        axis.setTickFont(QtGui.QFont('Arial', 8))
        axis.setZValue(10)
        self.set_axis(self.fig_hist, 'bottom', pen='w', label='blank')
 
        # Plot each histology region
        for ir, reg in enumerate(self.hist_data['region']):
            colour = QtGui.QColor(*self.hist_data['colour'][ir])
            region = pg.LinearRegionItem(values=(reg[0], reg[1]),
                                         orientation=pg.LinearRegionItem.Horizontal,
                                         brush=colour, movable=False)
            # Add a white line at the boundary between regions
            bound = pg.InfiniteLine(pos=reg[0], angle=0, pen='w')
            fig.addItem(region)
            fig.addItem(bound)
            # Need to keep track of each histology region for label pressed interaction
            self.hist_regions = np.vstack([self.hist_regions, region])

        self.selected_region = self.hist_regions[-2]

        # Boundary for final region
        bound = pg.InfiniteLine(pos=self.hist_data['region'][-1][1], angle=0,
                                pen='w')

        fig.addItem(bound)
        # Add dotted lines to plot to indicate region along probe track where electrode
        # channels are distributed
        self.tip_pos = pg.InfiniteLine(pos=self.probe_tip, angle=0, pen=self.kpen_dot,
                                       movable=movable)
        self.top_pos = pg.InfiniteLine(pos=self.probe_top, angle=0, pen=self.kpen_dot,
                                       movable=movable)

        # Lines can be moved to adjust location of channels along the probe track
        # Ensure distance between bottom and top channel is always constant at 3840um and that
        # lines can't be moved outside interpolation bounds
        # Add offset of 1um to keep within bounds of interpolation
        offset = 1
        self.tip_pos.setBounds((self.track[self.idx][0] * 1e6 + offset,
                                self.track[self.idx][-1] * 1e6 -
                                (self.probe_top + offset)))
        self.top_pos.setBounds((self.track[self.idx][0] * 1e6 + (self.probe_top + offset),
                                self.track[self.idx][-1] * 1e6 - offset))
        self.tip_pos.sigPositionChanged.connect(self.tip_line_moved)
        self.top_pos.sigPositionChanged.connect(self.top_line_moved)

        # Add lines to figure
        fig.addItem(self.tip_pos)
        fig.addItem(self.top_pos)

    def plot_histology_ref(self, fig, ax='right', movable=False):
        """
        Plots histology figure - brain regions that intersect with probe track
        :param fig: figure on which to plot
        :type fig: pyqtgraph PlotWidget
        :param ax: orientation of axis, must be one of 'left' (fig_hist) or 'right' (fig_hist_ref)
        :type ax: string
        :param movable: whether probe reference lines can be moved, True for fig_hist, False for
                        fig_hist_ref
        :type movable: Bool
        """

        # If no histology we can't plot histology
        if not self.histology_exists:
            return

        fig.clear()
        self.hist_ref_regions = np.empty((0, 1))
        axis = fig.getAxis(ax)
        axis.setTicks([self.hist_data_ref['axis_label']])
        axis.setZValue(10)
        self.set_axis(self.fig_hist_ref, 'bottom', pen='w', label='blank')

        # Plot each histology region
        for ir, reg in enumerate(self.hist_data_ref['region']):
            colour = QtGui.QColor(*self.hist_data_ref['colour'][ir])
            region = pg.LinearRegionItem(values=(reg[0], reg[1]),
                                         orientation=pg.LinearRegionItem.Horizontal,
                                         brush=colour, movable=False)
            bound = pg.InfiniteLine(pos=reg[0], angle=0, pen='w')
            fig.addItem(region)
            fig.addItem(bound)
            self.hist_ref_regions = np.vstack([self.hist_ref_regions, region])

        bound = pg.InfiniteLine(pos=self.hist_data_ref['region'][-1][1], angle=0,
                                pen='w')
        fig.addItem(bound)
        # Add dotted lines to plot to indicate region along probe track where electrode
        # channels are distributed
        self.tip_pos = pg.InfiniteLine(pos=self.probe_tip, angle=0, pen=self.kpen_dot,
                                       movable=movable)
        self.top_pos = pg.InfiniteLine(pos=self.probe_top, angle=0, pen=self.kpen_dot,
                                       movable=movable)
        # Add lines to figure
        fig.addItem(self.tip_pos)
        fig.addItem(self.top_pos)

    def plot_histology_nearby(self, fig, ax='right', movable=False):
        """
        Plots histology figure - brain regions that intersect with probe track
        :param fig: figure on which to plot
        :type fig: pyqtgraph PlotWidget
        :param ax: orientation of axis, must be one of 'left' (fig_hist) or 'right' (fig_hist_ref)
        :type ax: string
        :param movable: whether probe reference lines can be moved, True for fig_hist, False for
                        fig_hist_ref
        :type movable: Bool
        """

        # If no histology we can't plot histology
        if not self.histology_exists:
            return

        fig.clear()
        self.hist_ref_regions = np.empty((0, 1))
        axis = fig.getAxis(ax)
        axis.setTicks([self.hist_data_ref['axis_label']])
        axis.setZValue(10)

        self.set_axis(fig, 'bottom', label='dist to boundary (um)')
        fig.setXRange(min=0, max=100)
        fig.setYRange(min=self.probe_tip - self.probe_extra, max=self.probe_top + self.probe_extra,
                      padding=self.pad)

        # Plot nearby regions
        for ir, (x, y, c) in enumerate(zip(self.hist_nearby_x, self.hist_nearby_y,
                                           self.hist_nearby_col)):
            colour = QtGui.QColor(c)
            plot = pg.PlotCurveItem()
            plot.setData(x=x, y=y * 1e6, fillLevel=10, fillOutline=True)
            plot.setBrush(colour)
            plot.setPen(colour)
            fig.addItem(plot)

        for ir, (x, y, c) in enumerate(zip(self.hist_nearby_parent_x, self.hist_nearby_parent_y,
                                           self.hist_nearby_parent_col)):
            colour = QtGui.QColor(c)
            colour.setAlpha(70)
            plot = pg.PlotCurveItem()
            plot.setData(x=x, y=y * 1e6, fillLevel=10, fillOutline=True)
            plot.setBrush(colour)
            plot.setPen(colour)
            fig.addItem(plot)

        # Add dotted lines to plot to indicate region along probe track where electrode
        # channels are distributed
        self.tip_pos = pg.InfiniteLine(pos=self.probe_tip, angle=0, pen=self.kpen_dot,
                                       movable=movable)
        self.top_pos = pg.InfiniteLine(pos=self.probe_top, angle=0, pen=self.kpen_dot,
                                       movable=movable)
        # Add lines to figure
        fig.addItem(self.tip_pos)
        fig.addItem(self.top_pos)

    def offset_hist_data(self):
        """
        Offset location of probe tip along probe track
        """
        # If no histology we can't do alignment
        if not self.histology_exists:
            return

        self.track[self.idx] = (self.track[self.idx_prev] + self.tip_pos.value() / 1e6)
        self.features[self.idx] = (self.features[self.idx_prev])

        self.get_scaled_histology()

    def scale_hist_data(self):
        """
        Scale brain regions along probe track
        """

        # If no histology we can't do alignment
        if not self.histology_exists:
            return

        # Track --> histology plot
        line_track = np.array([line[0].pos().y() for line in self.lines_tracks]) / 1e6
        # Feature --> ephys data plots
        line_feature = np.array([line[0].pos().y() for line in self.lines_features]) / 1e6
        depths_track = np.sort(np.r_[self.track[self.idx_prev][[0, -1]], line_track])

        self.track[self.idx] = self.ephysalign.feature2track(depths_track,
                                                             self.features[self.idx_prev],
                                                             self.track[self.idx_prev])

        self.features[self.idx] = np.sort(np.r_[self.features[self.idx_prev]
                                                [[0, -1]], line_feature])

        if (self.features[self.idx].size >= 5) & self.lin_fit:
            self.features[self.idx], self.track[self.idx] = \
                self.ephysalign.adjust_extremes_linear(self.features[self.idx],
                                                       self.track[self.idx], self.extend_feature)

        else:
            self.track[self.idx] = self.ephysalign.adjust_extremes_uniform(self.features[self.idx],
                                                                           self.track[self.idx])

        self.get_scaled_histology()

    def get_scaled_histology(self):
        if self.hist_mapping == 'Allen':
            self.hist_data['region'], self.hist_data['axis_label'] \
                = self.ephysalign.scale_histology_regions(self.features[self.idx], self.track[self.idx])
            self.hist_data['colour'] = self.ephysalign.region_colour

            self.scale_data['region'], self.scale_data['scale'] \
                = self.ephysalign.get_scale_factor(self.hist_data['region'])

            self.hist_data_ref['region'], self.hist_data_ref['axis_label'] \
                = self.ephysalign.scale_histology_regions(self.ephysalign.track_extent,
                                                          self.ephysalign.track_extent)
            self.hist_data_ref['colour'] = self.ephysalign.region_colour

        elif self.hist_mapping == 'FP':
            self.hist_data['region'], self.hist_data['axis_label'] \
                = self.ephysalign.scale_histology_regions(self.features[self.idx], self.track[self.idx], region=self.region_fp,
                                                          region_label=self.region_label_fp)
            self.hist_data['colour'] = self.region_colour_fp
            self.scale_data['region'], self.scale_data['scale'] \
                = self.ephysalign.get_scale_factor(self.hist_data['region'], region_orig=self.region_fp)

            self.hist_data_ref['region'], self.hist_data_ref['axis_label'] \
                = self.ephysalign.scale_histology_regions(self.ephysalign.track_extent, self.ephysalign.track_extent,
                                                          region=self.region_fp, region_label=self.region_label_fp)
            self.hist_data_ref['colour'] = self.region_colour_fp

    def plot_scale_factor(self):
        """
        Plots the scale factor applied to brain regions along probe track, displayed
        alongside histology figure
        """

        # If no histology we can't do alignment
        if not self.histology_exists:
            return

        self.fig_scale.clear()
        self.scale_regions = np.empty((0, 1))
        self.scale_factor = self.scale_data['scale']
        scale_factor = self.scale_data['scale'] - 0.5
        color_bar = ColorBar('seismic')
        cbar = color_bar.makeColourBar(20, 5, self.fig_scale_cb, min=0.5, max=1.5,
                                       label='Scale Factor')
        colours = color_bar.map.mapToQColor(scale_factor)
        y_min, y_max = self.fig_img.viewRange()[1]

        for ir, reg in enumerate(self.scale_data['region']):
            region = pg.LinearRegionItem(values=(reg[0], reg[1]),
                                         orientation=pg.LinearRegionItem.Horizontal,
                                         brush=colours[ir], movable=False)
            bound = pg.InfiniteLine(pos=reg[0], angle=0, pen=colours[ir])

            self.fig_scale.addItem(region)
            self.fig_scale.addItem(bound)
            self.scale_regions = np.vstack([self.scale_regions, region])

            # Add text label showing the scale factor
            text_y = (max(y_min, reg[0]) + min(y_max, reg[1])) / 2  # Center of the region
            text_item = pg.TextItem(text=f"{self.scale_data['scale'][ir]:.2f}",
                                    anchor=(0.5, 0.5),
                                    color='black')
            text_item.setPos(-0.05, text_y)  # Position at minimum x axis
            self.fig_scale.addItem(text_item)

        bound = pg.InfiniteLine(pos=self.scale_data['region'][-1][1], angle=0,
                                pen=colours[-1])

        self.fig_scale.addItem(bound)

        self.fig_scale.setYRange(min=self.probe_tip - self.probe_extra,
                                 max=self.probe_top + self.probe_extra, padding=self.pad)
        self.set_axis(self.fig_scale, 'bottom', pen='w', label='blank')
        self.fig_scale_cb.addItem(cbar)

    def plot_fit(self):
        """
        Plots the scale factor and offset applied to channels along depth of probe track
        relative to orignal position of channels
        """

        # If no histology we can't do alignment
        if not self.histology_exists:
            return

        self.fit_plot.setData(x=self.features[self.idx] * 1e6,
                              y=self.track[self.idx] * 1e6)
        self.fit_scatter.setData(x=self.features[self.idx] * 1e6,
                                 y=self.track[self.idx] * 1e6)

        depth_lin = self.ephysalign.feature2track_lin(self.depth / 1e6, self.features[self.idx],
                                                      self.track[self.idx])
        if np.any(depth_lin):
            self.fit_plot_lin.setData(x=self.depth, y=depth_lin * 1e6)
        else:
            self.fit_plot_lin.setData()

    def plot_slice(self, data, img_type):

        # If no histology we can't do alignment
        if not self.histology_exists:
            return

        self.fig_slice.clear()
        self.slice_chns = []
        self.slice_lines = []
        img = pg.ImageItem()
        img.setImage(data[img_type])
        transform = [data['scale'][0], 0., 0., 0., data['scale'][1], 0., data['offset'][0],
                     data['offset'][1], 1.]
        img.setTransform(QtGui.QTransform(*transform))

        if img_type == 'label':
            self.fig_slice_layout.removeItem(self.slice_item)
            self.fig_slice_layout.addItem(self.fig_slice_hist_alt, 0, 1)
            self.slice_item = self.fig_slice_hist_alt
        else:
            color_bar = ColorBar('cividis')
            lut = color_bar.getColourMap()
            img.setLookupTable(lut)

            self.fig_slice_layout.removeItem(self.slice_item)
            self.fig_slice_hist = pg.HistogramLUTItem()
            self.fig_slice_hist.axis.hide()
            self.fig_slice_hist.setImageItem(img)
            self.fig_slice_hist.gradient.setColorMap(color_bar.map)
            self.fig_slice_hist.autoHistogramRange()
            self.fig_slice_layout.addItem(self.fig_slice_hist, 0, 1)
            hist_levels = self.fig_slice_hist.getLevels()
            hist_val, hist_count = img.getHistogram()
            upper_idx = np.where(hist_count > 10)[0][-1]
            upper_val = hist_val[upper_idx]
            if hist_levels[0] != 0:
                self.fig_slice_hist.setLevels(min=hist_levels[0], max=upper_val)
            self.slice_item = self.fig_slice_hist

        self.fig_slice.addItem(img)
        self.traj_line = pg.PlotCurveItem()

        self.traj_line.setData(x=self.xyz_track[:, 0] * 1e6 / self.loaddata.brain_atlas.spacing, 
                               y=self.xyz_track[:, 2] * 1e6 / self.loaddata.brain_atlas.spacing, pen=self.kpen_solid)
        self.fig_slice.addItem(self.traj_line)
        self.plot_channels()

    def plot_channels(self):

        # If no histology we can't do alignment
        if not self.histology_exists:
            return

        self.channel_status = True
        self.xyz_channels = self.ephysalign.get_channel_locations(self.features[self.idx],
                                                                  self.track[self.idx])

        if not self.slice_chns:
            self.slice_lines = []
            self.slice_chns = pg.ScatterPlotItem()
            self.slice_chns.setData(x=self.xyz_channels[:, 0], y=self.xyz_channels[:, 2], pen='r',
                                    brush='r')
            self.fig_slice.addItem(self.slice_chns)
            track_lines = self.ephysalign.get_perp_vector(self.features[self.idx],
                                                          self.track[self.idx])

            print('Reference lines', track_lines)
            for ref_line in track_lines:
                line = pg.PlotCurveItem()
                line.setData(x=ref_line[:, 0], y=ref_line[:, 2], pen=self.reference_line_kpen)
                self.fig_slice.addItem(line)
                self.slice_lines.append(line)

        else:
            for line in self.slice_lines:
                self.fig_slice.removeItem(line)
            self.slice_lines = []
            track_lines = self.ephysalign.get_perp_vector(self.features[self.idx],
                                                          self.track[self.idx])

            print('Reference lines', track_lines)
            for ref_line in track_lines:
                line = pg.PlotCurveItem()
                line.setData(x=ref_line[:, 0], y=ref_line[:, 2], pen=self.reference_line_kpen)
                self.fig_slice.addItem(line)
                self.slice_lines.append(line)
            self.slice_chns.setData(x=self.xyz_channels[:, 0], y=self.xyz_channels[:, 2], pen='r',
                                    brush='r')

    def plot_scatter(self, data):
        """
        Plots a 2D scatter plot with electrophysiology data
        param data: dictionary of data to plot
            {'x': x coordinate of data, np.array((npoints)), float
             'y': y coordinate of data, np.array((npoints)), float
             'size': size of data points, np.array((npoints)), float
             'colour': colour of data points, np.array((npoints)), QtGui.QColor
             'xrange': range to display of x axis, np.array([min range, max range]), float
             'xaxis': label for xaxis, string
            }
        type data: dict
        """
        if not data:
            print('data for this plot not available')
            return
        else:
            [self.fig_img.removeItem(plot) for plot in self.img_plots]
            [self.fig_img_cb.removeItem(cbar) for cbar in self.img_cbars]

            self.img_plots = []
            self.img_cbars = []

            size = data['size'].tolist()
            symbol = data['symbol'].tolist()

            color_bar = ColorBar(data['cmap'])
            cbar = color_bar.makeColourBar(20, 5, self.fig_img_cb, min=np.min(data['levels'][0]),
                                           max=np.max(data['levels'][1]), label=data['title'])
            self.fig_img_cb.addItem(cbar)
            self.img_cbars.append(cbar)

            brush = data['colours'].tolist()
            plot = pg.ScatterPlotItem()
            plot.setData(x=data['x'], y=data['y'],
                            symbol=symbol, size=size, brush=brush, pen=data['pen'])

            self.fig_img.addItem(plot)
            self.fig_img.setXRange(min=data['xrange'][0], max=data['xrange'][1],
                                   padding=0)
            self.fig_img.setYRange(min=self.probe_tip - self.probe_extra,
                                   max=self.probe_top + self.probe_extra, padding=self.pad)
            self.set_axis(self.fig_img, 'bottom', label=data['xaxis'])
            self.y_scale = 1
            self.img_plots.append(plot)
            self.data_plot = plot
            self.xrange = data['xrange']

            if data['cluster']:
                self.data = data['x']
                self.data_plot.sigClicked.connect(self.cluster_clicked)

    def plot_line(self, data):
        """
        Plots a 1D line plot with electrophysiology data
        param data: dictionary of data to plot
            {'x': x coordinate of data, np.array((npoints)), float
             'y': y coordinate of data, np.array((npoints)), float
             'xrange': range to display of x axis, np.array([min range, max range]), float
             'xaxis': label for xaxis, string
            }
        type data: dict
        """
        if not data:
            print('data for this plot not available')
            return
        else:
            [self.fig_line.removeItem(plot) for plot in self.line_plots]
            self.line_plots = []
            line = pg.PlotCurveItem()
            line.setData(x=data['x'], y=data['y'])
            line.setPen(self.kpen_solid)
            self.fig_line.addItem(line)
            self.fig_line.setXRange(min=data['xrange'][0], max=data['xrange'][1], padding=0)
            self.fig_line.setYRange(min=self.probe_tip - self.probe_extra,
                                    max=self.probe_top + self.probe_extra, padding=self.pad)
            self.set_axis(self.fig_line, 'bottom', label=data['xaxis'])
            self.line_plots.append(line)

    def plot_probe(self, data, bounds=None):
        """
        Plots a 2D image with probe geometry
        param data: dictionary of data to plot
            {'img': image data for each channel bank, list of np.array((1,ny)), list
             'scale': scaling to apply to each image, list of np.array([xscale,yscale]), list
             'offset': offset to apply to each image, list of np.array([xoffset,yoffset]), list
             'levels': colourbar extremes np.array([min val, max val]), float
             'cmap': colourmap to use, string
             'xrange': range to display of x axis, np.array([min range, max range]), float
             'title': description to place on colorbar, string
            }
        type data: dict
        """
        if not data:
            print('data for this plot not available')
            return
        else:
            [self.fig_probe.removeItem(plot) for plot in self.probe_plots]
            [self.fig_probe_cb.removeItem(cbar) for cbar in self.probe_cbars]
            [self.fig_probe.removeItem(line) for line in self.probe_bounds]
            self.set_axis(self.fig_probe_cb, 'top', pen='w')
            self.probe_plots = []
            self.probe_cbars = []
            self.probe_bounds = []
            color_bar = ColorBar(data['cmap'])
            lut = color_bar.getColourMap()
            for img, scale, offset in zip(data['img'], data['scale'], data['offset']):
                image = pg.ImageItem()
                image.setImage(img)
                transform = [scale[0], 0., 0., 0., scale[1], 0., offset[0],
                             offset[1], 1.]
                image.setTransform(QtGui.QTransform(*transform))
                image.setLookupTable(lut)
                image.setLevels((data['levels'][0], data['levels'][1]))
                self.fig_probe.addItem(image)
                self.probe_plots.append(image)

            cbar = color_bar.makeColourBar(20, 5, self.fig_probe_cb, min=data['levels'][0],
                                           max=data['levels'][1], label=data['title'], lim=True)
            self.fig_probe_cb.addItem(cbar)
            self.probe_cbars.append(cbar)

            self.fig_probe.setXRange(min=data['xrange'][0], max=data['xrange'][1], padding=0)
            self.fig_probe.setYRange(min=self.probe_tip - self.probe_extra,
                                     max=self.probe_top + self.probe_extra, padding=self.pad)
            # so stupid!!!!!
            self.set_axis(self.fig_probe, 'bottom', pen='w', label='blank')
            if bounds is not None:
                # add some infinite line stuff
                for bound in bounds:
                    line = pg.InfiniteLine(pos=bound, angle=0, pen='w')
                    self.fig_probe.addItem(line)
                    self.probe_bounds.append(line)

    def plot_image(self, data):
        """
        Plots a 2D image with with electrophysiology data
        param data: dictionary of data to plot
            {'img': image data, np.array((nx,ny)), float
             'scale': scaling to apply to each axis, np.array([xscale,yscale]), float
             'levels': colourbar extremes np.array([min val, max val]), float
             'offset': offset to apply to each image, np.array([xoffset,yoffset]), float
             'cmap': colourmap to use, string
             'xrange': range to display of x axis, np.array([min range, max range]), float
             'xaxis': label for xaxis, string
             'title': description to place on colorbar, string
            }
        type data: dict
        """
        if not data:
            print('data for this plot not available')
            return
        else:
            [self.fig_img.removeItem(plot) for plot in self.img_plots]
            [self.fig_img_cb.removeItem(cbar) for cbar in self.img_cbars]
            self.set_axis(self.fig_img_cb, 'top', pen='w')
            self.img_plots = []
            self.img_cbars = []

            image = pg.ImageItem()
            image.setImage(data['img'])
            transform = [data['scale'][0], 0., 0., 0., data['scale'][1], 0., data['offset'][0],
                         data['offset'][1], 1.]
            image.setTransform(QtGui.QTransform(*transform))
            cmap = data.get('cmap', [])
            if cmap:
                color_bar = ColorBar(data['cmap'])
                lut = color_bar.getColourMap()
                image.setLookupTable(lut)
                image.setLevels((data['levels'][0], data['levels'][1]))
                cbar = color_bar.makeColourBar(20, 5, self.fig_img_cb, min=data['levels'][0],
                                               max=data['levels'][1], label=data['title'])
                self.fig_img_cb.addItem(cbar)
                self.img_cbars.append(cbar)
            else:
                image.setLevels((1, 0))

            self.fig_img.addItem(image)
            self.img_plots.append(image)
            self.fig_img.setXRange(min=data['xrange'][0], max=data['xrange'][1], padding=0)
            self.fig_img.setYRange(min=self.probe_tip - self.probe_extra,
                                   max=self.probe_top + self.probe_extra, padding=self.pad)
            # TODO need to make this work, at the moment messes things up!
            # self.fig_img.setLimits(xMin=data['xrange'][0], xMax=data['xrange'][1])
            #                        yMin=self.probe_tip - self.probe_extra - self.pad,
            #                        yMax=self.probe_top + self.probe_extra + self.pad)
            self.set_axis(self.fig_img, 'bottom', label=data['xaxis'])
            self.y_scale = data['scale'][1]
            self.x_scale = data['scale'][0]
            self.data_plot = image
            self.xrange = data['xrange']


    """
    Interaction functions
    """
    def _update_ephys_alignments(self, folder_path: Path, skip_shanks=False):
        if Path('/data').is_dir():
            data_string =  f"{folder_path.parent.parent.stem}/{folder_path.parent.stem}/{folder_path.stem}"
            input_data_path = tuple(Path('/data').glob(f"*/{data_string}"))[0]
        else:
            input_data_path = folder_path

        if hasattr(self, 'current_shank_idx'):
            self.prev_alignments, shank_options = self.loaddata.get_info(folder_path, shank_idx=self.current_shank_idx, input_path=input_data_path, skip_shanks=skip_shanks)
        else:
            self.prev_alignments, shank_options = self.loaddata.get_info(folder_path, shank_idx=0, input_path=input_data_path, skip_shanks=skip_shanks)

        if hasattr(self, 'current_shank_idx'):
            self.feature_prev, self.track_prev = self.loaddata.get_starting_alignment(0, shank_idx=self.current_shank_idx, folder_path=folder_path)
        else: 
            self.feature_prev, self.track_prev = self.loaddata.get_starting_alignment(0, folder_path=folder_path)

        if shank_options is not None:
            self.populate_lists(shank_options, self.shank_list, self.shank_combobox)

        #if shank_options is None:
        #    self.data_status = True
        
        if not hasattr(self, 'current_shank_idx'):
            self.current_shank_idx = 0

        print('Input data path', input_data_path)
        self.data_button_pressed(input_data_path)
        print('Feature prev', self.feature_prev)

    def load_existing_alignments(self):
        folder_path = Path(QtWidgets.QFileDialog.getExistingDirectory(None, "Load Existing Alignments"))
        self.reload_folder_line.setText(str(folder_path))
            
        self._update_ephys_alignments(folder_path)

    def on_folder_selected(self):
        """
        Triggered in offline mode when folder button is clicked
        """
        self.data_status = False
        #folder_path = Path(QtWidgets.QFileDialog.getExistingDirectory(None, "Select Input Directory"))
        
        if Path('/data/').is_dir():
            # Default For code ocean.
            we_are_in_code_ocean = True
            folder_path = Path(QtWidgets.QFileDialog.getExistingDirectory(None, "Select Input Directory", directory='/data/'))
        else:
            # If not code ocean, will default to current directory
            we_are_in_code_ocean = False
            folder_path = Path(QtWidgets.QFileDialog.getExistingDirectory(None, "Select Input Directory"))
        # Set the output default based on the selected folder path
        if we_are_in_code_ocean:
            out_folder = Path('/results/').joinpath(folder_path.parent.stem)
            string_folder_path = folder_path.parent.stem
            subject_id_path = string_folder_path[string_folder_path.index('_')+1:]
            subject_id = subject_id_path[0:subject_id_path.index('_')]

            out_folder = Path('/results').joinpath(subject_id)
            print('Output folder', out_folder)
            
        else:
            out_folder = folder_path.parent/'out'
            
        self.input_path = folder_path

        # Make it compatible with path outside of code ocean
        self.loaddata.data_root = folder_path.parents[3]
        self.data_root = self.loaddata.data_root

        # Create the output folder if it doesn't exist
        os.makedirs(out_folder, exist_ok=True)
        # Set the output directory based on input name.
        self.output_directory = out_folder/folder_path.parent.stem/folder_path.stem
        print('Output dir', self.output_directory)
        self.loaddata.output_directory = self.output_directory
        self.output_folder_line.setText(str(self.output_directory))

        if folder_path:
            self.input_folder_line.setText(str(folder_path))
            self._update_ephys_alignments(folder_path)
            return True
        else:
            return False

    def on_histology_folder_selected(self):
        """
        Triggered in offline mode when folder button is clicked
        """
        self.data_status = False
        folder_path = Path(QtWidgets.QFileDialog.getExistingDirectory(None, "Select Histology Directory"))

        if folder_path:
            # self.histology_folder_line.setText(str(folder_path))
            self.loaddata.histology_path = folder_path
            if self.histology_exists:
                self.slice_data, self.fp_slice_data = self.loaddata.get_slice_images(self.ephysalign.xyz_samples)
            try:
                self.data_button_pressed()
            except TypeError:
                pass
            return True
        else:
            return False
        


    def on_output_folder_selected(self):
        """
        Triggered in offline mode when folder button is clicked
        """
        folder_path = Path(QtWidgets.QFileDialog.getExistingDirectory(None, "Select Output Directory"))

        if folder_path:
            self.output_folder_line.setText(str(folder_path))
            self.loaddata.output_directory = folder_path
            return True
        else:
            return False
        
    # def on_load_existing_button_pressed(self):
    #     if self.loaddata.output_directory is not None:
    #         folder_path = Path(QtWidgets.QFileDialog.getExistingDirectory(None, "Select Annotation Directory"),self.loaddata.output_directory)
    #     else:
    #         folder_path = Path(QtWidgets.QFileDialog.getExistingDirectory(None, "Select Annotation Directory"))
    #     self.prev_alignments, shank_options = self.loaddata.get_previous_info(folder_path)
    #     self.populate_lists(shank_options, self.shank_list, self.shank_combobox)

        
    def on_shank_selected(self, idx):
        """
        Triggered in offline mode for selecting shank when using NP2.0
        """
        #self.data_status = False
        # Update prev_alignments
        shank_text = self.shank_combobox.currentText()
        shank_id = int(shank_text.split('/')[0])
        self.current_shank_idx = shank_id - 1
        self.feature_prev, self.track_prev = self.loaddata.get_starting_alignment(0, shank_idx=self.current_shank_idx, folder_path=self.output_directory)

        if self.data_status:
            #if self.current_shank_idx != 0:
            self.data_button_pressed(self.input_path, load_new_shank=True)

    def on_alignment_selected(self, idx):
        self.feature_prev, self.track_prev = self.loaddata.get_starting_alignment(idx)

    def data_button_pressed(self, folder_path: Path, load_new_shank: bool = False):
        """
        Triggered when Get Data button pressed, uses subject and session info to find eid and
        downloads and computes data needed for GUI display
        """

        # Clear all plots from previous session
        [self.fig_img.removeItem(plot) for plot in self.img_plots]
        [self.fig_img.removeItem(cbar) for cbar in self.img_cbars]
        [self.fig_line.removeItem(plot) for plot in self.line_plots]
        [self.fig_probe.removeItem(plot) for plot in self.probe_plots]
        [self.fig_probe.removeItem(cbar) for cbar in self.probe_cbars]
        self.fig_slice.clear()
        self.fig_hist.clear()
        self.ax_hist.setTicks([])
        self.fig_hist_ref.clear()
        self.ax_hist_ref.setTicks([])
        self.fig_scale.clear()
        self.fit_plot.setData()
        self.fit_scatter.setData()
        self.remove_lines_points()
        self.init_variables()

        # Only run once
        if not self.data_status:
            self.probe_path, self.chn_depths, self.sess_notes, data = \
                self.loaddata.get_data()
            self.data = data
            if not self.probe_path:
                return
        if self.data_status and load_new_shank:
            self.probe_path, self.chn_depths, self.sess_notes, data = \
                self.loaddata.get_data(reload_data=False)
            self.data = data
            if not self.probe_path:
                return

        # Only get histology specific stuff if the histology tracing exists
        if self.histology_exists:
            self.xyz_picks = self.loaddata.get_xyzpicks(folder_path, self.current_shank_idx)

            if np.any(self.feature_prev):
                self.ephysalign = EphysAlignment(self.xyz_picks, self.chn_depths,
                                                 track_prev=self.track_prev,
                                                 feature_prev=self.feature_prev,
                                                 brain_atlas=self.loaddata.brain_atlas)
            else:
                self.ephysalign = EphysAlignment(self.xyz_picks, self.chn_depths,
                                                 brain_atlas=self.loaddata.brain_atlas)

            self.region_fp, self.region_label_fp, self.region_colour_fp, _ \
                = EphysAlignment.get_histology_regions(self.ephysalign.xyz_samples, self.ephysalign.sampling_trk,
                                                       self.loaddata.brain_atlas)

            self.features[self.idx], self.track[self.idx], self.xyz_track \
                = self.ephysalign.get_track_and_feature()

            self.get_scaled_histology()
        # If we have not loaded in the data before then we load eveything we need
        if not self.data_status or load_new_shank:
            print('Shank index', self.current_shank_idx)
            self.plotdata = pd.PlotData(self.probe_path, self.data,
                                        self.current_shank_idx)
            self.set_lims(np.min([0, self.plotdata.chn_min]), self.plotdata.chn_max)
            self.scat_drift_data = self.plotdata.get_depth_data_scatter()
            (self.scat_fr_data, self.scat_p2t_data,
             self.scat_amp_data) = self.plotdata.get_fr_p2t_data_scatter()
            self.img_spike_corr_data = self.plotdata.get_spike_correlation_data_img()
            self.img_fr_data = self.plotdata.get_fr_img()
            self.img_rms_APdata, self.probe_rms_APdata = self.plotdata.get_rms_data_img_probe('AP')
            self.img_rms_LFPdata, self.probe_rms_LFPdata = self.plotdata.get_rms_data_img_probe(
                'LF')
            self.img_rms_APdata_main, self.probe_rms_APdata_main = self.plotdata.get_rms_data_img_probe('AP_main')
            self.img_rms_LFPdata_main, self.probe_rms_LFPdata_main = self.plotdata.get_rms_data_img_probe('LF_main')
            self.img_lfp_data, self.probe_lfp_data = self.plotdata.get_lfp_spectrum_data('lf')
            self.img_lfp_data_main, self.probe_lfp_data_main = self.plotdata.get_lfp_spectrum_data('lf_main')

            self.img_lfp_corr_data = self.plotdata.get_lfp_correlation_data_img()

            self.line_fr_data, self.line_amp_data = self.plotdata.get_fr_amp_data_line()
            self.probe_rfmap, self.rfmap_boundaries = self.plotdata.get_rfmap_data()
            self.img_stim_data = self.plotdata.get_passive_events()
            if not self.offline:
                self.img_raw_data = self.plotdata.get_raw_data_image(self.loaddata.probe_id, one=self.loaddata.one)
            else:
                self.img_raw_data = {}
            if self.histology_exists:
                self.slice_data, self.fp_slice_data = self.loaddata.get_slice_images(self.ephysalign.xyz_samples)
            else:
                # probably need to return an empty array of things
                self.slice_data = {}
                self.fp_slice_data = None

            self.data_status = True
            self.init_menubar()
        else:
            self.set_lims(np.min([0, self.plotdata.chn_min]), self.plotdata.chn_max)

        # Initialise checked plots
        self.img_init.setChecked(True)
        self.line_init.setChecked(True)
        self.probe_init.setChecked(True)
        self.unit_init.setChecked(True)
        self.slice_init.setChecked(True)

        # Initialise ephys plots
        self.plot_image(self.img_fr_data)
        self.plot_probe(self.probe_rms_LFPdata)
        self.plot_line(self.line_fr_data)

        # Initialise histology plots
        self.plot_histology_ref(self.fig_hist_ref)
        self.plot_histology(self.fig_hist)
        self.label_status = False
        self.toggle_labels_button_pressed()
        self.plot_scale_factor()
        if np.any(self.feature_prev):
            self.create_lines(self.feature_prev[1:-1] * 1e6)
        # Initialise slice and fit images
        self.plot_fit()
        self.plot_slice(self.slice_data, 'ccf')

        # Only configure the view the first time the GUI is launched
        self.set_view(view=1, configure=self.configure)
        self.configure = False

    def compute_nearby_boundaries(self):

        # If no histology we can't plot histology
        if not self.histology_exists:
            return

        nearby_bounds = self.ephysalign.get_nearest_boundary(self.ephysalign.xyz_samples,
                                                             self.allen, steps=6,
                                                             brain_atlas=self.loaddata.brain_atlas)
        [self.hist_nearby_x, self.hist_nearby_y,
         self.hist_nearby_col] = self.ephysalign.arrange_into_regions(
            self.ephysalign.sampling_trk, nearby_bounds['id'], nearby_bounds['dist'],
            nearby_bounds['col'])

        [self.hist_nearby_parent_x,
         self.hist_nearby_parent_y,
         self.hist_nearby_parent_col] = self.ephysalign.arrange_into_regions(
            self.ephysalign.sampling_trk, nearby_bounds['parent_id'], nearby_bounds['parent_dist'],
            nearby_bounds['parent_col'])

    def toggle_histology_button_pressed(self):
        self.hist_bound_status = not self.hist_bound_status

        if not self.hist_bound_status:
            if self.hist_nearby_x is None:
                self.compute_nearby_boundaries()

            self.plot_histology_nearby(self.fig_hist_ref)
        else:
            self.plot_histology_ref(self.fig_hist_ref)

    def toggle_histology_map_button_pressed(self):

        if self.hist_mapping == 'Allen':
            self.hist_mapping = 'FP'
        else:
            self.hist_mapping = 'Allen'

        self.get_scaled_histology()
        self.plot_histology(self.fig_hist)
        self.plot_histology_ref(self.fig_hist_ref)
        self.remove_lines_points()
        self.add_lines_points()

    def _on_img_action_triggered(self, action):
        self.current_img_action = action
    
    def _on_line_action_triggered(self, action):
        self.line_img_action = action
    
    def _on_probe_action_triggered(self, action):
        self.probe_img_action = action
    
    def update_plot(self):
        """Re-run the plotting function for the current menu selection."""
        if self.current_img_action is not None:
            # directly invoke the same slot as if the user clicked
            self.current_img_action.trigger()
        
        if self.line_img_action is not None:
            # directly invoke the same slot as if the user clicked
            self.line_img_action.trigger()

        if self.probe_img_action is not None:
            # directly invoke the same slot as if the user clicked
            self.probe_img_action.trigger()


    def filter_unit_pressed(self, type):
        self.plotdata.filter_units(type)
        self.scat_drift_data = self.plotdata.get_depth_data_scatter()
        (self.scat_fr_data, self.scat_p2t_data,
         self.scat_amp_data) = self.plotdata.get_fr_p2t_data_scatter()
        self.img_spike_corr_data = self.plotdata.get_spike_correlation_data_img()
        self.img_fr_data = self.plotdata.get_fr_img()
        self.line_fr_data, self.line_amp_data = self.plotdata.get_fr_amp_data_line()
        self.probe_rfmap, self.rfmap_boundaries = self.plotdata.get_rfmap_data()
        self.img_stim_data = self.plotdata.get_passive_events()
        self.img_init.setChecked(True)
        self.line_init.setChecked(True)
        self.probe_init.setChecked(True)

        self.update_plot()

    def fit_button_pressed(self):
        """
        Triggered when fit button or Enter key pressed, applies scaling factor to brain regions
        according to locations of reference lines on ephys and histology plots. Updates all plots
        and indices after scaling has been applied
        """

        # If no histology we can't plot histology
        if not self.histology_exists:
            return

        # Use a cyclic buffer of length self.max_idx to hold information about previous moves,
        # when a new move is initiated ensures indexes are all correct so user can only access
        # fixed number of previous or next moves
        if self.current_idx < self.last_idx:
            self.total_idx = np.copy(self.current_idx)
            self.diff_idx = (np.mod(self.last_idx, self.max_idx) - np.mod(self.total_idx,
                                                                          self.max_idx))
            if self.diff_idx >= 0:
                self.diff_idx = self.max_idx - self.diff_idx
            else:
                self.diff_idx = np.abs(self.diff_idx)
        else:
            self.diff_idx = self.max_idx - 1

        self.total_idx += 1
        self.current_idx += 1
        self.idx_prev = np.copy(self.idx)
        self.idx = np.mod(self.current_idx, self.max_idx)
        self.scale_hist_data()
        self.plot_histology(self.fig_hist)
        self.plot_scale_factor()
        self.plot_fit()
        self.plot_channels()
        self.remove_lines_points()
        self.add_lines_points()
        self.update_lines_points()
        self.fig_hist.setYRange(min=self.probe_tip - self.probe_extra,
                                max=self.probe_top + self.probe_extra, padding=self.pad)
        self.update_string()

    def offset_button_pressed(self):
        """
        Triggered when offset button or o key pressed, applies offset to brain regions according to
        locations of probe tip line on histology plot. Updates all plots and indices after offset
        has been applied
        """

        # If no histology we can't plot histology
        if not self.histology_exists:
            return

        if self.current_idx < self.last_idx:
            self.total_idx = np.copy(self.current_idx)
            self.diff_idx = (np.mod(self.last_idx, self.max_idx) - np.mod(self.total_idx,
                                                                          self.max_idx))
            if self.diff_idx >= 0:
                self.diff_idx = self.max_idx - self.diff_idx
            else:
                self.diff_idx = np.abs(self.diff_idx)
        else:
            self.diff_idx = self.max_idx - 1

        self.total_idx += 1
        self.current_idx += 1
        self.idx_prev = np.copy(self.idx)
        self.idx = np.mod(self.current_idx, self.max_idx)
        self.offset_hist_data()
        self.plot_histology(self.fig_hist)
        self.plot_scale_factor()
        self.plot_fit()
        self.plot_channels()
        self.remove_lines_points()
        self.add_lines_points()
        self.update_lines_points()
        self.fig_hist.setYRange(min=self.probe_tip - self.probe_extra,
                                max=self.probe_top + self.probe_extra, padding=self.pad)
        self.update_string()

    def movedown_button_pressed(self):
        """
        Triggered when Shift+down key pressed. Moves probe tip down by 50um and offsets data
        """
        # If no histology we can't plot histology
        if not self.histology_exists:
            return

        if self.track[self.idx][-1] - 50 / 1e6 >= np.max(self.chn_depths) / 1e6:
            self.track[self.idx] -= 50 / 1e6
            self.offset_button_pressed()


    def moveup_button_pressed(self):
        """
        Triggered when Shift+down key pressed. Moves probe tip up by 50um and offsets data
        """
        # If no histology we can't plot histology
        if not self.histology_exists:
            return

        if self.track[self.idx][0] + 50 / 1e6 <= np.min(self.chn_depths) / 1e6:
            self.track[self.idx] += 50 / 1e6
            self.offset_button_pressed()

    def toggle_labels_button_pressed(self):
        """
        Triggered when Shift+A key pressed. Shows/hides labels Allen atlas labels on brain regions
        in histology plots
        """
        self.label_status = not self.label_status
        if not self.label_status:
            self.ax_hist_ref.setPen(None)
            self.ax_hist_ref.setTextPen(None)
            self.ax_hist.setPen(None)
            self.ax_hist.setTextPen(None)
            self.fig_hist_ref.update()
            self.fig_hist.update()

        else:
            self.ax_hist_ref.setPen('k')
            self.ax_hist_ref.setTextPen('k')
            self.ax_hist.setPen('k')
            self.ax_hist.setTextPen('k')
            self.fig_hist_ref.update()
            self.fig_hist.update()

    def toggle_line_button_pressed(self):
        """
        Triggered when Shift+L key pressed. Shows/hides reference lines on ephys and histology
        plots
        """
        self.line_status = not self.line_status
        if not self.line_status:
            self.remove_lines_points()
        else:
            self.add_lines_points()

    def toggle_channel_button_pressed(self):
        """
        Triggered when Shift+C key pressed. Shows/hides channels and trajectory on slice image
        """
        # If no histology we can't plot histology
        if not self.histology_exists:
            return

        self.channel_status = not self.channel_status
        if not self.channel_status:
            self.fig_slice.removeItem(self.traj_line)
            self.fig_slice.removeItem(self.slice_chns)
            for line in self.slice_lines:
                self.fig_slice.removeItem(line)

        else:
            self.fig_slice.addItem(self.traj_line)
            self.fig_slice.addItem(self.slice_chns)
            for line in self.slice_lines:
                self.fig_slice.addItem(line)

    def delete_line_button_pressed(self):
        """
        Triggered when mouse hovers over a reference line and shift+D keys are pressed. 
        Deletes a reference line from the ephys and histology plots
        """

        if self.selected_line:
            line_idx = np.where(self.lines_features == self.selected_line)[0]
            if line_idx.size == 0:
                line_idx = np.where(self.lines_tracks == self.selected_line)[0]
            line_idx = line_idx[0]

            self.fig_img.removeItem(self.lines_features[line_idx][0])
            self.fig_line.removeItem(self.lines_features[line_idx][1])
            self.fig_probe.removeItem(self.lines_features[line_idx][2])
            self.fig_hist.removeItem(self.lines_tracks[line_idx, 0])
            self.fig_fit.removeItem(self.points[line_idx, 0])
            self.lines_features = np.delete(self.lines_features, line_idx, axis=0)
            self.lines_tracks = np.delete(self.lines_tracks, line_idx, axis=0)
            self.points = np.delete(self.points, line_idx, axis=0)

    def describe_labels_pressed(self):

        # if no histology don't show
        if not self.histology_exists:
            return

        if self.selected_region:
            idx = np.where(self.hist_regions == self.selected_region)[0]
            if not np.any(idx):
                idx = np.where(self.hist_ref_regions == self.selected_region)[0]
            if not np.any(idx):
                idx = np.array([0])

            description, lookup = self.loaddata.get_region_description(
                self.ephysalign.region_id[idx[0]][0])
            item = self.struct_list.findItems(lookup, flags=QtCore.Qt.MatchRecursive)
            model_item = self.struct_list.indexFromItem(item[0])
            self.struct_view.collapseAll()
            self.struct_view.scrollTo(model_item)
            self.struct_view.setCurrentIndex(model_item)
            self.struct_description.setText(description)

            if not self.label_popup:
                self.label_win = ephys_gui.PopupWindow(title='Structure Information',
                                                       size=(500, 700), graphics=False)
                self.label_win.layout.addWidget(self.struct_view)
                self.label_win.layout.addWidget(self.struct_description)
                self.label_win.layout.setRowStretch(0, 7)
                self.label_win.layout.setRowStretch(1, 3)
                self.label_popup.append(self.label_win)
                self.label_win.closed.connect(self.label_closed)
                self.label_win.moved.connect(self.label_moved)
                self.activateWindow()
            else:
                self.label_win.show()
                self.activateWindow()

    def label_closed(self, popup):
        self.label_win.hide()

    def label_moved(self):
        self.activateWindow()

    def label_pressed(self, item):
        idx = int(item.model().itemFromIndex(item).accessibleText())
        description, lookup = self.loaddata.get_region_description(idx)
        item = self.struct_list.findItems(lookup,
                                          flags=QtCore.Qt.MatchRecursive)
        model_item = self.struct_list.indexFromItem(item[0])
        self.struct_view.setCurrentIndex(model_item)
        self.struct_description.setText(description)

    def next_button_pressed(self):
        """
        Triggered when right key pressed. Updates all plots and indices with next move. Ensures
        user cannot go past latest move
        """
        # If no histology we can't plot histology
        if not self.histology_exists:
            return

        if (self.current_idx < self.total_idx) & (self.current_idx >
                                                  self.total_idx - self.max_idx):
            self.current_idx += 1
            self.idx = np.mod(self.current_idx, self.max_idx)
            self.remove_lines_points()
            self.add_lines_points()
            self.get_scaled_histology()
            self.plot_histology(self.fig_hist)
            self.plot_scale_factor()
            self.remove_lines_points()
            self.add_lines_points()
            self.plot_fit()
            self.plot_channels()
            self.update_string()

    def prev_button_pressed(self):
        """
        Triggered when left key pressed. Updates all plots and indices with previous move.
        Ensures user cannot go back more than self.max_idx moves
        """

        # If no histology we can't plot histology
        if not self.histology_exists:
            return

        if self.total_idx > self.last_idx:
            self.last_idx = np.copy(self.total_idx)

        if (self.current_idx > np.max([0, self.total_idx - self.diff_idx])):
            self.current_idx -= 1
            self.idx = np.mod(self.current_idx, self.max_idx)
            self.remove_lines_points()
            self.add_lines_points()
            self.get_scaled_histology()
            self.plot_histology(self.fig_hist)
            self.plot_scale_factor()
            self.remove_lines_points()
            self.add_lines_points()
            self.plot_fit()
            self.plot_channels()
            self.update_string()

    def reset_button_pressed(self):
        """
        Triggered when reset button or Shift+R key pressed. Resets channel locations to orignal
        location
        """
        # If no histology we can't plot histology
        if not self.histology_exists:
            return

        self.remove_lines_points()
        self.lines_features = np.empty((0, 3))
        self.lines_tracks = np.empty((0, 1))
        self.points = np.empty((0, 1))
        if self.current_idx < self.last_idx:
            self.total_idx = np.copy(self.current_idx)
            self.diff_idx = (np.mod(self.last_idx, self.max_idx) - np.mod(self.total_idx,
                                                                          self.max_idx))
            if self.diff_idx >= 0:
                self.diff_idx = self.max_idx - self.diff_idx
            else:
                self.diff_idx = np.abs(self.diff_idx)
        else:
            self.diff_idx = self.max_idx - 1

        self.total_idx += 1
        self.current_idx += 1
        self.idx = np.mod(self.current_idx, self.max_idx)
        self.track[self.idx] = np.copy(self.ephysalign.track_init)
        self.features[self.idx] = np.copy(self.ephysalign.feature_init)

        self.get_scaled_histology()

        self.plot_histology(self.fig_hist)
        self.plot_scale_factor()
        if np.any(self.feature_prev):
            self.create_lines(self.feature_prev[1:-1] * 1e6)
        self.plot_fit()
        self.plot_channels()
        self.fig_hist.setYRange(min=self.probe_tip - self.probe_extra,
                                max=self.probe_top + self.probe_extra, padding=self.pad)
        self.update_string()

    def _transform_to_ccf(self, image_physical_space_coordinates: npt.NDArray) -> pandas.DataFrame:
        this_probe_df = pandas.DataFrame({'x': image_physical_space_coordinates[:, 0], 
                                      'y': image_physical_space_coordinates[:, 1], 
                                      'z': image_physical_space_coordinates[:, 2]})
        
        print("Loading transforms from stitched smartspim asset ...")
        subject_id = self.input_path.parent.parent.stem
        smartspim_template_affine_transform = tuple(self.data_root.glob(f'SmartSPIM_{subject_id}*/image_atlas_alignment/*/ls_to_template_SyN_0GenericAffine.mat'))
        if not smartspim_template_affine_transform:
            # try legacy way
            smartspim_template_affine_transform = tuple(self.data_root.glob(f'SmartSPIM_{subject_id}*/registration/ls_to_template_SyN_0GenericAffine.mat'))
            if not smartspim_template_affine_transform:
                raise FileNotFoundError('No affine transform from spim to template. Check attached assets')

        smartspim_template_warp_transform = tuple(self.data_root.glob(f'SmartSPIM_{subject_id}*/image_atlas_alignment/*/ls_to_template_SyN_1InverseWarp.nii.gz'))
        if not smartspim_template_warp_transform:
            smartspim_template_warp_transform = tuple(self.data_root.glob(f'SmartSPIM_{subject_id}*/registration/ls_to_template_SyN_1InverseWarp.nii.gz'))
            if not smartspim_template_warp_transform:
                raise FileNotFoundError('No warp transform from spim to template. Check attached assets')
        
        probe_template = ants.apply_transforms_to_points(ANTS_DIMENSION, this_probe_df, 
                                    [smartspim_template_affine_transform[0].as_posix(),
                                    smartspim_template_warp_transform[0].as_posix()],
                                    whichtoinvert=[True, False])

        template_to_ccf_affine_transform = tuple(self.data_root.glob('spim_template_to_ccf/syn_0GenericAffine.mat'))
        if not template_to_ccf_affine_transform:
            raise FileNotFoundError('No affine transform from template to ccf. Check attached assets')
        
        template_to_ccf_warp_transform = tuple(self.data_root.glob('spim_template_to_ccf/syn_1InverseWarp.nii.gz'))
        if not template_to_ccf_warp_transform:
            raise FileNotFoundError('No warp transform from template to ccf. Check attached assets')
        
        print("applying transforms ...")
        probe_ccf: pandas.DataFrame = ants.apply_transforms_to_points(ANTS_DIMENSION, probe_template, 
                                            [template_to_ccf_affine_transform[0].as_posix(),
                                            template_to_ccf_warp_transform[0].as_posix()],
                                            whichtoinvert=[True, False])
        
        return probe_ccf

    def run_complete_button_in_thread(self):
        self.thread = QThread()
        self.worker = Worker(self.complete_button_pressed_offline)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)           # Gracefully stop thread
        self.worker.finished.connect(self.worker.deleteLater)    # Clean up worker
        self.thread.finished.connect(self.thread.deleteLater)    # Clean up thread

    def complete_button_pressed_offline(self):
        """
        Triggered when save button or Shift+S keys are pressed. 
        Saves final channel locations to a JSON file
        """
        if not self.loaddata.output_directory:
            if not self.on_output_folder_selected():
                print("Channels locations not saved")
                return

        print("Warping to ccf and saving output files to results folder and docdb")
        self.loaddata.upload_data(self.features[self.idx], self.track[self.idx],
                                    self.xyz_channels, self.current_shank_idx + 1)
        self.loaddata.get_starting_alignment(0)
        
        if self.loaddata.n_shanks > 1:
            with open(self.output_directory / f'channel_locations_shank{self.current_shank_idx + 1}.json', 'r') as f:
                channel_results = json.load(f)
            
            with open(self.output_directory / f'prev_alignments_shank{self.current_shank_idx + 1}.json', 'r') as f:
                prev_alignments = json.load(f)
        else:
            with open(self.output_directory / 'channel_locations.json', 'r') as f:
                channel_results = json.load(f)
            with open(self.output_directory / 'prev_alignments.json', 'r') as f:
                prev_alignments = json.load(f)
        
        channel_ids = [channel for channel in tuple(channel_results.keys()) if 'channel' in channel]
        channel_coords = np.zeros((len(channel_ids), 3), dtype=int)
        for i in range(len(channel_ids)):
            channel_coords[i] = (self.loaddata.brain_atlas.image.shape[0] - int(np.round(channel_results[channel_ids[i]]['x'])), 
                                int(np.round(channel_results[channel_ids[i]]['y'])), 
                                self.loaddata.brain_atlas.image.shape[2] - int(np.round(channel_results[channel_ids[i]]['z'])))
        
        ants_physical_points = []
        for point in channel_coords:
            ants_physical_points.append(self.loaddata.brain_atlas.original_image.TransformIndexToPhysicalPoint(point.tolist()))

        ants_physical_points_array = np.array(ants_physical_points)
        ccf_coordinates_dataframe = self._transform_to_ccf(ants_physical_points_array)
        ccf_result_json = {}

        for channel in self.loaddata.channel_dict:
            if channel == 'origin':
                continue

            channel_dict_info = self.loaddata.channel_dict[channel]

            channel_index = int(channel[channel.index('_')+1:])
            ccf_channel_info = ccf_coordinates_dataframe.iloc[channel_index]
            ccf_result_json[channel] = {
                "x": ccf_channel_info['x'].astype(np.float64),
                "y": ccf_channel_info['y'].astype(np.float64),
                "z": ccf_channel_info['z'].astype(np.float64),
                "axial": channel_dict_info['axial'],
                "lateral": channel_dict_info['lateral'],
                "brain_region_id": channel_dict_info['brain_region_id'],
                "brain_region": channel_dict_info['brain_region']
            }
        
        if self.loaddata.n_shanks > 1:
            with open(self.output_directory / f'ccf_channel_locations_shank{self.current_shank_idx + 1}.json', "w") as f:
                json.dump(ccf_result_json, f, indent=2, separators=(",", ": "))
        else:
            with open(self.output_directory / 'ccf_channel_locations.json', "w") as f:
                json.dump(ccf_result_json, f, indent=2, separators=(",", ": "))


        probe_name_for_docdb = f'{self.output_directory.stem}_{self.current_shank_idx}'

        try:
            write_output_to_docdb(self.output_directory.parent.stem, probe_name_for_docdb, 
                                channel_results, prev_alignments, ccf_result_json)
        except ValueError as e:
            print(f"Failed to write to docdb with error {e}. Output saved to results folder")

        print(f"Channels locations saved, and ccf coordinates saved for {probe_name_for_docdb}")


    def display_qc_options(self):

        # If not histology don't show
        if not self.histology_exists:
            return

        self.qc_dialog.open()

    def qc_button_clicked(self):

        # If no histology we can't plot histology
        if not self.histology_exists:
            return

        align_qc = self.align_qc.currentText()
        ephys_qc = self.ephys_qc.currentText()
        ephys_desc = []
        for button in self.desc_buttons.buttons():
            if button.isChecked():
                ephys_desc.append(button.text())

        if ephys_qc != 'Pass' and len(ephys_desc) == 0:
            QtWidgets.QMessageBox.warning(self, 'Status', "You must select a reason for qc choice")
            self.display_qc_options()
            return

        self.loaddata.upload_dj(align_qc, ephys_qc, ephys_desc)
        self.complete_button_pressed()

    def reset_axis_button_pressed(self):
        self.fig_hist.setYRange(min=self.probe_tip - self.probe_extra,
                                max=self.probe_top + self.probe_extra, padding=self.pad)
        self.fig_hist_ref.setYRange(min=self.probe_tip - self.probe_extra,
                                    max=self.probe_top + self.probe_extra, padding=self.pad)
        self.fig_img.setXRange(min=self.xrange[0], max=self.xrange[1], padding=0)
        self.fig_img.setYRange(min=self.probe_tip - self.probe_extra,
                               max=self.probe_top + self.probe_extra, padding=self.pad)

    def display_session_notes(self):
        self.notes_win = ephys_gui.PopupWindow(title='Session notes from Alyx', size=(200, 100),
                                               graphics=False)
        notes = QtWidgets.QTextEdit()
        notes.setReadOnly(True)
        notes.setLineWrapMode(QtWidgets.QTextEdit.WidgetWidth)
        notes.setText(self.sess_notes)
        self.notes_win.layout.addWidget(notes)

    def display_nearby_sessions(self):

        # If no histology we can't get nearby sessions
        if not self.histology_exists:
            return

        if not self.nearby:
            self.nearby, self.dist, self.dist_mlap = self.loaddata.get_nearby_trajectories()

        self.nearby_win = ephys_gui.PopupWindow(title='Nearby Sessions', size=(400, 300),
                                                graphics=False)

        self.nearby_table = QtWidgets.QTableWidget()
        self.nearby_table.setRowCount(10)
        self.nearby_table.setColumnCount(3)

        self.nearby_table.setHorizontalHeaderItem(0, QtWidgets.QTableWidgetItem('Session'))
        self.nearby_table.setHorizontalHeaderItem(1, QtWidgets.QTableWidgetItem('dist'))
        self.nearby_table.setHorizontalHeaderItem(2, QtWidgets.QTableWidgetItem('dist_mlap'))
        self.nearby_table.setSortingEnabled(True)
        for iT, (near, dist, dist_mlap) in enumerate(zip(self.nearby, self.dist, self.dist_mlap)):
            sess_item = QtWidgets.QTableWidgetItem(near)
            dist_item = QtWidgets.QTableWidgetItem()
            dist_item.setData(0, int(dist))
            dist_mlap_item = QtWidgets.QTableWidgetItem()
            dist_mlap_item.setData(0, int(dist_mlap))
            self.nearby_table.setItem(iT, 0, sess_item)
            self.nearby_table.setItem(iT, 1, dist_item)
            self.nearby_table.setItem(iT, 2, dist_mlap_item)

        self.nearby_win.layout.addWidget(self.nearby_table)

    def popup_closed(self, popup):
        popup_idx = [iP for iP, pop in enumerate(self.cluster_popups) if pop == popup][0]
        self.cluster_popups.pop(popup_idx)

    def popup_moved(self):
        self.activateWindow()

    def close_popups(self):
        for pop in self.cluster_popups:
            pop.blockSignals(True)
            pop.close()
        self.cluster_popups = []

    def minimise_popups(self):
        self.popup_status = not self.popup_status
        if self.popup_status:
            for pop in self.cluster_popups:
                pop.showNormal()
            self.activateWindow()
        else:
            for pop in self.cluster_popups:
                pop.showMinimized()
            self.activateWindow()

    def lin_fit_option_changed(self, state):
        if state == 0:
            self.lin_fit = False
            self.fit_button_pressed()
        else:
            self.lin_fit = True
            self.fit_button_pressed()

    def cluster_clicked(self, item, point):
        point_pos = point[0].pos()
        clust_idx = np.argwhere(self.data == point_pos.x())[0][0]

        autocorr, clust_no = self.plotdata.get_autocorr(clust_idx)
        autocorr_plot = pg.PlotItem()
        autocorr_plot.setXRange(min=np.min(self.plotdata.t_autocorr),
                                max=np.max(self.plotdata.t_autocorr))
        autocorr_plot.setYRange(min=0, max=1.05 * np.max(autocorr))
        self.set_axis(autocorr_plot, 'bottom', label='T (ms)')
        self.set_axis(autocorr_plot, 'left', label='Number of spikes')
        plot = pg.BarGraphItem(x=self.plotdata.t_autocorr, height=autocorr, width=0.24,
                               brush=self.bar_colour)
        autocorr_plot.addItem(plot)

        template_wf = self.plotdata.get_template_wf(clust_idx)
        template_plot = pg.PlotItem()
        plot = pg.PlotCurveItem()
        template_plot.setXRange(min=np.min(self.plotdata.t_template),
                                max=np.max(self.plotdata.t_template))
        self.set_axis(template_plot, 'bottom', label='T (ms)')
        self.set_axis(template_plot, 'left', label='Amplitude (a.u.)')
        plot.setData(x=self.plotdata.t_template, y=template_wf, pen=self.kpen_solid)
        template_plot.addItem(plot)

        clust_layout = pg.GraphicsLayout()
        clust_layout.addItem(autocorr_plot, 0, 0)
        clust_layout.addItem(template_plot, 1, 0)

        self.clust_win = ephys_gui.PopupWindow(title=f'Cluster {clust_no}')
        self.clust_win.closed.connect(self.popup_closed)
        self.clust_win.moved.connect(self.popup_moved)
        self.clust_win.popup_widget.addItem(autocorr_plot, 0, 0)
        self.clust_win.popup_widget.addItem(template_plot, 1, 0)
        self.cluster_popups.append(self.clust_win)
        self.activateWindow()

        return clust_no

    def display_subject_scaling(self):
        if self.subj_win is not None:
            self.subj_win.close()

        self.subj_win = ScalingWindow(self.loaddata.probe_id, self.loaddata.subj, self.loaddata.one, self.loaddata.brain_atlas)

    def display_region_features(self):
        self.region_win = RegionFeatureWindow(self.loaddata.one, np.unique(np.array(self.ephysalign.region_id).ravel()),
                                              self.loaddata.brain_atlas, download=False)
        self.region_win.show()

    def on_mouse_double_clicked(self, event):
        """
        Triggered when a double click event is detected on ephys of histology plots. Adds reference
        line on ephys and histology plot that can be moved to align ephys signatures with brain
        regions. Also adds scatter point on fit plot
        :param event: double click event signals
        :type event: pyqtgraph mouseEvents
        """
        # If no histology no point adding lines
        if not self.histology_exists:
            return

        if event.double():
            pos = self.data_plot.mapFromScene(event.scenePos())
            pen, brush = self.create_line_style()
            line_track = pg.InfiniteLine(pos=pos.y() * self.y_scale, angle=0, pen=pen,
                                         movable=True)
            line_track.sigPositionChanged.connect(self.update_lines_track)
            line_track.setZValue(100)
            line_feature1 = pg.InfiniteLine(pos=pos.y() * self.y_scale, angle=0, pen=pen,
                                            movable=True)
            line_feature1.setZValue(100)
            line_feature1.sigPositionChanged.connect(self.update_lines_features)
            line_feature2 = pg.InfiniteLine(pos=pos.y() * self.y_scale, angle=0, pen=pen,
                                            movable=True)
            line_feature2.setZValue(100)
            line_feature2.sigPositionChanged.connect(self.update_lines_features)
            line_feature3 = pg.InfiniteLine(pos=pos.y() * self.y_scale, angle=0, pen=pen,
                                            movable=True)
            line_feature3.setZValue(100)
            line_feature3.sigPositionChanged.connect(self.update_lines_features)
            self.fig_hist.addItem(line_track)
            self.fig_img.addItem(line_feature1)
            self.fig_line.addItem(line_feature2)
            self.fig_probe.addItem(line_feature3)

            self.lines_features = np.vstack([self.lines_features, [line_feature1, line_feature2,
                                                                   line_feature3]])
            self.lines_tracks = np.vstack([self.lines_tracks, line_track])

            point = pg.PlotDataItem()
            point.setData(x=[line_track.pos().y()], y=[line_feature1.pos().y()],
                          symbolBrush=brush, symbol='o', symbolSize=10)
            self.fig_fit.addItem(point)
            self.points = np.vstack([self.points, point])

    def on_mouse_hover(self, items):
        """
        Returns the pyqtgraph items that the mouse is hovering over. Used to identify reference
        lines so that they can be deleted
        """
        if len(items) > 1:
            self.selected_line = []
            if type(items[0]) == pg.InfiniteLine:
                self.selected_line = items[0]
            elif (items[0] == self.fig_scale) & (type(items[1]) == pg.LinearRegionItem):
                idx = np.where(self.scale_regions == items[1])[0][0]
                self.fig_scale_ax.setLabel('Scale Factor = ' +
                                           str(np.around(self.scale_factor[idx], 2)))
            elif (items[0] == self.fig_hist) & (type(items[1]) == pg.LinearRegionItem):
                self.selected_region = items[1]
            elif (items[0] == self.fig_hist_ref) & (type(items[1]) == pg.LinearRegionItem):
                self.selected_region = items[1]

    def update_lines_features(self, line):
        """
        Triggered when reference line on ephys data plots is moved. Moves all three lines on the
        img_plot, line_plot and probe_plot and adjusts the corresponding point on the fit plot
        :param line: selected line
        :type line: pyqtgraph InfiniteLine
        """
        idx = np.where(self.lines_features == line)
        line_idx = idx[0][0]
        fig_idx = np.setdiff1d(np.arange(0, 3), idx[1][0])

        self.lines_features[line_idx][fig_idx[0]].setPos(line.value())
        self.lines_features[line_idx][fig_idx[1]].setPos(line.value())

        self.points[line_idx][0].setData(x=[self.lines_features[line_idx][0].pos().y()],
                                         y=[self.lines_tracks[line_idx][0].pos().y()])

    def update_lines_track(self, line):
        """
        Triggered when reference line on histology plot is moved. Adjusts the corresponding point
        on the fit plot
        :param line: selected line
        :type line: pyqtgraph InfiniteLine
        """
        line_idx = np.where(self.lines_tracks == line)[0][0]

        self.points[line_idx][0].setData(x=[self.lines_features[line_idx][0].pos().y()],
                                         y=[self.lines_tracks[line_idx][0].pos().y()])

    def tip_line_moved(self):
        """
        Triggered when dotted line indicating probe tip on self.fig_hist moved. Gets the y pos of
        probe tip line and ensures the probe top line is set to probe tip line y pos + 3840
        """
        self.top_pos.setPos(self.tip_pos.value() + self.probe_top)

    def top_line_moved(self):
        """
        Triggered when dotted line indicating probe top on self.fig_hist moved. Gets the y pos of
        probe top line and ensures the probe tip line is set to probe top line y pos - 3840
        """
        self.tip_pos.setPos(self.top_pos.value() - self.probe_top)

    def remove_lines_points(self):
        """
        Removes all reference lines and scatter points from the ephys, histology and fit plots
        """
        for line_feature, line_track, point in zip(self.lines_features, self.lines_tracks,
                                                   self.points):
            self.fig_img.removeItem(line_feature[0])
            self.fig_line.removeItem(line_feature[1])
            self.fig_probe.removeItem(line_feature[2])
            self.fig_hist.removeItem(line_track[0])
            self.fig_fit.removeItem(point[0])

    def add_lines_points(self):
        """
        Adds all reference lines and scatter points from the ephys, histology and fit plots
        """
        for line_feature, line_track, point in zip(self.lines_features, self.lines_tracks,
                                                   self.points):
            self.fig_img.addItem(line_feature[0])
            self.fig_line.addItem(line_feature[1])
            self.fig_probe.addItem(line_feature[2])
            self.fig_hist.addItem(line_track[0])
            self.fig_fit.addItem(point[0])

    def update_lines_points(self):
        """
        Updates position of reference lines on histology plot after fit has been applied. Also
        updates location of scatter point
        """
        for line_feature, line_track, point in zip(self.lines_features, self.lines_tracks,
                                                   self.points):
            line_track[0].setPos(line_feature[0].getYPos())
            point[0].setData(x=[line_feature[0].pos().y()], y=[line_feature[0].pos().y()])

    def create_lines(self, positions):
        for pos in positions:

            pen, brush = self.create_line_style()
            line_track = pg.InfiniteLine(pos=pos, angle=0, pen=pen, movable=True)
            line_track.sigPositionChanged.connect(self.update_lines_track)
            line_track.setZValue(100)
            line_feature1 = pg.InfiniteLine(pos=pos, angle=0, pen=pen,
                                            movable=True)
            line_feature1.setZValue(100)
            line_feature1.sigPositionChanged.connect(self.update_lines_features)
            line_feature2 = pg.InfiniteLine(pos=pos, angle=0, pen=pen,
                                            movable=True)
            line_feature2.setZValue(100)
            line_feature2.sigPositionChanged.connect(self.update_lines_features)
            line_feature3 = pg.InfiniteLine(pos=pos, angle=0, pen=pen,
                                            movable=True)
            line_feature3.setZValue(100)
            line_feature3.sigPositionChanged.connect(self.update_lines_features)
            self.fig_hist.addItem(line_track)
            self.fig_img.addItem(line_feature1)
            self.fig_line.addItem(line_feature2)
            self.fig_probe.addItem(line_feature3)

            self.lines_features = np.vstack([self.lines_features, [line_feature1, line_feature2,
                                                                   line_feature3]])
            self.lines_tracks = np.vstack([self.lines_tracks, line_track])

            point = pg.PlotDataItem()
            point.setData(x=[line_track.pos().y()], y=[line_feature1.pos().y()],
                          symbolBrush=brush, symbol='o', symbolSize=10)
            self.fig_fit.addItem(point)
            self.points = np.vstack([self.points, point])

    def create_line_style(self):
        """
        Create random choice of colour and style for reference line
        :return pen: style to use for the line
        :type pen: pyqtgraph Pen
        :return brush: colour use for the line
        :type brush: pyqtgraph Brush
        """
        colours = ['#cc0000', '#6aa84f', '#ff8d00', '#00FFF7', 
                   "#03fc84", '#fc03e7', "#1c03fc", "#000000"]
        style = [QtCore.Qt.SolidLine, QtCore.Qt.DashLine, QtCore.Qt.DashDotLine]
        col = QtGui.QColor(colours[randrange(len(colours))])
        sty = style[randrange(len(style))]
        pen = pg.mkPen(color=col, style=sty, width=7)
        brush = pg.mkBrush(color=col)
        return pen, brush

    def update_string(self):
        """
        Updates text boxes to indicate to user which move they are looking at
        """
        self.idx_string.setText(f"Current Index = {self.current_idx}")
        self.tot_idx_string.setText(f"Total Index = {self.total_idx}")


def viewer(probe_id, one=None, histology=False, spike_collection=None, title=None):
    """
    """
    qt.create_app()
    av = MainWindow._get_or_create(probe_id=probe_id, one=one, histology=histology,
                                   spike_collection=spike_collection, title=title)
    av.show()
    return av

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Offline vs online mode')
    parser.add_argument('-o', '--offline', default=True, required=False, help='Offline mode')
    parser.add_argument('-r', '--remote', default=False, required=False, action='store_true', help='Remote mode')
    parser.add_argument('-i', '--insertion', default=None, required=False, help='Insertion mode')
    args = parser.parse_args()

    app = QtWidgets.QApplication([])
    mainapp = MainWindow(offline=args.offline, probe_id=args.insertion, remote=args.remote)
    # mainapp = MainWindow(offline=True)
    mainapp.show()
    app.exec_()


if __name__ == '__main__':
    main()
    