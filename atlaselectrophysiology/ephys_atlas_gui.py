import os
import platform
from typing import Any, Union, Optional, List, Dict, Tuple, Callable

if platform.system() == 'Darwin':
    if platform.release().split('.')[0] >= '20':
        os.environ["QT_MAC_WANTS_LAYER"] = "1"

from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
import pyqtgraph.exporters
import numpy as np

import atlaselectrophysiology.utils.qt_utils as utils

from atlaselectrophysiology.load_data import LoadData
from ibllib.pipes.ephys_alignment import EphysAlignment
import atlaselectrophysiology.plot_data as pd
import atlaselectrophysiology.utils.ColorBar as cb
import atlaselectrophysiology.ephys_gui_setup as ephys_gui
from atlaselectrophysiology.subject_scaling import ScalingWindow
from ephysfeatures.features_across_region import RegionFeatureWindow
from atlaselectrophysiology.create_overview_plots import make_overview_plot
from pathlib import Path
from qt_helpers import qt
import matplotlib.pyplot as mpl  # noqa  # This is needed to make qt show properly :/


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

    def __init__(self, offline=False, probe_id=None, one=None, histology=True,
                 spike_collection=None, unity=False, loaddata=None):
        super(MainWindow, self).__init__()

        if unity:
            from atlaselectrophysiology.unity_data import UnityData
            self.unitydata = UnityData()
            self.unity = True
        else:
            self.unity = False

        self.init_variables()
        self.init_layout(self, offline=offline)
        self.configure = True

        if loaddata is not None:
            self.loaddata = loaddata
            self.loaddata.get_info(probe_id)
            self.loaddata.get_starting_alignment(0)
            self.data_button_pressed()
            self.offline = False


        if not offline and probe_id is None:
            self.loaddata = LoadData()
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
        # elif loaddata is None:
        #     self.loaddata = LoadDataLocal()
        #     self.offline = True
        #     self.histology_exists = True


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
        self.probe_tip = 0
        self.probe_top = 3840
        self.probe_extra = 100
        self.view_total = [-2000, 6000]
        self.depth = np.arange(self.view_total[0], self.view_total[1], 20)
        self.extend_feature = 1

        # Initialise with linear fit scaling as default
        self.lin_fit = True

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


        self.hist_nearby_x = None
        self.hist_nearby_y = None
        self.hist_nearby_col = None
        self.hist_nearby_parent_x = None
        self.hist_nearby_parent_y = None
        self.hist_nearby_parent_col = None
        self.hist_mapping = 'Allen'

        self.hist_regions = dict()


        self.nearby = None

        # keep track of unity display
        self.unity_plot = None
        self.unity_region_status = True
        self.point_size = 0.05

        # Filter by different types of units
        self.filter_type = 'all'


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

        # Remove all existing layout items to start fresh
        for item in [self.fig_img_cb, self.fig_probe_cb, self.fig_img, self.fig_line, self.fig_probe]:
            self.fig_data_layout.removeItem(item)

        # Define configurations for each view mode
        layout_configs = {
            1: {
                'items': [
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
                'items': [
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
                'items': [
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

        # Add layout items
        for item_args in config['items']:
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
            The group of QAction items representing plots to toggle through
        """

        current_act = options_group.checkedAction()
        actions = options_group.actions()
        current_idx = next(i for i, act in enumerate(actions) if act == current_act)
        next_idx = np.mod(current_idx + 1, len(actions))
        actions[next_idx].setChecked(True)
        actions[next_idx].trigger()

    @staticmethod
    def remove_items(fig, items):
        for item in items:
            fig.removeItem(item)
        return []

    def set_xaxis_range(self, fig, data, label=True):
        fig.setXRange(min=data['xrange'][0], max=data['xrange'][1], padding=0)
        if label:
            utils.set_axis(fig, 'bottom', label=data['xaxis'])

    def set_yaxis_range(self, fig):
        fig.setYRange(min=self.probe_tip - self.probe_extra, max=self.probe_top + self.probe_extra, padding=self.pad)


    # -------------------------------------------------------------------------------------------------
    # Plotting functions
    # -------------------------------------------------------------------------------------------------
    def plot_histology(
            self,
            fig: pg.PlotWidget,
            data: Dict[str, Any],
            ax: str ='left'
    ) -> None:
        """
        Plot histology regions on the given figure, shows brain regions intersecting with the probe track.

        Parameters
        ----------
        fig : pyqtgraph.PlotWidget
            The figure widget on which to plot the histology regions.
        data : dict
            A dictionary containing histology data with the following keys:
            - 'axis_label' : list of tuple(float, str)
                Tick labels and positions for the axis.
            - 'region' : list of tuple(float, float)
                Start and end coordinates for each histology region in micrometres.
            - 'colour' : list of tuple(int, int, int)
                RGB color tuples for each region.
        ax : 'left' or 'right', optional
            Orientation of the axis on which to add labels. 'left' for the main histology figure (fig_hist),
            and 'right' for the reference figure (fig_hist_ref). Default is 'left'.
        """
        # If no histology we can't plot histology
        if not self.histology_exists:
            return

        fig.clear()
        self.hist_regions[ax] = np.empty((0, 1), dtype=object)

        # Configure axis ticks and appearance
        axis = fig.getAxis(ax)
        axis.setTicks([data['axis_label']])
        axis.setZValue(10)
        utils.set_axis(fig, 'bottom', pen='w', label='blank')

        # Plot each histology region
        for ir, reg in enumerate(data['region']):
            colour = QtGui.QColor(*data['colour'][ir])
            region = pg.LinearRegionItem(values=(reg[0], reg[1]),
                                         orientation=pg.LinearRegionItem.Horizontal,
                                         brush=colour,
                                         movable=False)
            bound = pg.InfiniteLine(pos=reg[0], angle=0, pen='w')
            fig.addItem(region)
            fig.addItem(bound)
            # Keep track of each histology LinearRegionItem for label pressed interaction
            self.hist_regions[ax] = np.vstack([self.hist_regions[ax], region])

        # Add additional boundary for final region
        bound = pg.InfiniteLine(pos=data['region'][-1][1], angle=0, pen='w')
        fig.addItem(bound)

        # Add dotted probe track limits (tip and top) as horizontal reference lines
        # TODO this is ugly, also better handling of tip_pos and top_pos for the fig_hist vs fig_hist_ref
        if ax == 'left':
            self.tip_pos = pg.InfiniteLine(pos=self.probe_tip, angle=0, pen=self.kpen_dot, movable=True)
            self.top_pos = pg.InfiniteLine(pos=self.probe_top, angle=0, pen=self.kpen_dot, movable=True)

            offset = 1
            self.tip_pos.setBounds((self.shank.align.track[0] * 1e6 + offset,
                                    self.shank.align.track[-1] * 1e6 - (self.probe_top + offset)))
            self.top_pos.setBounds((self.shank.align.track[0] * 1e6 + (self.probe_top + offset),
                                    self.shank.align.track[-1] * 1e6 - offset))
            self.tip_pos.sigPositionChanged.connect(self.tip_line_moved)
            self.top_pos.sigPositionChanged.connect(self.top_line_moved)
        else:
            self.tip_pos = pg.InfiniteLine(pos=self.probe_tip, angle=0, pen=self.kpen_dot)
            self.top_pos = pg.InfiniteLine(pos=self.probe_top, angle=0, pen=self.kpen_dot)

        fig.addItem(self.tip_pos)
        fig.addItem(self.top_pos)

        # Set default selected region
        if ax == 'left':
            self.selected_region = self.hist_regions[ax][-2]


    def plot_histology_nearby(self, fig, ax='right', movable=False):
        # TODO

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

        utils.set_axis(fig, 'bottom', label='dist to boundary (um)')
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


    def plot_scale_factor(self) -> None:
        """
        Plot the scale factor applied to brain regions along the probe track, displayed alongside
        the histology figure. This visualizes regional scale adjustments as colored horizontal bands.
        """

        # If no histology we can't do alignment
        if not self.histology_exists:
            return

        self.fig_scale.clear()
        self.scale_regions = np.empty((0, 1))
        self.scale_factor = self.scale_data['scale']
        scale_factor = self.scale_data['scale'] - 0.5

        color_bar = cb.ColorBar('seismic')
        cbar = color_bar.makeColourBar(20, 5, self.fig_scale_cb, min=0.5, max=1.5, label='Scale Factor')
        colours = color_bar.map.mapToQColor(scale_factor)
        self.fig_scale_cb.addItem(cbar)

        for ir, reg in enumerate(self.scale_data['region']):
            region = pg.LinearRegionItem(values=(reg[0], reg[1]),
                                         orientation=pg.LinearRegionItem.Horizontal,
                                         brush=colours[ir], movable=False)
            bound = pg.InfiniteLine(pos=reg[0], angle=0, pen=colours[ir])

            self.fig_scale.addItem(region)
            self.fig_scale.addItem(bound)
            self.scale_regions = np.vstack([self.scale_regions, region])

        bound = pg.InfiniteLine(pos=self.scale_data['region'][-1][1], angle=0, pen=colours[-1])
        self.fig_scale.addItem(bound)

        self.fig_scale.setYRange(min=self.probe_tip - self.probe_extra,
                                 max=self.probe_top + self.probe_extra, padding=self.pad)
        utils.set_axis(self.fig_scale, 'bottom', pen='w', label='blank')


    def plot_fit(self) -> None:
        """
        Plot the scale factor and offset applied to channels along the depth of the probe track,
        relative to the original positions of the channels.
        """

        # If no histology we can't do alignment
        if not self.histology_exists:
            return

        x = self.shank.align.feature * 1e6
        y = self.shank.align.track * 1e6

        self.fit_plot.setData(x=x, y=y)
        self.fit_scatter.setData(x=x,y=y)

        depth_lin = self.shank.align.ephysalign.feature2track_lin(self.depth, x, y)

        if np.any(depth_lin):
            self.fit_plot_lin.setData(x=self.depth, y=depth_lin)
        else:
            self.fit_plot_lin.setData()

    def plot_slice(
            self,
            data: Dict,
            img_type: str):
        """
        Plot a slice image representing histology data, with optional label or intensity image.

        Displays the histology slice on the figure widget with appropriate color mapping,
        overlays trajectory lines, and updates channel plotting.

        Parameters
        ----------
        data : dict
            Dictionary containing slice image data and metadata. Expected keys include:
            - 'label' (np.ndarray): image data for annotation slice
            - 'hist_rd', 'hist_gr', 'ccf' (np.ndarray): image data for histology slice
            - 'scale' (tuple[float, float]): scaling factors for x and y axes
            - 'offset' (tuple[float, float]): offsets for x and y axes
        img_type : 'label', 'hist_rd', 'hist_gr', 'hist_cb' or 'ccf'
            The key of the image data to plot. If 'label', displays a labeled slice without LUT.
            For other types, a color lookup table and histogram are applied.
        """

        # If no histology we can't do alignment
        if not self.histology_exists:
            return

        # Clear previous display
        self.fig_slice.clear()
        self.slice_chns = pg.ScatterPlotItem()
        self.slice_lines = []

        # Create image item with data to display
        img = pg.ImageItem()
        img.setImage(data[img_type])

        # Construct QTransform from scale and offset data
        transform = [data['scale'][0], 0., 0., 0.,
                     data['scale'][1], 0.,
                     data['offset'][0], data['offset'][1], 1.]
        img.setTransform(QtGui.QTransform(*transform))


        if img_type == 'label':
            self.fig_slice_layout.removeItem(self.slice_item)
            self.fig_slice_layout.addItem(self.fig_slice_hist_alt, 0, 1)
            self.slice_item = self.fig_slice_hist_alt
        else:
            color_bar = cb.ColorBar('cividis')
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

        # Plot the trajectory line
        self.traj_line = pg.PlotCurveItem()
        self.traj_line.setData(x=self.shank.xyz_track[:, 0], y=self.shank.xyz_track[:, 2], pen=self.kpen_solid)
        self.fig_slice.addItem(self.traj_line)

        # Plot channels
        self.fig_slice.addItem(self.slice_chns)
        self.plot_channels()

    def plot_channels(self) -> None:
        """
        Plot electrode channels and alignment lines on the histological slice.

        Displays the locations of electrode channels and probe track reference lines
        (alignment lines) on the histology slice view.
        """

        # If no histology we can't do alignment
        if not self.histology_exists:
            return

        self.channel_status = True
        self.xyz_channels = self.shank.xyz_channels
        track_lines = self.shank.align.track_lines

        # Clear existing reference lines
        self.slice_lines = self.remove_items(self.fig_slice, self.slice_lines)

        self.slice_chns.setData(x=self.xyz_channels[:, 0], y=self.xyz_channels[:, 2], pen='r', brush='r')

        for ref_line in track_lines:
            line = pg.PlotCurveItem()
            line.setData(x=ref_line[:, 0], y=ref_line[:, 2], pen=self.kpen_dot)
            self.fig_slice.addItem(line)
            self.slice_lines.append(line)


    def plot_scatter(
            self,
            data: Optional[Dict]
        ) -> None:
        """
        Plot a 2D scatter plot of electrophysiological data on the figure.

        Parameters
        ----------
        data : dict, optional
            Dictionary containing the scatter plot data and display parameters. Expected keys:
                - 'x' : np.ndarray of float
                    X-coordinates of the data points.
                - 'y' : np.ndarray of float
                    Y-coordinates of the data points.
                - 'size' : np.ndarray of float
                    Size of each point.
                - 'symbol' : np.ndarray of str
                    Symbol type (e.g., 'o', 't', etc.) for each point.
                - 'colours' : np.ndarray
                    Either array of floats for colormap or array of QColor objects.
                - 'pen' : str or QPen
                    Pen used to outline each point.
                - 'levels' : tuple[float, float]
                    Min and max levels used for the colormap.
                - 'title' : str
                    Title for the colorbar.
                - 'xrange' : tuple[float, float]
                    Display range for the x-axis.
                - 'xaxis' : str
                    Label for the x-axis.
                - 'cmap' : str
                    Name of the colormap to use.
                - 'cluster' : bool
                    If True, enables interactive cluster selection on click.
        """

        if not data:
            print('data for this plot not available')
            return

        # Clear existing plot
        self.img_plots = self.remove_items(self.fig_img, self.img_plots)
        self.img_cbars = self.remove_items(self.fig_img_cb, self.img_cbars)
        # TODO check if I need this
        # utils.set_axis(self.fig_img_cb, 'top', pen='w')

        size = data['size'].tolist()
        symbol = data['symbol'].tolist()

        # Create colorbar and add to figure
        color_bar = cb.ColorBar(data['cmap'])
        cbar = color_bar.makeColourBar(20, 5, self.fig_img_cb, min=np.min(data['levels'][0]),
                                       max=np.max(data['levels'][1]), label=data['title'])
        self.fig_img_cb.addItem(cbar)
        self.img_cbars.append(cbar)

        # Determine brush type based on given data
        if isinstance(np.any(data['colours']), QtGui.QColor):
            brush = data['colours'].tolist()
        else:
            brush = color_bar.getBrush(data['colours'], levels=list(data['levels']))

        # Create scatter plot and add to figure
        plot = pg.ScatterPlotItem()
        plot.setData(x=data['x'], y=data['y'], symbol=symbol, size=size, brush=brush, pen=data['pen'])
        self.fig_img.addItem(plot)
        self.img_plots.append(plot)

        # Set axis ranges
        self.set_xaxis_range(self.fig_img, data)
        self.set_yaxis_range(self.fig_img)

        # Store plot references
        self.y_scale = 1
        self.ephys_plot = plot
        self.xrange = data['xrange']

        if data['cluster']:
            self.cluster_data = data['x']
            self.ephys_plot.sigClicked.connect(self.cluster_clicked)

    def plot_line(
            self,
            data: Optional[Dict]
        ) -> None:
        """
        Plot a 1D line plot of electrophysiological data on the figure.

        Parameters
        ----------
        data : dict, optional
            Dictionary containing the line plot data and display parameters. Expected keys:
                - 'x' : np.ndarray of float
                    X-coordinates of the data points.
                - 'y' : np.ndarray of float
                    Y-coordinates of the data points.
                - 'xrange' : tuple[float, float]
                    Display range for the x-axis.
                - 'xaxis' : str
                    Label for the x-axis.
        """

        if not data:
            print('data for this plot not available')
            return
        # Clear existing plots
        self.line_plots = self.remove_items(self.fig_line, self.line_plots)

        # Create line plot at add to figure
        line = pg.PlotCurveItem()
        line.setData(x=data['x'], y=data['y'])
        line.setPen(self.kpen_solid)
        self.fig_line.addItem(line)
        self.line_plots.append(line)

        # Set axis ranges
        self.set_xaxis_range(self.fig_line, data)
        self.set_yaxis_range(self.fig_line)

    def plot_probe(
            self,
            data: Optional[Dict],
        ) -> None:
        """
        Plot a 2D image representing the probe geometry, including individual channel banks and optional boundaries.

        Parameters
        ----------
        data : dict
            Dictionary containing information to plot the probe geometry.
            Expected keys:
                img : list of np.ndarray
                    Image data arrays, one for each channel bank.
                scale : list of np.ndarray
                    Scaling to apply to each image, as [xscale, yscale].
                offset : list of np.ndarray
                    Offset to apply to each image, as [xoffset, yoffset].
                levels : np.ndarray
                    Two-element array specifying color bar limits [min, max].
                cmap : str
                    Colormap to use.
                xrange : np.ndarray
                    Two-element array for x-axis limits [min, max].
                title : str
                    Label for the color bar.
        """
        if not data:
            print('data for this plot not available')
            return

        self.probe_plots = self.remove_items(self.fig_probe, self.probe_plots)
        self.probe_cbars = self.remove_items(self.fig_probe_cb, self.probe_cbars)
        self.probe_bounds = self.remove_items(self.fig_probe, self.probe_bounds)

        # Create colorbar and add to figure
        color_bar = cb.ColorBar(data['cmap'])
        lut = color_bar.getColourMap()
        cbar = color_bar.makeColourBar(20, 5, self.fig_probe_cb, min=data['levels'][0],
                                       max=data['levels'][1], label=data['title'], lim=False)
        self.fig_probe_cb.addItem(cbar)
        self.probe_cbars.append(cbar)

        # Create image plots per shank and add to figure
        for img, scale, offset in zip(data['img'], data['scale'], data['offset']):
            image = pg.ImageItem()
            image.setImage(img)
            transform = [scale[0], 0., 0., 0., scale[1], 0., offset[0], offset[1], 1.]
            image.setTransform(QtGui.QTransform(*transform))
            image.setLookupTable(lut)
            image.setLevels((data['levels'][0], data['levels'][1]))
            self.fig_probe.addItem(image)
            self.probe_plots.append(image)

        # Set axis ranges
        self.set_xaxis_range(self.fig_probe, data, label=False)
        self.set_yaxis_range(self.fig_probe)
        # Add in a fake label so that the appearence is the same as other plots
        utils.set_axis(self.fig_probe, 'bottom', pen='w', label='blank')

        # Optionally plot horizontal boundary lines
        bounds = data.get('boundaries', None)
        if bounds is not None:
            for bound in bounds:
                line = pg.InfiniteLine(pos=bound, angle=0, pen='w')
                self.fig_probe.addItem(line)
                self.probe_bounds.append(line)

    def plot_image(
            self,
            data: Optional[Dict],
        ) -> None:
        """
        Plot a 2D image of electrophysiology data

        Parameters
        ----------
        data : dict
            Dictionary containing image and display metadata. Expected keys:
                img : np.ndarray
                    2D image array of shape (nx, ny), representing the electrophysiological data.
                scale : np.ndarray
                    Two-element array [xscale, yscale] for scaling the image along each axis.
                offset : np.ndarray
                    Two-element array [xoffset, yoffset] specifying the image translation.
                levels : np.ndarray
                    Two-element array [min, max] specifying the color scaling limits.
                cmap : str
                    Name of the colormap to use.
                xrange : np.ndarray
                    Two-element array [xmin, xmax] for x-axis limits.
                xaxis : str
                    Label for the x-axis.
                title : str
                    Title or label for the colorbar.
        """
        if not data:
            print('data for this plot not available')
            return

        self.img_plots = self.remove_items(self.fig_img, self.img_plots)
        self.img_cbars = self.remove_items(self.fig_img_cb, self.img_cbars)
        # TODO check if I need this
        # utils.set_axis(self.fig_img_cb, 'top', pen='w')

        # Create image item and add to figure
        image = pg.ImageItem()
        image.setImage(data['img'])
        transform = [data['scale'][0], 0., 0., 0., data['scale'][1], 0., data['offset'][0], data['offset'][1], 1.]
        image.setTransform(QtGui.QTransform(*transform))
        self.fig_img.addItem(image)
        self.img_plots.append(image)

        # If applicable create colorbar and add to figure
        cmap = data.get('cmap', [])
        if cmap:
            color_bar = cb.ColorBar(data['cmap'])
            lut = color_bar.getColourMap()
            image.setLookupTable(lut)
            image.setLevels((data['levels'][0], data['levels'][1]))
            cbar = color_bar.makeColourBar(20, 5, self.fig_img_cb, min=data['levels'][0],
                                           max=data['levels'][1], label=data['title'])
            self.fig_img_cb.addItem(cbar)
            self.img_cbars.append(cbar)
        else:
            image.setLevels((1, 0))

        # Set axis ranges
        self.set_xaxis_range(self.fig_img, data)
        self.set_yaxis_range(self.fig_img)

        # Store plot references
        self.y_scale = data['scale'][1]
        self.x_scale = data['scale'][0]
        self.ephys_plot = image
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

    # def on_shank_selected(self, idx):
    #     """
    #     Triggered in offline mode for selecting shank when using NP2.0
    #     """
    #     self.current_shank_idx = idx
    #     # Update prev_alignments
    #     self.prev_alignments = self.loaddata.get_previous_alignments(self.current_shank_idx)
    #     utils.populate_lists(self.prev_alignments, self.align_list, self.align_combobox)
    #     self.loaddata.get_starting_alignment(0)


    def on_shank_selected(self, idx):
        """
        Online version
        """
        # Todo make this match the 'all' string
        if idx == 4:
            # Launch 3 more windows
            self.loaddata.get_info(0)
            # Update prev_alignments
            self.prev_alignments = self.loaddata.get_previous_alignments()
            utils.populate_lists(self.prev_alignments, self.align_list, self.align_combobox)
            self.loaddata.get_starting_alignment(0)
            self.data_button_pressed()
            for idx in [1, 2, 3]:
                multi_shank_viewer(title=f'window_{idx}', probe_id=idx, loaddata=self.loaddata)


        self.align_list.clear()
        self.loaddata.get_info(idx)
        # Update prev_alignments
        self.prev_alignments = self.loaddata.get_previous_alignments()
        utils.populate_lists(self.prev_alignments, self.align_list, self.align_combobox)
        self.loaddata.get_starting_alignment(0)

        if self.shank is not None:
            self.data_button_pressed()

    def on_folder_selected(self):
        """
        Triggered in offline mode when folder button is clicked
        """
        self.data_status = False
        folder_path = Path(QtWidgets.QFileDialog.getExistingDirectory(None, "Select Folder"))
        self.folder_line.setText(str(folder_path))
        self.prev_alignments, shank_options = self.loaddata.get_info(folder_path)
        utils.populate_lists(shank_options, self.shank_list, self.shank_combobox)
        utils.populate_lists(self.prev_alignments, self.align_list, self.align_combobox)
        self.on_shank_selected(0)

    def on_alignment_selected(self, idx):
        self.loaddata.get_starting_alignment(idx)
        if self.shank is not None:
            self.data_button_pressed()

    def add_alignment_pressed(self):
        file_path = Path(QtWidgets.QFileDialog.getOpenFileName()[0])
        if file_path.name != 'prev_alignments.json':
            print("Wrong file selected, must be of format prev_alignments.json")
            return
        else:
            self.prev_alignments = self.loaddata.add_extra_alignments(file_path)
            utils.populate_lists(self.prev_alignments, self.align_list, self.align_combobox)
            self.feature_prev, self.track_prev = self.loaddata.get_starting_alignment(0)

    def data_button_pressed(self):
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
        self.remove_reference_lines_from_display()
        self.init_variables()

        self.loaddata.load_data()

        self.shank = self.loaddata.get_selected_probe()
        # TODO figure out where to put this
        self.shank.align.features[self.shank.align.idx], self.shank.align.tracks[self.shank.align.idx], _ \
            = self.shank.align.ephysalign.get_track_and_feature()

        self.xyz_channels = self.shank.xyz_channels
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
        self.plot_image(self.shank.img_plots['Firing Rate'])
        self.plot_probe(self.shank.probe_plots['rms AP'])
        self.plot_line(self.shank.line_plots['Firing Rate'])

        # Initialise histology plots
        self.plot_histology(self.fig_hist_ref, self.hist_data_ref, ax='right')
        self.plot_histology(self.fig_hist, self.hist_data)
        self.label_status = False
        self.toggle_labels()
        self.plot_scale_factor()
        if np.any(self.shank.feature_prev):
            self.create_reference_lines(self.shank.feature_prev[1:-1] * 1e6)
        # Initialise slice and fit images
        self.plot_fit()
        self.plot_slice(self.shank.slice_data, 'hist_rd')

        # Initialise unity plot
        # if self.unity:
        #     self.unitydata.add_regions(np.unique(self.hist_data['axis_label'][:, 1]))
        #     self.set_unity_xyz()
        #     self.plot_unity('probe')

        # Only configure the view the first time the GUI is launched
        self.set_view(view=1, configure=self.configure)
        self.configure = False



    def filter_unit_pressed(self, filter_type):
        #TODO FIX
        self.filter_type = filter_type
        self.plotdata.filter_units(self.filter_type)
        self.scat_drift_data = self.plotdata.get_depth_data_scatter()
        (self.scat_fr_data, self.scat_p2t_data,
         self.scat_amp_data) = self.plotdata.get_fr_p2t_data_scatter()
        self.img_corr_data = self.plotdata.get_correlation_data_img()
        self.img_fr_data = self.plotdata.get_fr_img()
        self.line_fr_data, self.line_amp_data = self.plotdata.get_fr_amp_data_line()
        self.probe_rfmap, self.rfmap_boundaries = self.plotdata.get_rfmap_data()
        self.img_stim_data = self.plotdata.get_passive_events()
        self.img_init.setChecked(True)
        self.line_init.setChecked(True)
        self.probe_init.setChecked(True)
        self.plot_image(self.shank.img_plots['Firing Rate'])
        self.plot_probe(self.shank.probe_plots['rms AP'])
        self.plot_line(self.shank.line_plots['Firing Rate'])

        if self.unity:
            self.unity_plot = 'probe'
            self.set_unity_xyz()
            self.plot_unity()

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

        upload = QtWidgets.QMessageBox.question(self, '', "Upload alignment?",
                                                QtWidgets.QMessageBox.Yes |
                                                QtWidgets.QMessageBox.No)

        if upload == QtWidgets.QMessageBox.Yes:
            upload_channels = self.loaddata.upload_data(self.xyz_channels)
            self.loaddata.update_alignments(self.shank.align.features[self.idx], self.shank.align.tracks[self.idx])
            self.prev_alignments = self.loaddata.get_previous_alignments()
            utils.populate_lists(self.prev_alignments, self.align_list, self.align_combobox)
            self.loaddata.get_starting_alignment(0)
            resolved = self.loaddata.update_qc()

            if upload_channels and resolved == 0:
                # channels saved alignment not resolved
                QtWidgets.QMessageBox.information(self, 'Status',
                                                  ("Channels locations saved to Alyx. "
                                                   "Alignment not resolved"))
            if upload_channels and resolved == 1:
                # channels saved alignment resolved, writen to flatiron
                QtWidgets.QMessageBox.information(self, 'Status',
                                                  ("Channel locations saved to Alyx. "
                                                   "Alignment resolved and channels "
                                                   "datasets written to flatiron"))
            if not upload_channels and resolved == 1:
                # alignment already resolved, save alignment but channels not written
                QtWidgets.QMessageBox.information(self, 'Status',
                                                  ("Channel locations not saved to Alyx"
                                                   " as alignment has already been "
                                                   "resolved. New user reference lines"
                                                   " have been saved"))
        else:
            pass
            QtWidgets.QMessageBox.information(self, 'Status', "Channels not saved")

    def complete_button_pressed_offline(self):
        """
        Triggered when complete button or Shift+F key pressed. Uploads final channel locations to
        json file
        """
        upload = QtWidgets.QMessageBox.question(self, '', "Upload alignment?",
                                                QtWidgets.QMessageBox.Yes |
                                                QtWidgets.QMessageBox.No)

        if upload == QtWidgets.QMessageBox.Yes:
            self.loaddata.upload_data(self.shank.align.features[self.idx], self.shank.align.tracks[self.idx],
                                      self.xyz_channels)
            self.prev_alignments = self.loaddata.get_previous_alignments(self.current_shank_idx)
            utils.populate_lists(self.prev_alignments, self.align_list, self.align_combobox)
            self.loaddata.get_starting_alignment(0)
            QtWidgets.QMessageBox.information(self, 'Status', "Channels locations saved")
        else:
            pass
            QtWidgets.QMessageBox.warning(self, 'Status', "Channels not saved")

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
        self.shank.align.offset_hist_data(val)

    def scale_hist_data(self) -> None:
        """
        Scale brain regions along the probe track based on reference lines.
        """
        # If no histology we can't do alignment
        if not self.histology_exists:
            return

        line_track = np.array([line[0].pos().y() for line in self.lines_tracks]) / 1e6
        line_feature = np.array([line[0].pos().y() for line in self.lines_features]) / 1e6

        self.shank.align.scale_hist_data(line_track, line_feature,
                                         extend_feature=self.extend_feature, lin_fit=self.lin_fit)

    def get_scaled_histology(self) -> None:
        """
        Retrieve scaled histological data after alignment operations.
        """

        self.hist_data, self.hist_data_ref, self.scale_data = self.shank.align.get_scaled_histology()


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
        self.update_plots()

    def update_plots(self) -> None:
        """
        Refresh all plots to reflect the current alignment state.
        """

        self.get_scaled_histology()
        self.plot_histology(self.fig_hist, self.hist_data)
        self.plot_scale_factor()
        self.plot_fit()
        self.plot_channels()
        if self.unity:
            self.set_unity_xyz()
            self.plot_unity()
        self.remove_reference_lines_from_display()
        self.add_reference_lines_to_display()
        self.align_reference_lines()
        self.set_yaxis_range(self.fig_hist)
        self.update_string()


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

        if self.shank.align.buffer.next_idx():
            self.update_plots()


    def prev_button_pressed(self) -> None:
        """
        Called when left key pressed.
        Updates all plots and indices with previous alignment.
        Ensures user cannot go back more than self.max_idx moves
        """

        # If no histology we can't plot histology
        if not self.histology_exists:
            return

        if self.shank.align.buffer.previous_idx():
            self.update_plots()


    def reset_button_pressed(self) -> None:
        """
        Called when Reset button or Shift+R is pressed.
        Resets feature and track alignment to original alignment and updates plots.
        """
        # If no histology we can't plot histology
        if not self.histology_exists:
            return

        self.remove_reference_lines_from_display()
        self.lines_features = np.empty((0, 3))
        self.lines_tracks = np.empty((0, 1))
        self.points = np.empty((0, 1))

        self.shank.align.reset_features_and_tracks()

        self.get_scaled_histology()

        self.plot_histology(self.fig_hist, self.hist_data)
        self.plot_scale_factor()
        if np.any(self.shank.feature_prev):
            self.create_reference_lines(self.shank.feature_prev[1:-1] * 1e6)
        self.plot_fit()
        self.plot_channels()
        self.set_yaxis_range(self.fig_hist)
        self.update_string()


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
        self.idx_string.setText(f"Current Index = {self.shank.align.buffer.current_idx}")
        self.tot_idx_string.setText(f"Total Index = {self.shank.align.buffer.total_idx}")

    # -------------------------------------------------------------------------------------------------
    # Mouse interactions
    # -------------------------------------------------------------------------------------------------


    def on_mouse_double_clicked(self, event) -> None:
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
            pos = self.ephys_plot.mapFromScene(event.scenePos())
            self.create_reference_line(pos.y() * self.y_scale)

    def on_mouse_hover(self, items: List[pg.GraphicsObject]) -> None:
        """
        Handles mouse hover events over pyqtgraph plot items.

        Identifies reference lines or linear regions the mouse is hovering over
        to allow interactive operations like deletion or displaying additional info.

        Parameters
        ----------
        items : list of pyqtgraph.GraphicsObject
            List of items under the mouse cursor.
        """
        if len(items) > 1:
            self.selected_line = []
            item0, item1 = items[0], items[1]

            if isinstance(item0, pg.InfiniteLine):
                self.selected_line = item0
            elif (item0 == self.fig_scale) and isinstance(item1, pg.LinearRegionItem):
                idx = np.where(self.scale_regions == item1)[0][0]
                self.fig_scale_ax.setLabel('Scale Factor = ' + str(np.around(self.scale_factor[idx], 2)))
            elif (item0 == self.fig_hist) and isinstance(item1, pg.LinearRegionItem):
                self.selected_region = item1
            elif (item0 == self.fig_hist_ref) and isinstance(item1, pg.LinearRegionItem):
                self.selected_region = items




    # -------------------------------------------------------------------------------------------------
    # Display options
    # -------------------------------------------------------------------------------------------------
    def toggle_labels(self) -> None:
        """
        Toggle visibility of brain region labels on histology plots.

        Triggered by pressing Shift+A. Updates the pens for axis items on both the main
        and reference histology plots to show or hide Allen atlas region labels.
        """
        self.label_status = not self.label_status
        pen = 'k' if self.label_status else None
        self.ax_hist_ref.setPen(pen)
        self.ax_hist_ref.setTextPen(pen)
        self.ax_hist.setPen(pen)
        self.ax_hist.setTextPen(pen)

        self.fig_hist_ref.update()
        self.fig_hist.update()

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

    def toggle_channels(self) -> None:
        """
        Toggle visibility of channels and trajectory lines on the histology slice image.

        Triggered by pressing Shift+C. Adds or removes the visual indicators of the probe
        trajectory and channels on the slice view.
        """
        # If no histology we can't plot histology
        if not self.histology_exists:
            return

        self.channel_status = not self.channel_status

        # Choose the appropriate method (addItem or removeItem) based on toggle state
        toggleItem = self.fig_slice.addItem if self.channel_status else self.fig_slice.removeItem

        toggleItem(self.traj_line)
        toggleItem(self.slice_chns)
        for line in self.slice_lines:
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
        self.set_yaxis_range(self.fig_hist)
        self.set_yaxis_range(self.fig_hist_ref)
        self.set_yaxis_range(self.fig_img)
        self.set_xaxis_range(self.fig_img, {'xrange': [self.xrange[0], self.xrange[1]]}, label=False)

    # -------------------------------------------------------------------------------------------------
    # Probe top and tip lines
    # -------------------------------------------------------------------------------------------------

    def set_probe_lims(
            self,
            min_val: Union[int, float],
            max_val: Union[int, float]
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
        self.probe_tip = min_val
        self.probe_top = max_val

        for top_line in self.probe_top_lines:
            top_line.setY(self.probe_top)

        for tip_line in self.probe_tip_lines:
            tip_line.setY(self.probe_tip)

    def tip_line_moved(self) -> None:
        """
        Callback triggered when the probe tip line (dotted line) on the histology figure is moved.

        This function updates the probe top line's vertical position to maintain a fixed
        vertical distance between tip and top (i.e., the probe length).
        """

        self.top_pos.setPos(self.tip_pos.value() + self.probe_top)

    def top_line_moved(self) -> None:
        """
        Callback triggered when the probe top line (dotted line) on the histology figure is moved.

        This function updates the probe tip line's vertical position to maintain a fixed
        vertical distance between tip and top (i.e., the probe length).
        """

        self.tip_pos.setPos(self.top_pos.value() - self.probe_top)


    # -------------------------------------------------------------------------------------------------
    # Reference lines
    # -------------------------------------------------------------------------------------------------

    def create_reference_line(
            self,
            pos: float,
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

        pen, brush = utils.create_line_style()

        # Reference line on histology figure (track)
        line_track = pg.InfiniteLine(pos=pos, angle=0, pen=pen, movable=True)
        line_track.sigPositionChanged.connect(self.update_track_reference_line)
        line_track.setZValue(100)
        self.fig_hist.addItem(line_track)

        # Reference lines on electrophysiology figures (feature)
        line_features = []
        for fig in [self.fig_img, self.fig_line, self.fig_probe]:
            line_feature = pg.InfiniteLine(pos=pos, angle=0, pen=pen, movable=True)
            line_feature.setZValue(100)
            line_feature.sigPositionChanged.connect(self.update_feature_reference_line)
            fig.addItem(line_feature)
            line_features.append(line_feature)

        self.lines_features = np.vstack([self.lines_features, line_features])
        self.lines_tracks = np.vstack([self.lines_tracks, line_track])

        # Add marker to fit figure
        point = pg.PlotDataItem()
        point.setData(x=[line_track.pos().y()], y=[line_features[0].pos().y()],
                      symbolBrush=brush, symbol='o', symbolSize=10)
        self.fig_fit.addItem(point)
        self.points = np.vstack([self.points, point])

    def create_reference_lines(
            self,
            positions: Union[np.ndarray, List[float]]
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
            self.create_reference_line(pos)

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

        # Attempt to find selected line in feature lines
        line_idx = np.where(self.lines_features == self.selected_line)[0]
        if line_idx.size == 0:
            # If not found, try in track lines
            line_idx = np.where(self.lines_tracks == self.selected_line)[0]
            if line_idx.size == 0:
                return  # Couldn't find either

        line_idx = line_idx[0]

        # Remove line items from plots
        self.fig_img.removeItem(self.lines_features[line_idx][0])
        self.fig_line.removeItem(self.lines_features[line_idx][1])
        self.fig_probe.removeItem(self.lines_features[line_idx][2])
        self.fig_hist.removeItem(self.lines_tracks[line_idx, 0])
        self.fig_fit.removeItem(self.points[line_idx, 0])

        # Remove from tracking arrays
        self.lines_features = np.delete(self.lines_features, line_idx, axis=0)
        self.lines_tracks = np.delete(self.lines_tracks, line_idx, axis=0)
        self.points = np.delete(self.points, line_idx, axis=0)

    def update_feature_reference_line(
            self,
            feature_line: pg.InfiniteLine
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
        idx = np.where(self.lines_features == feature_line)

        line_idx = idx[0][0]
        fig_idx = np.setdiff1d(np.arange(0, 3), idx[1][0]) #  Indices of two other plots

        # Update the other two lines to the new y-position
        self.lines_features[line_idx][fig_idx[0]].setPos(feature_line.value())
        self.lines_features[line_idx][fig_idx[1]].setPos(feature_line.value())

        # Update scatter point on the fit figure
        self.points[line_idx][0].setData(x=[self.lines_features[line_idx][0].pos().y()],
                                         y=[self.lines_tracks[line_idx][0].pos().y()])

    def update_track_reference_line(
            self,
            track_line: pg.InfiniteLine
        ) -> None:
        """
        Callback triggered when a reference line in the histology plot is moved.
        This updates the corresponding scatter point in the fit plot.

        Parameters
        ----------
        track_line : pyqtgraph.InfiniteLine
            The line instance that was moved by the user.
        """
        line_idx = np.where(self.lines_tracks == track_line)[0][0]

        self.points[line_idx][0].setData(x=[self.lines_features[line_idx][0].pos().y()],
                                         y=[self.lines_tracks[line_idx][0].pos().y()])

    def align_reference_lines(self) -> None:
        """
        Align the positions of all track reference lines and scatter points based on the new positions
        of their corresponding feature reference lines.
        """
        for line_feature, line_track, point in zip(self.lines_features, self.lines_tracks, self.points):
            line_track[0].setPos(line_feature[0].getYPos())
            point[0].setData(x=[line_feature[0].pos().y()], y=[line_feature[0].pos().y()])


    def remove_reference_lines_from_display(self) -> None:
        """
        Remove all reference lines and scatter points from the electrophysiology, histology, and fit plots.
        """
        for line_feature, line_track, point in zip(self.lines_features, self.lines_tracks, self.points):
            self.fig_img.removeItem(line_feature[0])
            self.fig_line.removeItem(line_feature[1])
            self.fig_probe.removeItem(line_feature[2])
            self.fig_hist.removeItem(line_track[0])
            self.fig_fit.removeItem(point[0])

    def add_reference_lines_to_display(self) -> None:
        """
        Add previously created reference lines and scatter points to their respective plots.
        """
        for line_feature, line_track, point in zip(self.lines_features, self.lines_tracks, self.points):
            self.fig_img.addItem(line_feature[0])
            self.fig_line.addItem(line_feature[1])
            self.fig_probe.addItem(line_feature[2])
            self.fig_hist.addItem(line_track[0])
            self.fig_fit.addItem(point[0])

    # -------------------------------------------------------------------------------------------------
    # Plugins
    # -------------------------------------------------------------------------------------------------
    def cluster_clicked(self, item, point):
        point_pos = point[0].pos()
        clust_idx = np.argwhere(self.cluster_data == point_pos.x())[0][0]

        autocorr, clust_no = self.plotdata.get_autocorr(clust_idx)
        autocorr_plot = pg.PlotItem()
        autocorr_plot.setXRange(min=np.min(self.plotdata.t_autocorr),
                                max=np.max(self.plotdata.t_autocorr))
        autocorr_plot.setYRange(min=0, max=1.05 * np.max(autocorr))
        utils.set_axis(autocorr_plot, 'bottom', label='T (ms)')
        utils.set_axis(autocorr_plot, 'left', label='Number of spikes')
        plot = pg.BarGraphItem(x=self.plotdata.t_autocorr, height=autocorr, width=0.24,
                               brush=self.bar_colour)
        autocorr_plot.addItem(plot)

        template_wf = self.plotdata.get_template_wf(clust_idx)
        template_plot = pg.PlotItem()
        plot = pg.PlotCurveItem()
        template_plot.setXRange(min=np.min(self.plotdata.t_template),
                                max=np.max(self.plotdata.t_template))
        utils.set_axis(template_plot, 'bottom', label='T (ms)')
        utils.set_axis(template_plot, 'left', label='Amplitude (a.u.)')
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



    def prepare_shank_data(self, shank):
        shank_picks, shank_feature, shank_track = self.loaddata.get_alignment_for_insertion(shank)
        if shank_picks is None:
            return None
        shank_data, shank_path = self.loaddata.get_probe_data(shank['name'])
        shank_align = EphysAlignment(shank_picks, shank_data['channels']['localCoordinates'][:, 1],
                                     track_prev=shank_track,
                                     feature_prev=shank_feature,
                                     brain_atlas=self.loaddata.brain_atlas)

        shank_plot = pd.PlotData(shank_path, shank_data, 0)
        chn_data = shank_plot.get_channel_data()
        clust_data = shank_plot.get_cluster_data()

        shank_channels = shank_align.get_channel_locations(shank_align.feature_init, shank_align.track_init)

        chn_data['x'] = shank_channels[:, 0]
        chn_data['y'] = shank_channels[:, 1]
        chn_data['z'] = shank_channels[:, 2]

        clust_data['x'] = chn_data['x'][shank_data['clusters']['channels']]
        clust_data['y'] = chn_data['y'][shank_data['clusters']['channels']]
        clust_data['z'] = chn_data['z'][shank_data['clusters']['channels']]

        clust_idx = dict()
        clust_idx['all'] = shank_plot.clust

        for filter_type in ['KS good', 'KS mua', 'IBL good']:
            shank_plot.filter_units(filter_type)
            clust_idx[filter_type] = shank_plot.clust

        return chn_data, clust_data, clust_idx

    def set_unity_xyz(self):
        for i, a in enumerate(['x', 'y', 'z']):
            self.unity_data[self.loaddata.probe_label]['channels'][a] = self.xyz_channels[:, i]
            self.unity_data[self.loaddata.probe_label]['clusters'][a] = (self.xyz_channels[:, i][self.cluster_channels])

    def toggle_unity_regions(self):
        self.unity_region_status = not self.unity_region_status
        self.unitydata.toggle_regions(self.unity_region_status)

    def on_point_size_changed(self):
        self.point_size = self.unity_slider.value() / 100
        self.unitydata.set_point_size(self.point_size)

        #     if self.unity:
        #         self.unity_data = {}
        #         self.unitydata.toggle_regions(False)
        #         self.unitydata.delete_text()
        #         self.unitydata.init()
        #         other_shanks = self.loaddata.get_other_shanks()
        #         for shank in other_shanks:
        #             shank_data = self.prepare_shank_data(shank)
        #             if shank_data is not None:
        #                 self.unity_data[shank['name']] = {'channels': shank_data[0], 'clusters': shank_data[1],
        #                                                   'cluster_idx': shank_data[2]}
        #
        #         # Fill in for the selected shank
        #         clust_idx = dict()
        #         clust_idx['all'] = self.plotdata.clust
        #
        #         for filter_type in ['KS good', 'KS mua', 'IBL good']:
        #             self.plotdata.filter_units(filter_type)
        #             clust_idx[filter_type] = self.plotdata.clust
        #
        #         self.plotdata.filter_units('all')
        #
        #         self.unity_data[self.loaddata.probe_label] = {
        #             'channels': self.plotdata.get_channel_data(),
        #             'clusters': self.plotdata.get_cluster_data(),
        #             'cluster_idx': clust_idx,
        #         }
        #
        #         self.set_unity_xyz()

    def plot_unity(self, plot_type=None):
        plot_type = plot_type or self.unity_plot
        if plot_type == 'probe':
            feature = self.probe_options_group.checkedAction().text()
            points = 'channels'
        else:
            feature = self.img_options_group.checkedAction().text()
            if 'Cluster' not in feature:
                return
            points = 'clusters'

        self.unitydata.add_data(self.unity_data, points, feature, self.filter_type, self.point_size,
                                self.loaddata.brain_atlas)
        self.unity_plot = plot_type

    def save_plots(self, save_path=None):
        #TODOOOOO improve
        """
        Saves all plots from the GUI into folder
        """
        # make folder to save plots to
        try:
            sess_info = (self.loaddata.subj + '_' + str(self.loaddata.date) + '_' +
                         self.loaddata.probe_label + '_')
            image_path_overview = self.probe_path.joinpath('GUI_plots')
            image_path = image_path_overview.joinpath(sess_info[:-1])
        except Exception:
            sess_info = ''
            image_path_overview = self.probe_path.joinpath('GUI_plots')
            image_path = image_path_overview

        if save_path:
            image_path_overview = Path(save_path)

        os.makedirs(image_path_overview, exist_ok=True)
        os.makedirs(image_path, exist_ok=True)
        # Reset all axis, put view back to 1 and remove any reference lines
        self.reset_axis_button_pressed()
        self.set_view(view=1, configure=False)
        self.remove_reference_lines_from_display()

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

        utils.set_font(self.fig_img, 'left', ptsize=15, width=ax_width + 20)
        utils.set_font(self.fig_img, 'bottom', ptsize=15)
        utils.set_font(self.fig_img_cb, 'top', ptsize=15, height=ax_height + 15)

        self.fig_data_area.resize(700, height1)

        plot = None
        start_plot = self.img_options_group.checkedAction()

        while plot != start_plot:
            utils.set_font(self.fig_img_cb, 'top', ptsize=15, height=ax_height + 15)
            exporter = pg.exporters.ImageExporter(self.fig_data_layout.scene())
            exporter.export(str(image_path.joinpath(sess_info + 'img_' +
                                                    self.img_options_group.checkedAction()
                                                    .text() + '.png')))
            self.toggle_plots(self.img_options_group)
            self.remove_reference_lines_from_display()
            plot = self.img_options_group.checkedAction()

        utils.set_font(self.fig_img, 'left', ptsize=8, width=ax_width)
        utils.set_font(self.fig_img, 'bottom', ptsize=8)
        utils.set_font(self.fig_img_cb, 'top', ptsize=8, height=ax_height)
        utils.set_axis(self.fig_img, 'bottom', label=xlabel_img)
        self.fig_data_layout.removeItem(self.fig_img)
        self.fig_data_layout.removeItem(self.fig_img_cb)

        # Next go over probe plots
        self.fig_data_layout.addItem(self.fig_probe_cb, 0, 0, 1, 2)
        self.fig_data_layout.addItem(self.fig_probe, 1, 0)
        utils.set_axis(self.fig_probe, 'left', label='Distance from probe tip (uV)')
        self.fig_probe.setFixedWidth(self.fig_probe_width + self.fig_ax_width + 20)
        utils.set_font(self.fig_probe, 'left', ptsize=15, width=ax_width + 20)
        utils.set_font(self.fig_probe_cb, 'top', ptsize=15, height=ax_height + 15)
        self.fig_data_area.resize(250, height1)

        plot = None
        start_plot = self.probe_options_group.checkedAction()

        while plot != start_plot:
            utils.set_font(self.fig_probe_cb, 'top', ptsize=15, height=ax_height + 15)
            exporter = pg.exporters.ImageExporter(self.fig_data_layout.scene())
            exporter.export(str(image_path.joinpath(sess_info + 'probe_' +
                                                    self.probe_options_group.checkedAction().
                                                    text() + '.png')))
            self.toggle_plots(self.probe_options_group)
            plot = self.probe_options_group.checkedAction()

        self.fig_probe.setFixedWidth(self.fig_probe_width + self.fig_ax_width)
        utils.set_font(self.fig_probe, 'left', ptsize=8, width=ax_width)
        utils.set_font(self.fig_probe_cb, 'top', ptsize=8, height=ax_height)
        utils.set_axis(self.fig_probe, 'bottom', pen='w', label='blank')
        self.fig_data_layout.removeItem(self.fig_probe)
        self.fig_data_layout.removeItem(self.fig_probe_cb)

        # Next go through the line plots
        self.fig_data_layout.addItem(self.fig_probe_cb, 0, 0, 1, 2)
        self.fig_probe_cb.clear()
        text = self.fig_probe_cb.getAxis('top').label.toPlainText()
        utils.set_axis(self.fig_probe_cb, 'top', pen='w')
        self.fig_data_layout.addItem(self.fig_line, 1, 0)

        utils.set_axis(self.fig_line, 'left', label='Distance from probe tip (um)')
        utils.set_font(self.fig_line, 'left', ptsize=15, width=ax_width + 20)
        utils.set_font(self.fig_line, 'bottom', ptsize=15)
        self.fig_data_area.resize(200, height1)

        plot = None
        start_plot = self.line_options_group.checkedAction()
        while plot != start_plot:
            exporter = pg.exporters.ImageExporter(self.fig_data_layout.scene())
            exporter.export(str(image_path.joinpath(sess_info + 'line_' +
                                                    self.line_options_group.checkedAction().
                                                    text() + '.png')))
            self.toggle_plots(self.line_options_group)
            plot = self.line_options_group.checkedAction()

        [self.fig_probe_cb.addItem(cbar) for cbar in self.probe_cbars]
        utils.set_axis(self.fig_probe_cb, 'top', pen='k', label=text)
        utils.set_font(self.fig_line, 'left', ptsize=8, width=ax_width)
        utils.set_font(self.fig_line, 'bottom', ptsize=8)
        utils.set_axis(self.fig_line, 'bottom', label=xlabel_line)
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
            slice_name = self.slice_options_group.checkedAction().text()
            exporter = pg.exporters.ImageExporter(self.fig_slice)
            exporter.export(str(image_path.joinpath(sess_info + 'slice_' + slice_name + '.png')))
            self.toggle_plots(self.slice_options_group)
            plot = self.slice_options_group.checkedAction()

        plot = None
        start_plot = self.slice_options_group.checkedAction()
        while plot != start_plot:
            self.toggle_channel_button_pressed()
            self.traj_line.setData(x=self.xyz_channels[:, 0], y=self.xyz_channels[:, 2],
                                   pen=self.rpen_dot)
            self.fig_slice.addItem(self.traj_line)
            slice_name = self.slice_options_group.checkedAction().text()
            self.fig_slice.setXRange(min=np.min(self.xyz_channels[:, 0]) - 200 / 1e6,
                                     max=np.max(self.xyz_channels[:, 0]) + 200 / 1e6)
            self.fig_slice.setYRange(min=np.min(self.xyz_channels[:, 2]) - 500 / 1e6,
                                     max=np.max(self.xyz_channels[:, 2]) + 500 / 1e6)
            self.fig_slice.resize(50, self.slice_height)
            exporter = pg.exporters.ImageExporter(self.fig_slice)
            exporter.export(
                str(image_path.joinpath(sess_info + 'slice_zoom_' + slice_name + '.png')))
            self.fig_slice.resize(self.slice_width, self.slice_height)
            self.fig_slice.setRange(rect=self.slice_rect)
            self.toggle_plots(self.slice_options_group)
            plot = self.slice_options_group.checkedAction()

        # Save the brain regions image
        utils.set_axis(self.fig_hist_extra_yaxis, 'left')
        # Add labels to show which ones are aligned
        utils.set_axis(self.fig_hist, 'bottom', label='aligned')
        utils.set_font(self.fig_hist, 'bottom', ptsize=12)
        utils.set_axis(self.fig_hist_ref, 'bottom', label='original')
        utils.set_font(self.fig_hist_ref, 'bottom', ptsize=12)
        exporter = pg.exporters.ImageExporter(self.fig_hist_layout.scene())
        exporter.export(str(image_path.joinpath(sess_info + 'hist.png')))
        utils.set_axis(self.fig_hist_extra_yaxis, 'left', pen=None)
        utils.set_font(self.fig_hist, 'bottom', ptsize=8)
        utils.set_axis(self.fig_hist, 'bottom', pen='w', label='blank')
        utils.set_font(self.fig_hist_ref, 'bottom', ptsize=8)
        utils.set_axis(self.fig_hist_ref, 'bottom', pen='w', label='blank')

        make_overview_plot(image_path, sess_info, save_folder=image_path_overview)

        self.add_reference_lines_to_display()



def viewer(probe_id, one=None, histology=False, spike_collection=None, title=None):
    """
    """
    qt.create_app()
    av = MainWindow._get_or_create(probe_id=probe_id, one=one, histology=histology,
                                   spike_collection=spike_collection, title=title)
    av.show()
    return av


def multi_shank_viewer(probe_id, loaddata, title):
    """
    """
    av = MainWindow._get_or_create(title=title, probe_id=probe_id, loaddata=loaddata)
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
