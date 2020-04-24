from PyQt5 import QtWidgets, QtCore, QtGui
from pyqtgraph.widgets import MatplotlibWidget as matplot
import pyqtgraph as pg
import numpy as np
import atlaselectrophysiology.load_data as ld
import atlaselectrophysiology.ColorBar as cb
from random import randrange


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.resize(1600, 800)
        self.setWindowTitle('Electrophysiology Atlas')
        self.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)

        self.init_variables()
        self.init_layout()

        subj = 'ZM_2407'
        date = '2019-11-07'
        probe = 0
        self.loaddata = ld.LoadData(subj, date, sess=1, probe_id=probe)
        self.sdata = self.loaddata.get_scatter_data()
        #self.amplitude_data = self.loaddata.get_amplitude_data()
        self.title_string.setText(f"{subj} {date} probe0{probe}")

        region, label, colour = self.loaddata.get_histology_regions(self.idx)
        self.hist_data['region'][self.idx] = region
        self.hist_data['axis_label'][self.idx] = label
        self.hist_data['colour'][self.idx] = colour
        self.brain_atlas = self.loaddata.brain_atlas

        # Initialise with scatter plot
        self.data_plot = self.plot_scatter()
        # Plot histology reference first, should add another argument
        self.plot_histology(self.fig_hist_ref)
        self.plot_histology(self.fig_hist)
        self.plot_slice()

    def init_layout(self):

        # Make menu bar
        menu_bar = QtWidgets.QMenuBar(self)
        menu_options = menu_bar.addMenu("Plot Options")
        menu_bar.setGeometry(QtCore.QRect(0, 0, 1002, 22))
        menu_options.addAction('Scatter Plot')
        menu_options.addAction('Depth Plot')
        menu_options.addAction('Correlation Plot')
        menu_bar.triggered.connect(self.on_menu_clicked)
        self.setMenuBar(menu_bar)

        # Fit associated items
        hlayout_fit = QtWidgets.QHBoxLayout()
        self.fit_button = QtWidgets.QPushButton('Fit', font=self.font)
        self.fit_button.clicked.connect(self.fit_button_pressed)
        self.fit_string = QtWidgets.QLabel(font=self.font)
        self.idx_string = QtWidgets.QLabel(font=self.font)
        hlayout_fit.addWidget(self.fit_button, stretch=1)
        hlayout_fit.addWidget(self.fit_string, stretch=3)

        # Idx associated items
        hlayout_button = QtWidgets.QHBoxLayout()
        hlayout_string = QtWidgets.QHBoxLayout()
        self.next_button = QtWidgets.QPushButton('Next', font=self.font)
        self.next_button.clicked.connect(self.next_button_pressed)
        self.prev_button = QtWidgets.QPushButton('Previous', font=self.font)
        self.prev_button.clicked.connect(self.prev_button_pressed)
        hlayout_button.addWidget(self.prev_button)
        hlayout_button.addWidget(self.next_button)
        self.idx_string = QtWidgets.QLabel(font=self.font)
        self.tot_idx_string = QtWidgets.QLabel(font=self.font)
        self.reset_button = QtWidgets.QPushButton('Reset', font=self.font)
        self.reset_button.clicked.connect(self.reset_button_pressed)
        hlayout_string.addWidget(self.idx_string)
        hlayout_string.addWidget(self.tot_idx_string)
        vlayout = QtWidgets.QVBoxLayout()
        vlayout.addLayout(hlayout_button)
        vlayout.addLayout(hlayout_string)
        vlayout.addWidget(self.reset_button)

        # Title item
        self.title_string = QtWidgets.QLabel(font=self.title_font)
        self.update_string()

        # Figure items
        self.init_figures()

        # Main Widget
        main_widget = QtGui.QWidget()
        self.setCentralWidget(main_widget)
        # Add everything to the main widget
        layout_main = QtWidgets.QGridLayout()
        layout_main.addWidget(self.fig_data, 0, 0, 10, 4)
        layout_main.addWidget(self.fig_hist, 0, 4, 10, 2)
        layout_main.addWidget(self.fig_hist_ref, 0, 6, 10, 2)
        layout_main.addWidget(self.title_string, 0, 8, 1, 2)
        layout_main.addWidget(self.fig_slice, 1, 8, 3, 2)
        layout_main.addLayout(hlayout_fit, 4, 8, 1, 2)
        layout_main.addWidget(self.fig_fit, 5, 8, 3, 2)
        layout_main.addLayout(vlayout, 8, 8, 2, 2)
        layout_main.setColumnStretch(0, 4)
        layout_main.setColumnStretch(4, 2)
        layout_main.setColumnStretch(6, 2)
        layout_main.setColumnStretch(8, 2)

        main_widget.setLayout(layout_main)

    def init_variables(self):

        # Line styles and fonts
        self.kpen_dot = pg.mkPen(color='k', style=QtCore.Qt.DotLine, width=2)
        self.kpen_solid = pg.mkPen(color='k', style=QtCore.Qt.SolidLine, width=2)
        self.bpen_solid = pg.mkPen(color='b', style=QtCore.Qt.SolidLine, width=3)
        self.font = QtGui.QFont()
        self.font.setPointSize(12)
        self.title_font = QtGui.QFont()
        self.title_font.setPointSize(18)
        self.pad = 0.05

        # Constant variables
        self.probe_tip = 0
        self.probe_top = 3840
        self.probe_extra = 500
        self.view_total = [-2000, 6000]
        #self.depth = np.arange(self.probe_tip, self.probe_top, 20)
        self.depth = np.arange(self.view_total[0], self.view_total[1], 20)
        self.probe_offset = 0

        # Variables to keep track of number of fits (max 10)
        self.idx = 0
        self.total_idx = 0
        self.last_idx = 0
        self.max_idx = 10
        self.current_idx = 0

        # Variables to keep track of applied fits
        self.depth_fit_brain = np.empty(((self.max_idx + 1), len(self.depth)))
        self.depth_fit_brain[0] = self.depth
        self.depth_fit_probe = np.empty(((self.max_idx + 1), len(self.depth)))
        self.depth_fit_probe[0] = self.depth
        self.tot_fit_brain = np.empty((self.max_idx + 1, 2))
        self.tot_fit_brain[0] = [1, 0]
        self.tot_fit_probe = np.empty((self.max_idx + 1, 2))
        self.tot_fit_probe[0] = [1, 0]

        self.hist_data = {
            'region': [0] * (self.max_idx + 1),
            'axis_label': [0] * (self.max_idx + 1),
            'colour': [0] * (self.max_idx + 1),
            'chan_int': 20
        }

        # Variables to keep track of number of lines added
        self.lines = np.empty((0, 2))
        self.points = np.empty((0, 1))
        self.scale = 1

        # Variables for colour bar
        self.color_bar = []
        self.data_plot = []

    def init_figures(self):

        # Create data figure
        self.fig_data = pg.PlotWidget(background='w')
        self.fig_data.setMouseEnabled(x=False, y=False)
        self.fig_data.scene().sigMouseClicked.connect(self.on_mouse_double_clicked)
        self.fig_data.scene().sigMouseHover.connect(self.on_mouse_hover)
        self.fig_data.setYRange(min=self.probe_tip - self.probe_extra, max=self.probe_top +
                                self.probe_extra, padding=self.pad)
        self.fig_data.addLine(y=self.probe_tip, pen=self.kpen_dot)
        self.fig_data.addLine(y=self.probe_top, pen=self.kpen_dot)
        self.fig_data_vb = self.fig_data.getViewBox()
        self.set_axis(self.fig_data)

        # Create histology figure
        self.fig_hist = pg.PlotWidget(background='w')
        self.fig_hist.scene().sigMouseClicked.connect(self.on_mouse_double_clicked)
        self.fig_hist.scene().sigMouseHover.connect(self.on_mouse_hover)
        self.fig_hist.setMouseEnabled(x=False)
        self.fig_hist.setYRange(min=self.probe_tip - self.probe_extra, max=self.probe_top +
                                self.probe_extra, padding=self.pad)
        axis = self.fig_hist.plotItem.getAxis('bottom')
        axis.setTicks([[(0, ''), (0.5, ''), (1, '')]])
        axis.setLabel('')

        # Create reference histology figure
        self.fig_hist_ref = pg.PlotWidget(background='w')
        self.fig_hist_ref.setMouseEnabled(x=False)
        self.fig_hist_ref.setYRange(min=self.probe_tip - self.probe_extra, max=self.probe_top +
                                    self.probe_extra, padding=self.pad)
        axis = self.fig_hist_ref.plotItem.getAxis('bottom')
        axis.setTicks([[(0, ''), (0.5, ''), (1, '')]])
        axis.setLabel('')

        # Create figure showing fit line
        self.fig_fit = pg.PlotWidget(background='w')
        #self.fig_fit.setMouseEnabled(x=False, y=False)
        self.fig_fit.setXRange(min=self.view_total[0], max=self.view_total[1])
        self.fig_fit.setYRange(min=self.view_total[0], max=self.view_total[1])
        self.set_axis(self.fig_fit)
        plot = pg.PlotCurveItem()
        plot.setData(x=self.depth, y=self.depth, pen=self.kpen_dot)
        self.fit_plot = pg.PlotCurveItem(pen=self.kpen_solid)
        self.tot_fit_plot = pg.PlotCurveItem(pen=self.bpen_solid)
        self.fig_fit.addItem(plot)
        self.fig_fit.addItem(self.fit_plot)
        self.fig_fit.addItem(self.tot_fit_plot)

        # Create figure to show coronal slice and probe
        self.fig_slice = matplot.MatplotlibWidget()
        self.fig = self.fig_slice.getFigure()
        self.fig.canvas.toolbar.hide()
        self.fig_slice_ax = self.fig.gca()

    def set_axis(self, fig, label=None):
        """
        Adds label to x and y axis of figure and sets pen of axis to black
        :param fig: pg.PlotWidget() to add labels to
        :param label: 2d array of axis labels [xlabel, ylabel]
        """
        if not label:
            label = ['', '']
        ax_x = fig.plotItem.getAxis('bottom')
        ax_x.setPen('k')
        ax_x.setLabel(label[0])
        ax_y = fig.plotItem.getAxis('left')
        ax_y.setPen('k')
        ax_y.setLabel(label[1])

    """
    Plot updates functions
    """

    def plot_histology(self, fig):
        """
        Plots brain regions along the probe
        :param fig: pg.PlotWidget()
        """
        fig.clear()
        axis = fig.plotItem.getAxis('left')
        axis.setTicks([self.hist_data['axis_label'][self.idx]])
        axis.setPen('k')

        for ir, reg in enumerate(self.hist_data['region'][self.idx]):
            x, y = self.create_hist_data(reg, self.hist_data['chan_int'])
            curve_item = pg.PlotCurveItem()
            curve_item.setData(x=x, y=y, fillLevel=50)
            colour = QtGui.QColor(*self.hist_data['colour'][self.idx][ir])
            curve_item.setBrush(colour)
            fig.addItem(curve_item)

        self.tip_pos = pg.InfiniteLine(pos=self.probe_tip, angle=0, pen=self.kpen_dot,
                                       movable=True)
        self.top_pos = pg.InfiniteLine(pos=self.probe_top, angle=0, pen=self.kpen_dot,
                                       movable=True)
        # Add offset of 1um to stop it reaching upper and lower bounds of interpolation
        offset = 1
        self.tip_pos.setBounds((self.loaddata.track[self.idx][0] * 1e6 + offset,
                                self.loaddata.track[self.idx][-1] * 1e6 -
                                (self.probe_top + offset)))
        self.top_pos.setBounds((self.loaddata.track[self.idx][0] * 1e6 + (self.probe_top + offset),
                                self.loaddata.track[self.idx][-1] * 1e6 - offset))
        self.tip_pos.sigPositionChanged.connect(self.tip_line_moved)
        self.top_pos.sigPositionChanged.connect(self.top_line_moved)

        fig.addItem(self.tip_pos)
        fig.addItem(self.top_pos)

    def tip_line_moved(self):
        self.top_pos.setPos(self.tip_pos.value() + self.probe_top)

    def top_line_moved(self):
        self.tip_pos.setPos(self.top_pos.value() - self.probe_top)

    def create_hist_data(self, reg, chan_int):
        y = np.arange(reg[0], reg[1] + chan_int, chan_int, dtype=int)
        x = np.ones(len(y), dtype=int)
        x = np.r_[0, x, 0]
        y = np.r_[reg[0], y, reg[1]]
        return x, y

    def offset_hist_data(self):

        self.probe_offset += self.tip_pos.value()
        self.loaddata.track[self.idx] = (self.loaddata.track[self.idx_prev] -
                                         self.tip_pos.value() / 1e6)
        self.loaddata.features[self.idx] = (self.loaddata.features[self.idx_prev] -
                                            self.tip_pos.value() / 1e6)
        self.loaddata.track_init[self.idx] = self.loaddata.track_init[self.idx_prev]
        region, label, colour = self.loaddata.get_histology_regions(self.idx)
        self.hist_data['region'][self.idx] = region
        self.hist_data['axis_label'][self.idx] = label
        self.hist_data['colour'][self.idx] = colour

    def scale_hist_data(self):
        # Lines on the histology plot
        line_track = np.array([line[0].pos().y() for line in self.lines]) / 1e6
        # Lines on the data plot
        line_feature = np.array([line[1].pos().y() for line in self.lines]) / 1e6
        depths_track = np.sort(np.r_[self.loaddata.track[self.idx_prev][[0, -1]], line_track])

        self.loaddata.track[self.idx] = self.loaddata.feature2track(depths_track, self.idx_prev)
        self.loaddata.features[self.idx] = np.sort(np.r_[self.loaddata.features[self.idx_prev]
                                                   [[0, -1]], line_feature])
        self.loaddata.track_init[self.idx] = np.sort(np.r_[self.loaddata.track_init[self.idx_prev]
                                                     [[0, -1]], line_feature -
                                                     self.tip_pos.value() / 1e6])

        region, label, colour = self.loaddata.get_histology_regions(self.idx)
        self.hist_data['region'][self.idx] = region
        self.hist_data['axis_label'][self.idx] = label
        self.hist_data['colour'][self.idx] = colour

    def plot_fit(self):
        self.tot_fit_plot.setData(x=self.loaddata.track_init[self.idx] * 1e6,
                                  y=self.loaddata.track[self.idx] * 1e6)

    def update_fit(self, line):
        idx = np.where(self.lines == line)[0][0]
        self.points[idx][0].setData(x=[self.lines[idx][1].pos().y()],
                                    y=[self.lines[idx][0].pos().y()])

    def plot_slice(self):
        """
        Upper right widget containing tilted coronal slice of the Allen brain atlas annotation
        volume overlayed with full track and channel positions
        :return:
        """
        self.fig_slice_ax.cla()
        xyz_trk = self.loaddata.xyz_track
        # recomputes from scratch, for hovering function it would have to be pre-computed
        xyz_ch = self.loaddata.get_channels_coordinates(self.idx)
        self.brain_atlas.plot_tilted_slice(xyz_trk, axis=1, volume='annotation',
                                           ax=self.fig_slice_ax)
        self.fig_slice_ax.plot(xyz_trk[:, 0] * 1e6, xyz_trk[:, 2] * 1e6, 'b')
        self.fig_slice_ax.plot(xyz_ch[:, 0] * 1e6, xyz_ch[:, 2] * 1e6, 'k*')
        self.fig_slice.draw()

    def plot_scatter(self):
        connect = np.zeros(len(self.sdata['times']), dtype=int)
        plot = pg.PlotDataItem()
        plot.setData(x=self.sdata['times'], y=self.sdata['depths'], connect=connect,
                     symbol='o', symbolSize=2)
        self.fig_data.addItem(plot)
        self.fig_data.setXRange(min=self.sdata['times'].min(), max=self.sdata['times'].max())
        self.fig_data.setYRange(min=self.probe_tip - self.probe_extra, max=self.probe_top +
                                self.probe_extra, padding=self.pad)
        self.scale = 1
        return plot

    def plot_image(self):
        self.color_bar = cb.ColorBar('cividis')
        lut = self.color_bar.getColourMap()
        image_plot = pg.ImageItem()
        #img = np.flip(self.amplitude_data['corr'],axis=0)
        img = self.amplitude_data['corr']
        bins = self.amplitude_data['bins']
        self.scale = (bins[-1] - bins[0]) / img.shape[0]
        #image_plot.setImage(img, autoLevels=True)
        image_plot.setImage(img)
        image_plot.scale(self.scale, self.scale)
        image_plot.setLookupTable(lut)
        image_plot.setLevels((img.min(), img.max()))

        self.color_bar.makeColourBar(3000, 150, min=img.min(), max=img.max(), label='Correlation')
        # Make this so it automatically sets based on size of colourbar!!
        self.color_bar.setPos(500, 4500)
        self.fig_data.addItem(image_plot)
        self.fig_data.addItem(self.color_bar)
        self.fig_data.setXRange(min=self.probe_tip, max=self.probe_top)
        self.fig_data.setYRange(min=self.probe_tip - self.probe_extra, max=self.probe_top +
                                self.probe_extra, padding=self.pad)
        return image_plot

    """
    Interaction functions
    """

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Delete:
            if self.selected_line:
                line = np.where(self.lines == self.selected_line)[0][0]
                self.fig_data.removeItem(self.lines[line, 1])
                self.fig_hist.removeItem(self.lines[line, 0])
                self.fig_fit.removeItem(self.points[line, 0])
                self.lines = np.delete(self.lines, line, axis=0)
                self.points = np.delete(self.points, line, axis=0)

        if event.key() == QtCore.Qt.Key_Return:
            self.fit_button_pressed()

        if event.key() == QtCore.Qt.Key_Left:
            self.prev_button_pressed()

        if event.key() == QtCore.Qt.Key_Right:
            self.next_button_pressed()

        if event.key() == QtCore.Qt.Key_O:
            self.offset_button_pressed()

        if event.key() == QtCore.Qt.Key_Down:
            if self.loaddata.track[self.idx][-1] - 50 / 1e6 >= np.max(self.loaddata.channel_coords
                                                                      [:, 1]) / 1e6:
                self.loaddata.features[self.idx] -= 50 / 1e6
                self.loaddata.track[self.idx] -= 50 / 1e6
                self.offset_button_pressed()

        if event.key() == QtCore.Qt.Key_Up:
            if self.loaddata.track[self.idx][0] + 50 / 1e6 <= np.min(self.loaddata.channel_coords
                                                                     [:, 1]) / 1e6:
                self.loaddata.features[self.idx] += 50 / 1e6
                self.loaddata.track[self.idx] += 50 / 1e6
                self.offset_button_pressed()

        if event.key() == QtCore.Qt.Key_H:
            self.remove_lines_points()

        if event.key() == QtCore.Qt.Key_S:
            self.add_lines_points()

    def offset_button_pressed(self):

        if self.current_idx < self.last_idx:
            self.total_idx = np.copy(self.current_idx)
            self.diff_idx = (np.mod(self.last_idx, self.max_idx) - np.mod(self.total_idx,
                                                                          self.max_idx))
            if self.diff_idx >= 0:
                self.diff_idx = self.max_idx - self.diff_idx
                print(self.diff_idx)
            else:
                self.diff_idx = np.abs(self.diff_idx)
                print(self.diff_idx)
        else:
            self.diff_idx = self.max_idx - 1

        self.total_idx += 1
        self.current_idx += 1
        self.idx_prev = np.copy(self.idx)
        self.idx = np.mod(self.current_idx, self.max_idx)
        self.offset_hist_data()
        self.plot_histology(self.fig_hist)
        self.plot_fit()
        self.plot_slice()
        self.remove_lines_points()
        self.add_lines_points()
        self.update_lines_points()
        self.fig_hist.setYRange(min=self.probe_tip - self.probe_extra, max=self.probe_top +
                                self.probe_extra, padding=self.pad)
        self.update_string()

    def fit_button_pressed(self):

        if self.current_idx < self.last_idx:
            self.total_idx = np.copy(self.current_idx)
            self.diff_idx = (np.mod(self.last_idx, self.max_idx) - np.mod(self.total_idx,
                                                                          self.max_idx))
            if self.diff_idx >= 0:
                self.diff_idx = self.max_idx - self.diff_idx
                print(self.diff_idx)
            else:
                self.diff_idx = np.abs(self.diff_idx)
                print(self.diff_idx)
        else:
            self.diff_idx = self.max_idx - 1

        self.total_idx += 1
        self.current_idx += 1
        self.idx_prev = np.copy(self.idx)
        self.idx = np.mod(self.current_idx, self.max_idx)
        self.scale_hist_data()
        self.plot_histology(self.fig_hist)
        self.plot_fit()
        self.plot_slice()
        self.remove_lines_points()
        self.add_lines_points()
        self.update_lines_points()
        self.fig_hist.setYRange(min=self.probe_tip - self.probe_extra, max=self.probe_top +
                                self.probe_extra, padding=self.pad)
        self.update_string()

    def next_button_pressed(self):
        if (self.current_idx < self.total_idx) & (self.current_idx >
                                                  self.total_idx - self.max_idx):
            self.current_idx += 1
            self.idx = np.mod(self.current_idx, self.max_idx)
            self.remove_lines_points()
            self.add_lines_points()
            self.plot_histology(self.fig_hist)
            self.remove_lines_points()
            self.add_lines_points()
            self.plot_fit()
            self.plot_slice()
            self.update_string()

    def prev_button_pressed(self):
        if self.total_idx > self.last_idx:
            self.last_idx = np.copy(self.total_idx)

        if (self.current_idx > np.max([0, self.total_idx - self.diff_idx])):
            self.current_idx -= 1
            self.idx = np.mod(self.current_idx, self.max_idx)
            self.remove_lines_points()
            self.add_lines_points()
            self.plot_histology(self.fig_hist)
            self.remove_lines_points()
            self.add_lines_points()
            self.plot_fit()
            self.plot_slice()
            self.update_string()

    def reset_button_pressed(self):
        self.remove_lines_points()
        self.lines = np.empty((0, 2))
        # self.init_variables()
        self.scale_hist_data()
        self.plot_histology(self.fig_hist)
        self.tot_fit_plot.setData()
        self.fit_plot.setData()
        self.update_string()

    def on_mouse_double_clicked(self, event):
        if event.double():
            pos = self.data_plot.mapFromScene(event.scenePos())
            marker, pen, brush = self.create_line_style()
            line_track = pg.InfiniteLine(pos=pos.y() * self.scale, angle=0, pen=pen, movable=True)
            line_track.sigPositionChangeFinished.connect(self.update_fit)
            line_track.setZValue(100)
            line_feature = pg.InfiniteLine(pos=pos.y() * self.scale, angle=0, pen=pen,
                                           movable=True)
            line_feature.setZValue(100)
            line_feature.sigPositionChangeFinished.connect(self.update_fit)
            self.fig_hist.addItem(line_track)
            self.fig_data.addItem(line_feature)
            self.lines = np.vstack([self.lines, [line_track, line_feature]])

            point = pg.PlotDataItem()
            point.setData(x=[line_track.pos().y()], y=[line_feature.pos().y()],
                          symbolBrush=brush, symbol='o', symbolSize=10)
            self.fig_fit.addItem(point)
            self.points = np.vstack([self.points, point])

    def on_mouse_hover(self, items):
        if items:
            if type(items[0]) == pg.InfiniteLine:
                self.selected_line = items[0]
            else:
                self.selected_line = []

    def on_menu_clicked(self, action):
        self.fig_data.removeItem(self.data_plot)
        self.fig_data.removeItem(self.color_bar)

        if action.text() == 'Scatter Plot':
            self.data_plot = self.plot_scatter()
            self.remove_lines_points()
            self.add_lines_points()

        if action.text() == 'Depth Plot':
            self.data_plot = self.plot_bar()
            self.remove_lines_points()
            self.add_lines_points()
        if action.text() == 'Correlation Plot':
            self.data_plot = self.plot_image()
            self.remove_lines_points()
            self.add_lines_points()

    def remove_lines_points(self):
        for lines, points in zip(self.lines, self.points):
            self.fig_hist.removeItem(lines[0])
            self.fig_data.removeItem(lines[1])
            self.fig_fit.removeItem(points[0])

    def add_lines_points(self):
        for lines, points in zip(self.lines, self.points):
            self.fig_hist.addItem(lines[0])
            self.fig_data.addItem(lines[1])
            self.fig_fit.addItem(points[0])

    def update_lines_points(self):
        for lines, points in zip(self.lines, self.points):
            lines[0].setPos(lines[1].getYPos())
            points[0].setData(x=[lines[1].pos().y()], y=[lines[1].pos().y()])

    def create_line_style(self):
        # Create random choice of line colour and style for infiniteLine
        markers = [['o', 10], ['v', 15]]
        mark = markers[randrange(len(markers))]
        colours = ['#000000', '#cc0000', '#6aa84f', '#1155cc', '#a64d79']
        style = [QtCore.Qt.SolidLine, QtCore.Qt.DashLine, QtCore.Qt.DashDotLine]
        col = QtGui.QColor(colours[randrange(len(colours))])
        sty = style[randrange(len(style))]
        pen = pg.mkPen(color=col, style=sty, width=3)
        brush = pg.mkBrush(color=col)
        return mark, pen, brush

    def update_string(self):
        self.idx_string.setText(f"Current Index = {self.current_idx}")
        self.tot_idx_string.setText(f"Total Index = {self.total_idx}")


if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    mainapp = MainWindow()
    mainapp.show()
    app.exec_()
