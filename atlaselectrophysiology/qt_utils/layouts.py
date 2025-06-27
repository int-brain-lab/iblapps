from qtpy import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
from iblutil.util import Bunch
from atlaselectrophysiology.qt_utils import utils
from atlaselectrophysiology.qt_utils.AdaptedAxisItem import replace_axis
import numpy as np



class GridTabSwitcher(QtWidgets.QWidget):
    custom_signal = QtCore.Signal(str)
    def __init__(self):

        super().__init__()
        self.setFocusPolicy(QtCore.Qt.StrongFocus)

        # Layouts and containers
        self.main_layout = QtWidgets.QVBoxLayout(self)

        self.panels = []
        # Tab widget
        self.tab_widget = QtWidgets.QTabWidget()
        self.tab_widget.setTabPosition(QtWidgets.QTabWidget.South)
        self.tab_widget.hide()

        # TODO add rounded corners
        self.tab_widget.setStyleSheet("""
        QTabBar::tab:selected {
            background-color: #2c3e50;
            color: white;
            font-weight: bold;
        }
        """)

        # Splitter widget
        self.splitter_main = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        #self.splitter_main.setContentsMargins(0, 0, 0, 0)
        self.splitter_top = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.splitter_bottom = QtWidgets.QSplitter(QtCore.Qt.Horizontal)

        # Splitter layout
        self.grid_layout = True

    def initialise(self, panels, names, headers=()):
        self.headers = headers
        self.panels = panels
        self.panel_names = list(names)

        if len(self.panels) == 4:
            self.splitter_main.addWidget(self.splitter_top)
            self.splitter_main.addWidget(self.splitter_bottom)

        self.add_splitter_layout()
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

    def add_splitter_layout(self):

        if len(self.panels) == 1:
            self.splitter_main.addWidget(self.panels[0])
        elif len(self.panels) == 4:
            self.splitter_top.addWidget(self.panels[0])
            self.splitter_top.addWidget(self.panels[1])
            self.splitter_bottom.addWidget(self.panels[2])
            self.splitter_bottom.addWidget(self.panels[3])
        else:
            return

        self.splitter_main.show()
        for panel in self.panels:
            panel.show()

        self.main_layout.addWidget(self.splitter_main)

    def delete_widgets(self):
        if self.grid_layout:
            self.remove_splitter_layout(delete=True)
        else:
            self.remove_tab_layout(delete=True)
        self.panels = []

    def remove_header(self):
        if self.headers:
            for panel, header in zip(self.panels, self.headers):
                panel.layout().removeWidget(header)

    def add_header(self):
        if self.headers:
            for panel, header in zip(self.panels, self.headers):
                panel.layout().insertWidget(0, header)

    def remove_splitter_layout(self, delete=False):

        if len(self.panels) == 1:
            splitters = [self.splitter_main]
        elif len(self.panels) == 4:
            splitters = [self.splitter_top, self.splitter_bottom]
        else:
            return

        for splitter in splitters:
            for i in reversed(range(splitter.count())):
                widget = splitter.widget(i)
                widget.setParent(None)
                if delete:
                    del widget

        self.main_layout.removeWidget(self.splitter_main)
        self.splitter_main.hide()

    def add_tab_layout(self):

        for i, w in enumerate(self.panels):
            self.tab_widget.addTab(w, f"{self.panel_names[i]}")
        self.main_layout.addWidget(self.tab_widget)
        self.tab_widget.show()

    def remove_tab_layout(self, delete=False):

        for i in reversed(range(self.tab_widget.count())):
            widget = self.tab_widget.widget(i)
            widget.setParent(None)
            if delete:
                del widget

        self.main_layout.removeWidget(self.tab_widget)
        self.tab_widget.hide()

    def toggle_layout(self):
        self.tab_widget.blockSignals(True)
        if self.grid_layout:
            # Switch to tab layout
            self.remove_splitter_layout()
            self.remove_header()
            self.add_tab_layout()
        else:
            # Switch to grid layout
            self.remove_tab_layout()
            self.add_header()
            self.add_splitter_layout()
            # Emit so we can signal that we have to add the lines for the fit as we now show 4 displays
            self.custom_signal.emit("lala")

        self.grid_layout = not self.grid_layout
        self.tab_widget.blockSignals(False)



class Setup():

    def init_layout(self, offline=False):
        self.resize(1600, 800)
        self.setWindowTitle('Electrophysiology Atlas')
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.offline = offline

        self.menu_bar = QtWidgets.QMenuBar(self)
        self.menu_bar.setNativeMenuBar(False)
        self.setMenuBar(self.menu_bar)
        # self.menu_bar.setStyleSheet("""QMenuBar {padding-bottom: 10px;}""")

        self.init_session_selection()
        self.init_button_features()
        self.init_fit_figures()
        self.init_slice_figures()
        self.init_shank_tabs()

        self.menu_bar.setCornerWidget(self.selection_widget, corner=QtCore.Qt.TopRightCorner)

        splitter_left = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        splitter_left.addWidget(self.fig_slice_area)
        splitter_left.addWidget(self.fig_fit)

        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(splitter_left, stretch=6)
        layout.addLayout(self.button_layout, stretch=1)
        layout.setContentsMargins(0, 0, 0, 0)
        container.setLayout(layout)

        self.main_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.main_splitter.addWidget(self.shank_tabs)
        self.main_splitter.addWidget(container)
        self.main_splitter.setStretchFactor(0, 80)
        self.main_splitter.setStretchFactor(1, 1)


    def init_display(self):
        self.setCentralWidget(self.main_splitter)

    # TODO
    def clear_tabs(self):
        self.shank_tabs.tab_widget.blockSignals(True)
        self.hist_tabs.tab_widget.blockSignals(True)
        self.shank_tabs.delete_widgets()
        self.hist_tabs.delete_widgets()
        self.docks = []
        self.histology_docks = []
        self.dock_items = {}
        self.shank_tabs.tab_widget.blockSignals(False)
        self.hist_tabs.tab_widget.blockSignals(False)


    def init_tabs(self):
        self.dock_items = dict()
        self.docks = []
        self.histology_docks = []

        headers = []
        for i, probe in enumerate(self.loaddata.probes):
            widget, items = self.create_shank_docks(f'{probe}', i)
            self.docks.append(widget)
            self.dock_items[probe] = items
            self.fig_fit.addItem(items.fit_plot)
            self.fig_fit.addItem(items.fit_scatter)
            self.fig_fit.addItem(items.fit_plot_lin)
            self.histology_docks.append(items.fig_slice_area)
            headers.append(items.header)

        plot = pg.PlotCurveItem()
        plot.setData(x=self.depth, y=self.depth, pen=self.kpen_dot)
        self.fig_fit.addItem(plot)

        self.shank_tabs.initialise(self.docks, self.dock_items.keys(), headers)
        self.hist_tabs.initialise(self.histology_docks, self.dock_items.keys())


    def create_shank_docks(self, name, idx):
        shank_items = self.init_shank_figures(name, idx)

        shank_container = QtWidgets.QWidget()
        shank_container.setContentsMargins(0, 0, 0, 0)

        shank_layout = QtWidgets.QVBoxLayout()
        shank_layout = QtWidgets.QVBoxLayout()
        shank_layout.setContentsMargins(0, 0, 0, 0)
        shank_layout.setSpacing(0)
        shank_layout.addWidget(shank_items.header)
        shank_layout.addWidget(shank_items.fig_area)
        shank_container.setLayout(shank_layout)

        return shank_container, shank_items


    def init_session_selection(self):

        if not self.offline:
            # If offline mode is False, read in Subject and Session options from Alyx
            # Drop down list to choose subject
            self.subj_list, self.subj_combobox = utils.create_combobox(self.on_subject_selected, editable=True)

            # Drop down list to choose session
            self.sess_list, self.sess_combobox = utils.create_combobox(self.on_session_selected)
        else:
            # If offline mode is True, provide dialog to select local folder that holds data
            self.folder_line = QtWidgets.QLineEdit()
            self.folder_button = QtWidgets.QToolButton()
            self.folder_button.setText('...')
            self.folder_button.clicked.connect(self.on_folder_selected)

        # Drop down list to choose previous alignments
        self.align_list, self.align_combobox = utils.create_combobox(self.on_alignment_selected)

        # Drop down list to select shank
        self.shank_list, self.shank_combobox = utils.create_combobox(self.on_shank_selected)

        # Plus button to select alignment file
        self.align_extra = QtWidgets.QToolButton()
        self.align_extra.setText('+')
        self.align_extra.clicked.connect(self.add_alignment_pressed)

        # Button to get data to display in GUI
        self.data_button = QtWidgets.QPushButton('Load')
        self.data_button.clicked.connect(self.data_button_pressed)

        # Layout to group combo
        self.selection_widget = QtWidgets.QWidget()
        self.selection_layout = QtWidgets.QHBoxLayout()
        self.selection_layout.setContentsMargins(0, 0, 0, 0)
        self.selection_widget.setLayout(self.selection_layout)

        if not self.offline:
            self.selection_layout.addWidget(self.subj_combobox)
            self.selection_layout.addWidget(self.sess_combobox)
            self.selection_layout.addWidget(self.shank_combobox)
            self.selection_layout.addWidget(self.align_combobox)
            self.selection_layout.addWidget(self.data_button)
        else:
            self.selection_layout.addWidget(self.folder_line)
            self.selection_layout.addWidget(self.folder_button)
            self.selection_layout.addWidget(self.shank_combobox)
            self.selection_layout.addWidget(self.align_combobox)
            self.selection_layout.addWidget(self.align_extra)
            self.selection_layout.addWidget(self.data_button)



    def init_button_features(self):
        """
        Create all interaction widgets that will be added to the GUI
        """

        # Button to apply interpolation
        self.fit_button = QtWidgets.QPushButton('Fit')
        self.fit_button.clicked.connect(self.fit_button_pressed)
        # Button to apply offset
        self.offset_button = QtWidgets.QPushButton('Offset')
        self.offset_button.clicked.connect(self.offset_button_pressed)
        # String to display current move index
        self.idx_string = QtWidgets.QLabel()
        # String to display total number of moves
        self.tot_idx_string = QtWidgets.QLabel()
        # Button to reset GUI to initial state
        self.reset_button = QtWidgets.QPushButton('Reset')
        self.reset_button.clicked.connect(self.reset_button_pressed)
        # Button to upload final state to Alyx/ to local file
        self.complete_button = QtWidgets.QPushButton('Upload')
        self.complete_button.clicked.connect(self.complete_button_pressed)

        # Arrange interaction features into three different layout groups
        # Group 1
        hlayout1 = QtWidgets.QHBoxLayout()
        hlayout2 = QtWidgets.QHBoxLayout()
        hlayout1.addWidget(self.fit_button, stretch=1)
        hlayout1.addWidget(self.offset_button, stretch=1)
        hlayout1.addWidget(self.tot_idx_string, stretch=2)
        hlayout2.addWidget(self.reset_button, stretch=1)
        hlayout2.addWidget(self.complete_button, stretch=1)
        hlayout2.addWidget(self.idx_string, stretch=2)
        self.button_layout = QtWidgets.QVBoxLayout()
        self.button_layout.addLayout(hlayout1)
        self.button_layout.addLayout(hlayout2)

    def init_slice_figures(self):

        lut_area = pg.GraphicsLayoutWidget()
        lut_layout = pg.GraphicsLayout()
        self.slice_LUT = pg.HistogramLUTItem()
        lut_layout.addItem(self.slice_LUT)
        lut_area.addItem(lut_layout)

        self.hist_tabs = GridTabSwitcher()
        self.hist_tabs.tab_widget.currentChanged.connect(self.hist_tab_changed)

        self.fig_slice_area = QtWidgets.QWidget()
        self.fig_slice_layout = QtWidgets.QHBoxLayout()
        self.fig_slice_layout.setContentsMargins(0, 0, 0, 0)
        self.fig_slice_layout.setSpacing(0)
        self.fig_slice_layout.addWidget(self.hist_tabs, stretch=4)
        self.fig_slice_layout.addWidget(lut_area, stretch=1)
        self.fig_slice_area.setLayout(self.fig_slice_layout)

    def init_shank_tabs(self):

        self.shank_tabs = GridTabSwitcher()
        self.shank_tabs.tab_widget.currentChanged.connect(self.tab_changed)
        self.shank_tabs.custom_signal.connect(self.layout_changed)

    def init_fit_figures(self):

        # Figure to show fit and offset applied by user
        self.fig_fit = pg.PlotWidget(background='w')
        self.fig_fit.setMouseEnabled(x=False, y=False)
        self.fig_fit_exporter = pg.exporters.ImageExporter(self.fig_fit.plotItem)
        self.fig_fit.sigDeviceRangeChanged.connect(self.on_fig_size_changed)
        self.fig_fit.setXRange(min=self.view_total[0], max=self.view_total[1])
        self.fig_fit.setYRange(min=self.view_total[0], max=self.view_total[1])
        utils.set_axis(self.fig_fit, 'bottom', label='Original coordinates (um)')
        utils.set_axis(self.fig_fit, 'left', label='New coordinates (um)')

        self.lin_fit_option = QtWidgets.QCheckBox('Linear fit', self.fig_fit)
        self.lin_fit_option.setChecked(True)
        self.lin_fit_option.stateChanged.connect(self.lin_fit_option_changed)
        self.on_fig_size_changed()


    def init_shank_figures(self, name, idx):
        """
        Create all figures that will be added to the GUI
        """
        items = Bunch()
        items.name = name
        items.idx = idx

        items.probe_top_lines = []
        items.probe_tip_lines = []

        # Figures to show ephys data
        # 2D scatter/ image plot
        items.fig_img = pg.PlotItem()
        items.fig_img.setYRange(min=self.probe_tip - self.probe_extra,
                                max=self.probe_top + self.probe_extra, padding=self.pad)
        items.probe_tip_lines.append(items.fig_img.addLine(y=self.probe_tip, pen=utils.kpen_dot, z=50))
        items.probe_top_lines.append(items.fig_img.addLine(y=self.probe_top, pen=utils.kpen_dot, z=50))
        utils.set_axis(items.fig_img, 'bottom')
        items.fig_data_ax = utils.set_axis(items.fig_img, 'left', label='Distance from probe tip (uV)')

        items.fig_img_cb = pg.PlotItem()
        items.fig_img_cb.setMaximumHeight(70)
        items.fig_img_cb.setMouseEnabled(x=False, y=False)
        utils.set_axis(items.fig_img_cb, 'bottom', show=False)
        utils.set_axis(items.fig_img_cb, 'left', pen='w')
        utils.set_axis(items.fig_img_cb, 'top', pen='w')

        # 1D line plot
        items.fig_line = pg.PlotItem()
        items.fig_line.setMouseEnabled(x=False, y=False)
        items.fig_line.setYRange(min=self.probe_tip - self.probe_extra,
                                 max=self.probe_top + self.probe_extra, padding=self.pad)
        items.probe_tip_lines.append(items.fig_line.addLine(y=self.probe_tip, pen=utils.kpen_dot, z=50))
        items.probe_top_lines.append(items.fig_line.addLine(y=self.probe_top, pen=utils.kpen_dot, z=50))
        utils.set_axis(items.fig_line, 'bottom')
        utils.set_axis(items.fig_line, 'left', show=False)

        # 2D probe plot
        items.fig_probe = pg.PlotItem()
        items.fig_probe.setMouseEnabled(x=False, y=False)
        items.fig_probe.setMaximumWidth(50)
        items.fig_probe.setYRange(min=self.probe_tip - self.probe_extra,
                                  max=self.probe_top + self.probe_extra, padding=self.pad)
        items.probe_tip_lines.append(items.fig_probe.addLine(y=self.probe_tip, pen=utils.kpen_dot, z=50))
        items.probe_top_lines.append(items.fig_probe.addLine(y=self.probe_top, pen=utils.kpen_dot, z=50))
        utils.set_axis(items.fig_probe, 'bottom', pen='w')
        utils.set_axis(items.fig_probe, 'left', show=False)

        items.fig_probe_cb = pg.PlotItem()
        items.fig_probe_cb.setMouseEnabled(x=False, y=False)
        items.fig_probe_cb.setMaximumHeight(70)
        utils.set_axis(items.fig_probe_cb, 'bottom', show=False)
        utils.set_axis(items.fig_probe_cb, 'left', pen='w')
        utils.set_axis(items.fig_probe_cb, 'top', pen='w')

        # Add img plot, line plot, probe plot, img colourbar and probe colourbar to a graphics
        # layout widget so plots can be arranged and moved easily
        items.fig_data_area = pg.GraphicsLayoutWidget(border=None)
        items.fig_data_area.setContentsMargins(0, 0, 0, 0)
        items.fig_data_area.ci.setContentsMargins(0, 0, 0, 0)
        items.fig_data_area.ci.layout.setSpacing(0)
        items.fig_data_area.scene().sigMouseClicked.connect(lambda event, i=idx: self.on_mouse_double_clicked(event, i))
        items.fig_data_area.scene().sigMouseHover.connect(self.on_mouse_hover)
        items.fig_data_layout = pg.GraphicsLayout()

        items.fig_data_layout.addItem(items.fig_img_cb, 0, 0)
        items.fig_data_layout.addItem(items.fig_probe_cb, 0, 1, 1, 2)
        items.fig_data_layout.addItem(items.fig_img, 1, 0)
        items.fig_data_layout.addItem(items.fig_line, 1, 1)
        items.fig_data_layout.addItem(items.fig_probe, 1, 2)
        items.fig_data_layout.layout.setColumnStretchFactor(0, 6)
        items.fig_data_layout.layout.setColumnStretchFactor(1, 2)
        items.fig_data_layout.layout.setColumnStretchFactor(2, 1)
        items.fig_data_layout.layout.setRowStretchFactor(0, 1)
        items.fig_data_layout.layout.setRowStretchFactor(1, 10)

        items.fig_data_area.addItem(items.fig_data_layout)

        # Figures to show histology data
        # Histology figure that will be updated with user input
        items.fig_hist = pg.PlotItem()
        items.fig_hist.setContentsMargins(0, 0, 0, 0)
        items.fig_hist.setMouseEnabled(x=False)
        items.fig_hist.setYRange(min=self.probe_tip - self.probe_extra,
                                 max=self.probe_top + self.probe_extra, padding=self.pad)
        utils.set_axis(items.fig_hist, 'bottom', pen='w')

        # This is the solution from pyqtgraph people, but doesn't show ticks
        # self.fig_hist.showGrid(False, True, 0)

        replace_axis(items.fig_hist)
        items.ax_hist = utils.set_axis(items.fig_hist, 'left', pen=None)
        items.ax_hist.setWidth(0)
        items.ax_hist.setStyle(tickTextOffset=-70)

        items.fig_scale = pg.PlotItem()
        items.fig_scale.setMaximumWidth(50)
        items.fig_scale.setMouseEnabled(x=False)
        items.scale_label = pg.LabelItem(color='k')
        utils.set_axis(items.fig_scale, 'bottom', pen='w')
        utils.set_axis(items.fig_scale, 'left', show=False)
        (items.fig_scale).setYLink(items.fig_hist)

        # Figure that will show scale factor of histology boundaries
        items.fig_scale_cb = pg.PlotItem()
        items.fig_scale_cb.setMouseEnabled(x=False, y=False)
        items.fig_scale_cb.setMaximumHeight(70)
        utils.set_axis(items.fig_scale_cb, 'bottom', show=False)
        utils.set_axis(items.fig_scale_cb, 'left', show=False)
        items.fig_scale_ax = utils.set_axis(items.fig_scale_cb, 'top', pen='w')
        utils.set_axis(items.fig_scale_cb, 'right', show=False)

        # Histology figure that will remain at initial state for reference
        items.fig_hist_ref = pg.PlotItem()
        items.fig_hist_ref.setMouseEnabled(x=False)
        items.fig_hist_ref.setYRange(min=self.probe_tip - self.probe_extra,
                                     max=self.probe_top + self.probe_extra, padding=self.pad)
        utils.set_axis(items.fig_hist_ref, 'bottom', pen='w')
        utils.set_axis(items.fig_hist_ref, 'left', show=False)
        replace_axis(items.fig_hist_ref, orientation='right', pos=(2, 2))
        items.ax_hist_ref = utils.set_axis(items.fig_hist_ref, 'right', pen=None)
        items.ax_hist_ref.setWidth(0)
        items.ax_hist_ref.setStyle(tickTextOffset=-70)

        items.fig_hist_area = pg.GraphicsLayoutWidget(border=None)
        items.fig_hist_area.setContentsMargins(0, 0, 0, 0)
        items.fig_hist_area.ci.setContentsMargins(0, 0, 0, 0)
        items.fig_hist_area.ci.layout.setSpacing(0)
        items.fig_hist_area.setMouseTracking(True)
        items.fig_hist_area.scene().sigMouseClicked.connect(lambda event, i=idx: self.on_mouse_double_clicked(event, i))
        items.fig_hist_area.scene().sigMouseHover.connect(self.on_mouse_hover)

        items.fig_hist_extra_yaxis = pg.PlotItem()
        items.fig_hist_extra_yaxis.setMouseEnabled(x=False, y=False)
        items.fig_hist_extra_yaxis.setMaximumWidth(2)
        items.fig_hist_extra_yaxis.setYRange(min=self.probe_tip - self.probe_extra,
                                             max=self.probe_top + self.probe_extra, padding=self.pad)

        utils.set_axis(items.fig_hist_extra_yaxis, 'bottom', pen='w')
        items.ax_hist2 = utils.set_axis(items.fig_hist_extra_yaxis, 'left', pen=None)
        items.ax_hist2.setWidth(10)

        items.fig_hist_layout = pg.GraphicsLayout()
        items.fig_hist_layout.addItem(items.fig_scale_cb, 0, 0, 1, 4)
        items.fig_hist_layout.addItem(items.fig_hist_extra_yaxis, 1, 0)
        items.fig_hist_layout.addItem(items.fig_hist, 1, 1)
        items.fig_hist_layout.addItem(items.fig_scale, 1, 2)
        items.fig_hist_layout.addItem(items.fig_hist_ref, 1, 3)
        items.fig_hist_layout.layout.setColumnStretchFactor(0, 1)
        items.fig_hist_layout.layout.setColumnStretchFactor(1, 4)
        items.fig_hist_layout.layout.setColumnStretchFactor(2, 1)
        items.fig_hist_layout.layout.setColumnStretchFactor(3, 4)
        items.fig_hist_layout.layout.setRowStretchFactor(0, 1)
        items.fig_hist_layout.layout.setRowStretchFactor(1, 10)
        items.fig_hist_area.addItem(items.fig_hist_layout)

        # Figure to show coronal slice through the brain
        items.fig_slice_area = pg.GraphicsLayoutWidget(border=None)
        items.fig_slice_area.setContentsMargins(0, 0, 0, 0)
        items.fig_slice_area.ci.setContentsMargins(0, 0, 0, 0)
        items.fig_slice_area.ci.layout.setSpacing(0)
        items.fig_slice = pg.ViewBox(enableMenu=False)
        items.fig_slice.setContentsMargins(0, 0, 0, 0)
        items.fig_slice_area.addItem(items.fig_slice)


        items.pen = pg.mkPen(color=utils.colours[idx], style=QtCore.Qt.SolidLine, width=3)
        items.pen_dot = pg.mkPen(color=utils.colours[idx], style=QtCore.Qt.DotLine, width=2)
        items.brush = pg.mkBrush(color=utils.colours[idx])
        items.colour = QtGui.QColor(utils.colours[idx])


        items.fit_plot = pg.PlotCurveItem(pen=items.pen)
        items.fit_scatter = pg.ScatterPlotItem(size=7, symbol='o', brush='w', pen=items.pen)
        items.fit_plot_lin = pg.PlotCurveItem(pen=items.pen_dot)

        items.header = QtWidgets.QLabel(name)
        items.header.setAlignment(QtCore.Qt.AlignCenter)

        items.fig_area = QtWidgets.QWidget()
        items.fig_area_layout = QtWidgets.QHBoxLayout()
        items.fig_area_layout.setContentsMargins(0, 0, 0, 0)
        items.fig_area_layout.setSpacing(0)
        items.fig_area_layout.addWidget(items.fig_data_area)
        items.fig_area_layout.addWidget(items.fig_hist_area)
        items.fig_area_layout.setStretch(0, 3)
        items.fig_area_layout.setStretch(1, 1)
        items.fig_area.setLayout(items.fig_area_layout)

        return items

    def on_fig_size_changed(self):
        # fig_width = self.fig_fit_exporter.getTargetRect().width()
        # fig_height = self.fig_fit_exporter.getTargetRect().width()
        self.lin_fit_option.move(70, 10)


    # TODO make it so only the options are changed menu bar stays the same
    def init_menubar(self):
        """
        Create menu bar and add all possible menu options. These are:
            - Image Plots: possible 2D image/scatter plots
            - Line Plots: possible 1D line plots
            - Probe Plots: possible 2D plots arranged according to probe geometry
            - Slice Plots: possible coronal slice images
            - Filter Units: filter displayed plots by unit type (All, Good, MUA)
            - Fit Options: possible keyboard interactions for applying alignment
            - Display Options: possible keyboard interactions to what is displayed on GUI
            - Session Information: extra info, session notes and Allen brain regions description
        """
        # Create menubar widget and add it to the main GUI window

        # Add menu bar for 2D image plot options
        self.img_options_group, img_options, self.img_init = utils.create_action_menu(
            self.menu_bar, self.shank.img_plots.keys(), self.plot_image_panels, title='Image Plots')

        # Attach scatter plot options to this menu bar
        _ = utils.create_action_menu(
            self.menu_bar, self.shank.scatter_plots.keys(), self.plot_scatter_panels, set_checked=False,
            action_menu=img_options, action_group=self.img_options_group)

        # Add menu bar for 1D line plot options
        self.line_options_group, _, self.line_init = utils.create_action_menu(
            self.menu_bar, self.shank.line_plots.keys(), self.plot_line_panels, title='Line Plots')

        # Add menu bar for 2D probe plot options
        self.probe_options_group, _, self.probe_init = utils.create_action_menu(
            self.menu_bar, self.shank.probe_plots.keys(), self.plot_probe_panels, title='Probe Plots')


        # Add menu bar for coronal slice plot options
        self.slice_options_group, _, self.slice_init = utils.create_action_menu(
            self.menu_bar, self.shank.slice_plots.keys(),self.plot_slice_panels, title='Slice Plots')


        # Add menu bar for unit filtering options
        # TODO get the filter options from the clusters.metrics
        _, _, self.unit_init = utils.create_action_menu(
            self.menu_bar,['All', 'KS good', 'KS mua', 'IBL good'], self.filter_unit_pressed, title='Filter Units')

        # Add menu bar for all fitting options
        fit_options = self.menu_bar.addMenu("Fit Options")

        # Add menu bar for all display options
        display_options = self.menu_bar.addMenu('Display Options')

        # Define all possible keyboard shortcut interactions for GUI
        keyboard = {
            'Fit': # Shortcuts to apply fit
                {'shortcut': 'Return', 'callback': self.fit_button_pressed, 'menu': fit_options},
            'Offset': # Shortcuts to apply offset
                {'shortcut': 'O', 'callback': self.offset_button_pressed, 'menu': fit_options},
            'Offset + 100um':
                {'shortcut': 'Shift+Up', 'callback': self.moveup_button_pressed, 'menu': fit_options},
            'Offset - 100um':
                {'shortcut': 'Shift+Down', 'callback': self.movedown_button_pressed, 'menu': fit_options},
            'Remove Line': # Shortcut to remove a reference line
                {'shortcut': 'Shift+D', 'callback': self.delete_reference_line, 'menu': fit_options},
            'Next': # Shortcut to move between previous/next moves
                {'shortcut': 'Right', 'callback': self.next_button_pressed, 'menu': fit_options},
            'Previous':
                {'shortcut': 'Left', 'callback': self.prev_button_pressed, 'menu': fit_options},
            'Reset': # Shortcut to reset GUI to initial state
                {'shortcut': 'Shift+R', 'callback': self.reset_button_pressed, 'menu': fit_options},
            'Upload': # Shortcut to upload final state to Alyx/to local file
                {'shortcut': 'Shift+U', 'callback': self.complete_button_pressed, 'menu': fit_options},
            'Toggle Image Plots': # Shortcuts to toggle between plots options
                {'shortcut': 'Alt+1', 'callback': lambda:
                self.toggle_plots(self.img_options_group), 'menu': display_options},
            'Toggle Line Plots':
                {'shortcut': 'Alt+2', 'callback': lambda:
                self.toggle_plots(self.line_options_group), 'menu': display_options},
            'Toggle Probe Plots':
                {'shortcut': 'Alt+3', 'callback': lambda:
                self.toggle_plots(self.probe_options_group), 'menu': display_options},
            'Toggle Slice Plots':
                {'shortcut': 'Alt+4', 'callback': lambda:
                self.toggle_plots(self.slice_options_group), 'menu': display_options},
            'View 1': # Shortcuts to switch order of 3 panels in ephys plot
                {'shortcut': 'Shift+1', 'callback': lambda: self.set_view(view=1), 'menu': display_options},
            'View 2':
                {'shortcut': 'Shift+2', 'callback': lambda: self.set_view(view=2), 'menu': display_options},
            'View 3':
                {'shortcut': 'Shift+3', 'callback': lambda: self.set_view(view=3), 'menu': display_options},
            'Reset Axis': # Shortcut to reset axis on figures
                {'shortcut': 'Shift+A', 'callback': self.reset_axis_button_pressed, 'menu': display_options},
            'Hide/Show Labels': # Shortcut to hide/show region labels
                {'shortcut': 'Shift+L', 'callback': self.toggle_labels, 'menu': display_options},
            'Hide/Show Lines': # Shortcut to hide/show reference lines
                {'shortcut': 'Shift+H', 'callback': self.toggle_reference_lines, 'menu': display_options},
            'Hide/Show Channels': # Shortcut to hide/show reference lines and channels on slice image
                {'shortcut': 'Shift+C', 'callback': self.toggle_channels, 'menu': display_options},
            'Hide/Show Nearby Boundaries': # Shortcut to change default histology reference image
                {'shortcut': 'Shift+N', 'callback': self.toggle_histology, 'menu': display_options},
            'Change Histology Map': # Option to change histology regions from Allen to Franklin Paxinos
                {'shortcut': 'Shift+M', 'callback': self.toggle_histology_map, 'menu': display_options},
            'Toggle layout':  # Option to change histology regions from Allen to Franklin Paxinos
                {'shortcut': 'T', 'callback': self.toggle_layout, 'menu': display_options},

        }

        # Add all these shortcuts and options onto the relevant menu bar
        for key, val in keyboard.items():
            option = QtWidgets.QAction(key, self)
            shortcut = val.get('shortcut', None)
            if shortcut:
                option.setShortcut(shortcut)
            option.triggered.connect(val['callback'])
            val['menu'].addAction(option)


        # Plugin menu bar
        self.plugin_options = self.menu_bar.addMenu('Plugins')
        self.plugins = dict()

        # TODO
        # setup_cluster_popup(self)
        # setup_region_tree(self)
        # setup_export_pngs(self)
        #if not self.offline:
            # setup_nearby_sessions(self)
            # setup_subject_scaling(self)
            # setup_ephys_features(self)
            # setup_session_notes(self)
            # setup_qc_dialog(self)
