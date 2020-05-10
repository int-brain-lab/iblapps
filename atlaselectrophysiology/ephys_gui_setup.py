from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
from pyqtgraph.widgets import MatplotlibWidget as matplot
import pyqtgraph.exporters
import numpy as np
from random import randrange
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')


class Setup():
    def init_layout(self, main_window):
        self.resize(1600, 800)
        self.setWindowTitle('Electrophysiology Atlas')
        self.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)

        main_widget = QtWidgets.QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QtWidgets.QGridLayout()

        self.init_menubar()
        self.init_interaction_features()
        self.init_figures()

        main_layout = QtWidgets.QGridLayout()
        main_layout.addWidget(self.fig_data_area, 0, 0, 10, 1)
        main_layout.addWidget(self.fig_hist_area, 0, 1, 10, 1)
        main_layout.addLayout(self.interaction_layout3, 0, 2, 1, 1)
        main_layout.addWidget(self.fig_slice, 1, 2, 3, 1)
        main_layout.addLayout(self.interaction_layout1, 4, 2, 1, 1)
        main_layout.addWidget(self.fig_fit, 5, 2, 3, 1)
        main_layout.addLayout(self.interaction_layout2, 8, 2, 2, 1)
        main_layout.setColumnStretch(0, 6)
        main_layout.setColumnStretch(1, 2)
        main_layout.setColumnStretch(2, 2)

        main_widget.setLayout(main_layout)

    def init_menubar(self):
        """
        Create menu bar and add all possible plot and keyboard interaction options
        """
        # Create menubar widget and add it to the main GUI window
        menu_bar = QtWidgets.QMenuBar(self)
        self.setMenuBar(menu_bar)

        # Define all 2D scatter/ image plot options
        scatter_drift = QtGui.QAction('Amplitude', self, checkable=True, checked=False)
        scatter_drift.triggered.connect(lambda: self.plot_scatter(self.scat_drift_data))
        scatter_fr = QtGui.QAction('Cluster Amp vs Depth vs FR', self, checkable=True,
                                   checked=False)
        scatter_fr.triggered.connect(lambda: self.plot_scatter(self.scat_fr_data))
        scatter_p2t = QtGui.QAction('Cluster Amp vs Depth vs Duration', self, checkable=True,
                                    checked=False)
        scatter_p2t.triggered.connect(lambda: self.plot_scatter(self.scat_p2t_data))
        scatter_amp = QtGui.QAction('Cluster FR vs Depth vs Amp', self, checkable=True,
                                    checked=False)
        scatter_amp.triggered.connect(lambda: self.plot_scatter(self.scat_amp_data))
        img_fr = QtGui.QAction('Firing Rate', self, checkable=True, checked=True)
        img_fr.triggered.connect(lambda: self.plot_image(self.img_fr_data))
        img_corr = QtGui.QAction('Correlation', self, checkable=True, checked=False)
        img_corr.triggered.connect(lambda: self.plot_image(self.img_corr_data))
        img_rmsAP = QtGui.QAction('rms AP', self, checkable=True, checked=False)
        img_rmsAP.triggered.connect(lambda: self.plot_image(self.img_rms_APdata))
        img_rmsLFP = QtGui.QAction('rms LFP', self, checkable=True, checked=False)
        img_rmsLFP.triggered.connect(lambda: self.plot_image(self.img_rms_LFPdata))
        img_LFP = QtGui.QAction('LFP Spectrum', self, checkable=True, checked=False)
        img_LFP.triggered.connect(lambda: self.plot_image(self.img_lfp_data))
        self.img_init = img_fr
        # Add menu bar for 2D scatter/ image plot options
        img_options = menu_bar.addMenu('Image Plots')
        img_options_group = QtGui.QActionGroup(img_options)
        img_options_group.setExclusive(True)
        img_options.addAction(img_fr)
        img_options_group.addAction(img_fr)
        img_options.addAction(scatter_drift)
        img_options_group.addAction(scatter_drift)
        img_options.addAction(img_corr)
        img_options_group.addAction(img_corr)
        img_options.addAction(img_rmsAP)
        img_options_group.addAction(img_rmsAP)
        img_options.addAction(img_rmsLFP)
        img_options_group.addAction(img_rmsLFP)
        img_options.addAction(img_LFP)
        img_options_group.addAction(img_LFP)
        img_options.addAction(scatter_fr)
        img_options_group.addAction(scatter_fr)
        img_options.addAction(scatter_p2t)
        img_options_group.addAction(scatter_p2t)
        img_options.addAction(scatter_amp)
        img_options_group.addAction(scatter_amp)

        # Define all 1D line plot options
        line_fr = QtGui.QAction('Firing Rate', self, checkable=True, checked=True)
        line_fr.triggered.connect(lambda: self.plot_line(self.line_fr_data))
        line_amp = QtGui.QAction('Amplitude', self, checkable=True, checked=False)
        line_amp.triggered.connect(lambda: self.plot_line(self.line_amp_data))

        self.line_init = line_fr
        # Add menu bar for 1D line plot options
        line_options = menu_bar.addMenu('Line Plots')
        line_options_group = QtGui.QActionGroup(line_options)
        line_options_group.setExclusive(True)
        line_options.addAction(line_fr)
        line_options_group.addAction(line_fr)
        line_options.addAction(line_amp)
        line_options_group.addAction(line_amp)

        # Define all 2D probe plot options
        probe_options = menu_bar.addMenu("Probe Plots")
        probe_options_group = QtGui.QActionGroup(probe_options)
        probe_options_group.setExclusive(True)
        probe_rmsAP = QtGui.QAction('rms AP', self, checkable=True, checked=True)
        probe_rmsAP.triggered.connect(lambda: self.plot_probe(self.probe_rms_APdata))
        probe_rmsLFP = QtGui.QAction('rms LFP', self, checkable=True, checked=False)
        probe_rmsLFP.triggered.connect(lambda: self.plot_probe(self.probe_rms_LFPdata))
        self.probe_init = probe_rmsAP

        # Add menu bar for 2D probe plot options
        probe_options.addAction(probe_rmsAP)
        probe_options_group.addAction(probe_rmsAP)
        probe_options.addAction(probe_rmsLFP)
        probe_options_group.addAction(probe_rmsLFP)

        # Add the different frequency band options in a loop. These must be the same as in
        # load_data
        freq_bands = np.vstack(([0, 4], [4, 10], [10, 30], [30, 80], [80, 200]))
        for iF, freq in enumerate(freq_bands):
            band = f"{freq[0]} - {freq[1]} Hz"
            probe = QtGui.QAction(band, self, checkable=True, checked=False)
            probe.triggered.connect(lambda checked, item=band: self.plot_probe(
                                    self.probe_lfp_data[item]))
            probe_options.addAction(probe)
            probe_options_group.addAction(probe)

        # Define unit filter options
        all_units = QtGui.QAction('All', self, checkable=True, checked=True)
        all_units.triggered.connect(lambda: self.filter_unit_pressed('all'))
        good_units = QtGui.QAction('Good', self, checkable=True, checked=False)
        good_units.triggered.connect(lambda: self.filter_unit_pressed('good'))
        mua_units = QtGui.QAction('MUA', self, checkable=True, checked=False)
        mua_units.triggered.connect(lambda: self.filter_unit_pressed('mua'))
        self.unit_init = all_units

        unit_filter_options = menu_bar.addMenu("Filter Units")
        unit_filter_options_group = QtGui.QActionGroup(unit_filter_options)
        unit_filter_options_group.setExclusive(True)
        unit_filter_options.addAction(all_units)
        unit_filter_options_group.addAction(all_units)
        unit_filter_options.addAction(good_units)
        unit_filter_options_group.addAction(good_units)
        unit_filter_options.addAction(mua_units)
        unit_filter_options_group.addAction(mua_units)

        # Define all possible keyboard interactions for GUI
        # Shortcut to apply interpolation
        fit_option = QtGui.QAction('Fit', self)
        fit_option.setShortcut('Return')
        fit_option.triggered.connect(self.fit_button_pressed)
        # Shortcuts to apply offset
        offset_option = QtGui.QAction('Offset', self)
        offset_option.setShortcut('O')
        offset_option.triggered.connect(self.offset_button_pressed)
        moveup_option = QtGui.QAction('Offset + 50um', self)
        moveup_option.setShortcut('Shift+Up')
        moveup_option.triggered.connect(self.moveup_button_pressed)
        movedown_option = QtGui.QAction('Offset - 50um', self)
        movedown_option.setShortcut('Shift+Down')
        movedown_option.triggered.connect(self.movedown_button_pressed)
        # Shortcut to hide/show region labels
        toggle_labels_option = QtGui.QAction('Hide/Show Labels', self)
        toggle_labels_option.setShortcut('Shift+L')
        toggle_labels_option.triggered.connect(self.toggle_labels_button_pressed)
        # Shortcut to hide/show reference lines
        toggle_lines_option = QtGui.QAction('Hide/Show Lines', self)
        toggle_lines_option.setShortcut('Shift+H')
        toggle_lines_option.triggered.connect(self.toggle_line_button_pressed)
        # Shortcut to remove a reference line
        delete_line_option = QtGui.QAction('Remove Line', self)
        delete_line_option.setShortcut('Del')
        delete_line_option.triggered.connect(self.delete_line_button_pressed)
        # Shortcut to reset axis on histology figure
        axis_option = QtGui.QAction('Reset Axis', self)
        axis_option.setShortcut('Shift+A')
        axis_option.triggered.connect(self.reset_axis_button_pressed)
        # Shortcut to move between previous/next moves
        next_option = QtGui.QAction('Next', self)
        next_option.setShortcut('Right')
        next_option.triggered.connect(self.next_button_pressed)
        prev_option = QtGui.QAction('Previous', self)
        prev_option.setShortcut('Left')
        prev_option.triggered.connect(self.prev_button_pressed)
        # Shortcut to reset GUI to initial state
        reset_option = QtGui.QAction('Reset', self)
        reset_option.setShortcut('Shift+R')
        # Shortcut to upload final state to Alyx
        reset_option.triggered.connect(self.reset_button_pressed)
        complete_option = QtGui.QAction('Upload', self)
        complete_option.setShortcut('Shift+U')
        complete_option.triggered.connect(self.complete_button_pressed)

        # Shortcuts to switch between different views on left most data plot
        view1_option = QtGui.QAction('View 1', self)
        view1_option.setShortcut('Shift+1')
        view1_option.triggered.connect(lambda: self.set_view(view=1))
        view2_option = QtGui.QAction('View 2', self)
        view2_option.setShortcut('Shift+2')
        view2_option.triggered.connect(lambda: self.set_view(view=2))
        view3_option = QtGui.QAction('View 3', self)
        view3_option.setShortcut('Shift+3')
        view3_option.triggered.connect(lambda: self.set_view(view=3))

        toggle1_option = QtGui.QAction('Toggle Image Plots', self)
        toggle1_option.setShortcut('Alt+1')
        toggle1_option.triggered.connect(lambda: self.toggle_plots(img_options_group))
        toggle2_option = QtGui.QAction('Toggle Line Plots', self)
        toggle2_option.setShortcut('Alt+2')
        toggle2_option.triggered.connect(lambda: self.toggle_plots(line_options_group))
        toggle3_option = QtGui.QAction('Toggle Probe Plots', self)
        toggle3_option.setShortcut('Alt+3')
        toggle3_option.triggered.connect(lambda: self.toggle_plots(probe_options_group))

        # Add menu bar with all possible keyboard interactions
        shortcut_options = menu_bar.addMenu("Shortcut Keys")
        shortcut_options.addAction(fit_option)
        shortcut_options.addAction(offset_option)
        shortcut_options.addAction(moveup_option)
        shortcut_options.addAction(movedown_option)
        shortcut_options.addAction(toggle_labels_option)
        shortcut_options.addAction(toggle_lines_option)
        shortcut_options.addAction(delete_line_option)
        shortcut_options.addAction(axis_option)
        shortcut_options.addAction(next_option)
        shortcut_options.addAction(prev_option)
        shortcut_options.addAction(reset_option)
        shortcut_options.addAction(complete_option)
        shortcut_options.addAction(toggle1_option)
        shortcut_options.addAction(toggle2_option)
        shortcut_options.addAction(toggle3_option)
        shortcut_options.addAction(view1_option)
        shortcut_options.addAction(view2_option)
        shortcut_options.addAction(view3_option)

        popup_minimise = QtGui.QAction('Minimise/Show', self)
        popup_minimise.setShortcut('Alt+M')
        popup_minimise.triggered.connect(self.minimise_popups)
        popup_close = QtGui.QAction('Close', self)
        popup_close.setShortcut('Alt+X')
        popup_close.triggered.connect(self.close_popups)

        popup_options = menu_bar.addMenu('Popup Windows')
        popup_options.addAction(popup_minimise)
        popup_options.addAction(popup_close)

        notes_options = menu_bar.addMenu('Session Notes')
        show_notes = QtGui.QAction('Display', self)
        show_notes.triggered.connect(self.display_session_notes)
        notes_options.addAction(show_notes)

    def init_interaction_features(self):
        """
        Create all interaction widgets that will be added to the GUI
        """
        # Button to apply interpolation
        self.fit_button = QtWidgets.QPushButton('Fit')
        self.fit_button.clicked.connect(self.fit_button_pressed)
        # Button to apply offset
        self.offset_button = QtWidgets.QPushButton('Offset')
        self.offset_button.clicked.connect(self.offset_button_pressed)
        # Button to go to next move
        self.next_button = QtWidgets.QPushButton('Next')
        self.next_button.clicked.connect(self.next_button_pressed)
        # Button to go to previous move
        self.prev_button = QtWidgets.QPushButton('Previous')
        self.prev_button.clicked.connect(self.prev_button_pressed)
        # String to display current move index
        self.idx_string = QtWidgets.QLabel()
        # String to display total number of moves
        self.tot_idx_string = QtWidgets.QLabel()
        # Button to reset GUI to initial state
        self.reset_button = QtWidgets.QPushButton('Reset')
        self.reset_button.clicked.connect(self.reset_button_pressed)
        # Button to upload final state to Alyx
        self.complete_button = QtWidgets.QPushButton('Upload')
        self.complete_button.clicked.connect(self.complete_button_pressed)
        # Drop down list to choose subject
        self.subj_list = QtGui.QStandardItemModel()
        self.subj_combobox = QtWidgets.QComboBox()
        # Add line edit and completer to be able to search for subject
        self.subj_combobox.setLineEdit(QtWidgets.QLineEdit())
        subj_completer = QtWidgets.QCompleter()
        subj_completer.setCaseSensitivity(QtCore.Qt.CaseInsensitive)
        self.subj_combobox.setCompleter(subj_completer)
        self.subj_combobox.setModel(self.subj_list)
        self.subj_combobox.completer().setModel(self.subj_list)
        self.subj_combobox.textActivated.connect(self.on_subject_selected)
        # Drop down list to choose session
        self.sess_list = QtGui.QStandardItemModel()
        self.sess_combobox = QtWidgets.QComboBox()
        self.sess_combobox.setModel(self.sess_list)
        self.sess_combobox.activated.connect(self.on_session_selected)

        # Button to get data to display in GUI
        self.data_button = QtWidgets.QPushButton('Get Data')
        self.data_button.clicked.connect(self.data_button_pressed)

        # Arrange interaction features into three different layout groups
        # Group 1
        hlayout1 = QtWidgets.QHBoxLayout()
        hlayout2 = QtWidgets.QHBoxLayout()
        hlayout1.addWidget(self.fit_button, stretch=1)
        hlayout1.addWidget(self.offset_button, stretch=1)
        hlayout1.addWidget(self.tot_idx_string, stretch=2)
        hlayout2.addWidget(self.prev_button, stretch=1)
        hlayout2.addWidget(self.next_button, stretch=1)
        hlayout2.addWidget(self.idx_string, stretch=2)
        self.interaction_layout1 = QtWidgets.QVBoxLayout()
        self.interaction_layout1.addLayout(hlayout1)
        self.interaction_layout1.addLayout(hlayout2)
        # Group 2
        self.interaction_layout2 = QtWidgets.QHBoxLayout()
        self.interaction_layout2.addWidget(self.reset_button)
        self.interaction_layout2.addWidget(self.complete_button)
        # Group 3
        self.interaction_layout3 = QtWidgets.QHBoxLayout()
        self.interaction_layout3.addWidget(self.subj_combobox, stretch=1)
        self.interaction_layout3.addWidget(self.sess_combobox, stretch=2)
        self.interaction_layout3.addWidget(self.data_button, stretch=1)

    def init_figures(self):
        """
        Create all figures that will be added to the GUI
        """
        # Figures to show ephys data
        # 2D scatter/ image plot
        self.fig_img = pg.PlotItem()
        #self.fig_img.setMouseEnabled(x=False, y=False)
        self.fig_img.setYRange(min=self.probe_tip - self.probe_extra, max=self.probe_top +
                               self.probe_extra, padding=self.pad)
        self.fig_img.addLine(y=self.probe_tip, pen=self.kpen_dot, z=50)
        self.fig_img.addLine(y=self.probe_top, pen=self.kpen_dot, z=50)
        self.set_axis(self.fig_img, 'bottom')
        self.fig_data_ax = self.set_axis(self.fig_img, 'left',
                                         label='Distance from probe tip (uV)')

        self.fig_img_cb = pg.PlotItem()
        self.fig_img_cb.setMaximumHeight(70)
        self.fig_img_cb.setMouseEnabled(x=False, y=False)
        self.set_axis(self.fig_img_cb, 'bottom', show=False)
        self.set_axis(self.fig_img_cb, 'left', pen='w')
        self.set_axis(self.fig_img_cb, 'top', pen='w')

        # 1D line plot
        self.fig_line = pg.PlotItem()
        self.fig_line.setMouseEnabled(x=False, y=False)
        self.fig_line.setYRange(min=self.probe_tip - self.probe_extra, max=self.probe_top +
                                self.probe_extra, padding=self.pad)
        self.fig_line.addLine(y=self.probe_tip, pen=self.kpen_dot, z=50)
        self.fig_line.addLine(y=self.probe_top, pen=self.kpen_dot, z=50)
        self.set_axis(self.fig_line, 'bottom')
        self.set_axis(self.fig_line, 'left', show=False)

        # 2D probe plot
        self.fig_probe = pg.PlotItem()
        self.fig_probe.setMouseEnabled(x=False, y=False)
        self.fig_probe.setMaximumWidth(50)
        self.fig_probe.setYRange(min=self.probe_tip - self.probe_extra, max=self.probe_top +
                                 self.probe_extra, padding=self.pad)
        self.fig_probe.addLine(y=self.probe_tip, pen=self.kpen_dot, z=50)
        self.fig_probe.addLine(y=self.probe_top, pen=self.kpen_dot, z=50)
        self.set_axis(self.fig_probe, 'bottom', pen='w')
        self.set_axis(self.fig_probe, 'left', show=False)

        self.fig_probe_cb = pg.PlotItem()
        self.fig_probe_cb.setMouseEnabled(x=False, y=False)
        self.fig_probe_cb.setMaximumHeight(70)
        self.set_axis(self.fig_probe_cb, 'bottom', show=False)
        self.set_axis(self.fig_probe_cb, 'left', show=False)
        self.set_axis(self.fig_probe_cb, 'top', pen='w')

        # Add img plot, line plot, probe plot, img colourbar and probe colourbar to a graphics
        # layout widget so plots can be arranged and moved easily
        self.fig_data_area = pg.GraphicsLayoutWidget()
        self.fig_data_area.scene().sigMouseClicked.connect(self.on_mouse_double_clicked)
        self.fig_data_area.scene().sigMouseHover.connect(self.on_mouse_hover)
        self.fig_data_layout = pg.GraphicsLayout()

        self.fig_data_layout.addItem(self.fig_img_cb, 0, 0)
        self.fig_data_layout.addItem(self.fig_probe_cb, 0, 1, 1, 2)
        self.fig_data_layout.addItem(self.fig_img, 1, 0)
        self.fig_data_layout.addItem(self.fig_line, 1, 1)
        self.fig_data_layout.addItem(self.fig_probe, 1, 2)
        self.fig_data_layout.layout.setColumnStretchFactor(0, 6)
        self.fig_data_layout.layout.setColumnStretchFactor(1, 2)
        self.fig_data_layout.layout.setColumnStretchFactor(2, 1)
        self.fig_data_layout.layout.setRowStretchFactor(0, 1)
        self.fig_data_layout.layout.setRowStretchFactor(1, 10)

        self.fig_data_area.addItem(self.fig_data_layout)

        # Figures to show histology data
        # Histology figure that will be updated with user input
        self.fig_hist = pg.PlotItem()
        self.fig_hist.setContentsMargins(0, 0, 0, 0)
        self.fig_hist.setMouseEnabled(x=False)
        self.fig_hist.setYRange(min=self.probe_tip - self.probe_extra, max=self.probe_top +
                                self.probe_extra, padding=self.pad)
        self.set_axis(self.fig_hist, 'bottom', pen='w', ticks=False)
        self.ax_hist = self.set_axis(self.fig_hist, 'left', pen=None)
        self.ax_hist.setWidth(0)
        self.ax_hist.setStyle(tickTextOffset=-70)

        self.fig_scale = pg.PlotItem()
        self.fig_scale.setMaximumWidth(50)
        self.fig_scale.setMouseEnabled(x=False)
        self.scale_label = pg.LabelItem(color='k')
        self.set_axis(self.fig_scale, 'bottom', pen='w', ticks=False)
        self.set_axis(self.fig_scale, 'left', show=False)
        (self.fig_scale).setYLink(self.fig_hist)

        # Figure that will show scale factor of histology boundaries
        self.fig_scale_cb = pg.PlotItem()
        self.fig_scale_cb.setMouseEnabled(x=False, y=False)
        self.fig_scale_cb.setMaximumHeight(70)
        self.set_axis(self.fig_scale_cb, 'bottom', show=False)
        self.set_axis(self.fig_scale_cb, 'left', show=False)
        self.fig_scale_ax = self.set_axis(self.fig_scale_cb, 'top', pen='w')
        self.set_axis(self.fig_scale_cb, 'right', show=False)

        # Histology figure that will remain at initial state for reference
        self.fig_hist_ref = pg.PlotItem()
        self.fig_hist_ref.setMouseEnabled(x=False)
        self.fig_hist_ref.setYRange(min=self.probe_tip - self.probe_extra, max=self.probe_top +
                                    self.probe_extra, padding=self.pad)
        self.set_axis(self.fig_hist_ref, 'bottom', pen='w', ticks=False)
        self.set_axis(self.fig_hist_ref, 'left', show=False)
        self.ax_hist_ref = self.set_axis(self.fig_hist_ref, 'right', pen=None)
        self.ax_hist_ref.setWidth(0)
        self.ax_hist_ref.setStyle(tickTextOffset=-70)

        self.fig_hist_area = pg.GraphicsLayoutWidget()
        self.fig_hist_area.setMouseTracking(True)
        self.fig_hist_area.scene().sigMouseClicked.connect(self.on_mouse_double_clicked)
        self.fig_hist_area.scene().sigMouseHover.connect(self.on_mouse_hover)

        self.fig_hist_layout = pg.GraphicsLayout()
        self.fig_hist_layout.addItem(self.fig_scale_cb, 0, 0, 1, 3)
        self.fig_hist_layout.addItem(self.fig_hist, 1, 0)
        self.fig_hist_layout.addItem(self.fig_scale, 1, 1)
        self.fig_hist_layout.addItem(self.fig_hist_ref, 1, 2,)
        self.fig_hist_layout.layout.setColumnStretchFactor(0, 4)
        self.fig_hist_layout.layout.setColumnStretchFactor(1, 1)
        self.fig_hist_layout.layout.setColumnStretchFactor(2, 4)
        self.fig_hist_layout.layout.setRowStretchFactor(0, 1)
        self.fig_hist_layout.layout.setRowStretchFactor(1, 10)
        self.fig_hist_area.addItem(self.fig_hist_layout)

        # Figure to show probe location through coronal slice of brain
        self.fig_slice = matplot.MatplotlibWidget()
        fig = self.fig_slice.getFigure()
        fig.canvas.toolbar.hide()
        self.fig_slice_ax = fig.gca()
        self.fig_slice_ax.axis('off')

        # Figure to show fit and offset applied by user
        self.fig_fit = pg.PlotWidget(background='w')
        self.fig_fit.setMouseEnabled(x=False, y=False)
        self.fig_fit_exporter = pg.exporters.ImageExporter(self.fig_fit.plotItem)
        self.fig_fit.sigDeviceRangeChanged.connect(self.on_fig_size_changed)
        self.fig_fit.setXRange(min=self.view_total[0], max=self.view_total[1])
        self.fig_fit.setYRange(min=self.view_total[0], max=self.view_total[1])
        self.set_axis(self.fig_fit, 'bottom', label='Original coordinates (um)')
        self.set_axis(self.fig_fit, 'left', label='New coordinates (um)')
        plot = pg.PlotCurveItem()
        plot.setData(x=self.depth, y=self.depth, pen=self.kpen_dot)
        self.fit_plot = pg.PlotCurveItem(pen=self.bpen_solid)
        self.fit_scatter = pg.ScatterPlotItem(size=7, symbol='o', brush='w', pen='b')
        self.fit_plot_lin = pg.PlotCurveItem(pen=self.rpen_dot)
        self.fig_fit.addItem(plot)
        self.fig_fit.addItem(self.fit_plot)
        self.fig_fit.addItem(self.fit_plot_lin)
        self.fig_fit.addItem(self.fit_scatter)

        self.lin_fit_option = QtGui.QCheckBox('Linear fit', self.fig_fit)
        self.lin_fit_option.setChecked(True)
        self.lin_fit_option.stateChanged.connect(self.lin_fit_option_changed)
        self.on_fig_size_changed()

    def on_fig_size_changed(self):
        fig_width = self.fig_fit_exporter.getTargetRect().width()
        fig_height = self.fig_fit_exporter.getTargetRect().width()
        self.lin_fit_option.move(fig_width - 70, fig_height - 60)


class ClustPopupWindow(QtGui.QMainWindow):
    closed = QtCore.pyqtSignal(QtGui.QMainWindow)
    moved = QtCore.pyqtSignal()

    def __init__(self, title, parent=None):
        super(ClustPopupWindow, self).__init__()
        self.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)
        self.resize(300, 300)
        self.move(randrange(30) + 1000, randrange(30) + 200)
        self.clust_widget = pg.GraphicsLayoutWidget()
        self.setCentralWidget(self.clust_widget)
        self.setWindowTitle(title)
        self.show()

    def closeEvent(self, event):
        self.closed.emit(self)
        self.close()

    def leaveEvent(self, event):
        self.moved.emit()