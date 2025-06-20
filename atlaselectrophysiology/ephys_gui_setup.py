from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
import pyqtgraph.exporters
import numpy as np
from random import randrange
from atlaselectrophysiology.AdaptedAxisItem import replace_axis
from ibllib.qc.critical_reasons import CriticalInsertionNote


pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')


class Setup():
    def init_layout(self, main_window, offline=False):
        self.resize(1600, 800)
        self.setWindowTitle('Electrophysiology Atlas')
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.offline = offline
        main_widget = QtWidgets.QWidget()
        self.setCentralWidget(main_widget)

        self.init_interaction_features()
        self.init_figures()

        main_layout = QtWidgets.QGridLayout()
        main_layout.addWidget(self.fig_data_area, 0, 0, 10, 1)
        main_layout.addWidget(self.fig_hist_area, 0, 1, 10, 1)
        main_layout.addLayout(self.interaction_layout3, 0, 2, 1, 1)
        main_layout.addWidget(self.fig_slice_area, 1, 2, 3, 1)
        main_layout.addLayout(self.interaction_layout1, 4, 2, 1, 1)
        main_layout.addWidget(self.fig_fit, 5, 2, 3, 1)
        main_layout.addLayout(self.interaction_layout2, 8, 2, 2, 1)
        main_layout.setColumnStretch(0, 5)
        main_layout.setColumnStretch(1, 2)
        main_layout.setColumnStretch(2, 3)

        main_widget.setLayout(main_layout)

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
        menu_bar = QtWidgets.QMenuBar(self)
        menu_bar.setNativeMenuBar(False)
        self.setMenuBar(menu_bar)

        # IMAGE PLOTS MENU BAR
        # Define all 2D scatter/ image plot options
        #TOOOODOOOO make it a dict like the other ones so we can iterate through them without all this hardcoded nonsense

        # Add menu bar for 2D scatter/ image plot options
        img_options = menu_bar.addMenu('Image Plots')
        # Add action group so we can toggle through 2D scatter/ image plot options
        self.img_options_group = QtWidgets.QActionGroup(img_options)
        # Only allow one to plot to be selected at any one time
        self.img_options_group.setExclusive(True)

        for i, img_plot in enumerate(self.shank.img_plots):
            checked = True if i==0 else False
            img = QtWidgets.QAction(img_plot, self, checkable=True, checked=checked)
            img.triggered.connect(lambda checked, item=img_plot: self.plot_image(
                                  self.shank.img_plots[item]))
            img_options.addAction(img)
            self.img_options_group.addAction(img)
            if i == 0:
                self.img_init = img

        for scat_plot in self.shank.scatter_plots:
            img = QtWidgets.QAction(scat_plot, self, checkable=True, checked=False)
            img.triggered.connect(lambda checked, item=scat_plot: self.plot_scatter(
                                  self.shank.scatter_plots[item]))
            img_options.addAction(img)
            self.img_options_group.addAction(img)

        # If unity add a callback from the image options group to update the unity plots
        if self.unity:
            self.img_options_group.triggered.connect(lambda: self.plot_unity('image'))

        # LINE PLOTS MENU BAR
        # Define all 1D line plot options
        line_options = menu_bar.addMenu('Line Plots')
        # Add action group so we can toggle through 2D scatter/ image plot options
        self.line_options_group = QtWidgets.QActionGroup(line_options)
        # Only allow one to plot to be selected at any one time
        self.line_options_group.setExclusive(True)

        for i, line_plot in enumerate(self.shank.line_plots):
            checked = True if i == 0 else False
            img = QtWidgets.QAction(line_plot, self, checkable=True, checked=checked)
            img.triggered.connect(lambda checked, item=line_plot: self.plot_line(
                                  self.shank.line_plots[item]))
            line_options.addAction(img)
            self.line_options_group.addAction(img)
            if i == 0:
                self.line_init = img

        # PROBE PLOTS MENU BAR
        # Define all 2D probe plot options
        probe_options = menu_bar.addMenu("Probe Plots")
        # Add action group so we can toggle through probe plot options
        self.probe_options_group = QtWidgets.QActionGroup(probe_options)
        self.probe_options_group.setExclusive(True)

        for i, probe_plot in enumerate(self.shank.probe_plots):
            checked = True if i == 0 else False
            img = QtWidgets.QAction(probe_plot, self, checkable=True, checked=checked)
            img.triggered.connect(lambda checked, item=probe_plot: self.plot_probe(
                self.shank.probe_plots[item]))
            probe_options.addAction(img)
            self.probe_options_group.addAction(img)
            if i == 0:
                self.probe_init = img

        # If unity add a callback from the probe options group to update the unity plots
        if self.unity:
            self.probe_options_group.triggered.connect(lambda: self.plot_unity('probe'))

        # SLICE PLOTS MENU BAR
        # Define all coronal slice plot options
        slice_hist_rd = QtWidgets.QAction('Histology Red', self, checkable=True, checked=True)
        slice_hist_rd.triggered.connect(lambda: self.plot_slice(self.shank.slice_data, 'hist_rd'))
        slice_hist_gr = QtWidgets.QAction('Histology Green', self, checkable=True, checked=False)
        slice_hist_gr.triggered.connect(lambda: self.plot_slice(self.shank.slice_data, 'hist_gr'))
        slice_ccf = QtWidgets.QAction('CCF', self, checkable=True, checked=False)
        slice_ccf.triggered.connect(lambda: self.plot_slice(self.shank.slice_data, 'ccf'))
        slice_label = QtWidgets.QAction('Annotation', self, checkable=True, checked=False)
        slice_label.triggered.connect(lambda: self.plot_slice(self.shank.slice_data, 'label'))

        if self.shank.fp_slice_data is not None:
            fp_slice_label = QtWidgets.QAction('Annotation FP', self, checkable=True, checked=False)
            fp_slice_label.triggered.connect(lambda: self.plot_slice(self.shank.fp_slice_data, 'label'))

        if not self.offline:
            slice_hist_cb = QtWidgets.QAction('Histology cerebellar example', self, checkable=True, checked=False)
            slice_hist_cb.triggered.connect(lambda: self.plot_slice(self.shank.slice_data, 'hist_cb'))
        # Initialise with raw histology image
        self.slice_init = slice_hist_rd

        # Add menu bar for slice plot

        slice_options = menu_bar.addMenu("Slice Plots")
        # Add action group so we can toggle through slice plot options
        self.slice_options_group = QtWidgets.QActionGroup(slice_options)
        self.slice_options_group.setExclusive(True)
        slice_options.addAction(slice_hist_rd)
        self.slice_options_group.addAction(slice_hist_rd)
        slice_options.addAction(slice_hist_gr)
        self.slice_options_group.addAction(slice_hist_gr)
        slice_options.addAction(slice_ccf)
        self.slice_options_group.addAction(slice_ccf)
        slice_options.addAction(slice_label)
        self.slice_options_group.addAction(slice_label)
        if self.shank.fp_slice_data is not None:
            slice_options.addAction(fp_slice_label)
            self.slice_options_group.addAction(fp_slice_label)

        if not self.offline:
            slice_options.addAction(slice_hist_cb)
            self.slice_options_group.addAction(slice_hist_cb)

        # FILTER UNITS MENU BAR
        # Define unit filtering options
        unit_filter_options = menu_bar.addMenu("Filter Units")
        # Add action group so we can toggle through unit options
        unit_filter_options_group = QtWidgets.QActionGroup(unit_filter_options)
        unit_filter_options_group.setExclusive(True)

        units = ['All', 'KS good', 'KS mua', 'IBL good']
        for i, unit in enumerate(units):
            checked = True if i == 0 else False
            all_units = QtWidgets.QAction(unit, self, checkable=True, checked=checked)
            all_units.triggered.connect(lambda: self.filter_unit_pressed(unit))
            unit_filter_options.addAction(all_units)
            unit_filter_options_group.addAction(all_units)
            if i == 0:
                self.unit_init = all_units


        # FIT OPTIONS MENU BAR
        fit_options = menu_bar.addMenu("Fit Options")
        # DISPLAY OPTIONS MENU BAR
        display_options = menu_bar.addMenu('Display Options')

        upload_callback = self.complete_button_pressed_offline if self.offline else self.display_qc_options
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
                {'shortcut': 'Shift+U', 'callback': upload_callback, 'menu': fit_options},
            'Toggle Image Plots': # Shortcuts to toggle between plots options
                {'shortcut': 'Alt+1', 'callback': lambda: self.toggle_plots(self.img_options_group), 'menu': display_options},
            'Toggle Line Plots':
                {'shortcut': 'Alt+2', 'callback': lambda: self.toggle_plots(self.line_options_group), 'menu': display_options},
            'Toggle Probe Plots':
                {'shortcut': 'Alt+3', 'callback': lambda: self.toggle_plots(self.probe_options_group), 'menu': display_options},
            'Toggle Slice Plots':
                {'shortcut': 'Alt+4', 'callback': lambda: self.toggle_plots(self.slice_options_group), 'menu': display_options},
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
            'Minimise/Show Cluster Popup': # Shortcuts for cluster popup window
                {'shortcut': 'Alt+M', 'callback': self.minimise_popups, 'menu': display_options},
            'Close Cluster Popup':
                {'shortcut': 'Alt+X', 'callback': self.close_popups, 'menu': display_options},
            'Save Plots': # Option to save all plots
                {'callback': self.save_plots, 'menu': display_options},

        }

        for key, val in keyboard.items():
            option = QtWidgets.QAction(key, self)
            shortcut = val.get('shortcut', None)
            if shortcut:
                option.setShortcut(shortcut)
            option.triggered.connect(val['callback'])
            val['menu'].addAction(option)

        # SESSION INFORMATION MENU BAR
        # Define all session information options
        # Display any notes associated with recording session
        session_notes = QtWidgets.QAction('Session Notes', self)
        session_notes.triggered.connect(self.display_session_notes)
        # Shortcut to show label information
        region_info = QtWidgets.QAction('Region Info', self)
        region_info.setShortcut('Shift+I')
        region_info.triggered.connect(lambda: region_lookup_callback(self))

        # Add menu bar with all possible session info options
        info_options = menu_bar.addMenu('Session Information')
        info_options.addAction(session_notes)
        info_options.addAction(region_info)

        # Display other sessions that are closeby if online mode
        if not self.offline:
            nearby_info = QtWidgets.QAction('Nearby Sessions', self)
            nearby_info.triggered.connect(self.display_nearby_sessions)
            info_options.addAction(nearby_info)

            scaling_info = QtWidgets.QAction('Subject Scaling', self)
            scaling_info.triggered.connect(self.display_subject_scaling)
            info_options.addAction(scaling_info)

            feature_info = QtWidgets.QAction('Region Feature', self)
            feature_info.triggered.connect(self.display_region_features)
            info_options.addAction(feature_info)

        # UNITY MENU BAR
        if self.unity:
            unity_regions = QtWidgets.QAction('Show Regions', self, checkable=True, checked=True)
            unity_regions.triggered.connect(self.toggle_unity_regions)

            unity_options = menu_bar.addMenu('Urchin')
            unity_options.addAction(unity_regions)


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
        # Button to upload final state to Alyx/ to local file
        self.complete_button = QtWidgets.QPushButton('Upload')
        if not self.offline:
            self.complete_button.clicked.connect(self.display_qc_options)
        else:
            self.complete_button.clicked.connect(self.complete_button_pressed_offline)

        if not self.offline:
            # If offline mode is False, read in Subject and Session options from Alyx
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
            self.subj_combobox.activated.connect(self.on_subject_selected)

            # Drop down list to choose session
            self.sess_list = QtGui.QStandardItemModel()
            self.sess_combobox = QtWidgets.QComboBox()
            self.sess_combobox.setModel(self.sess_list)
            self.sess_combobox.activated.connect(self.on_session_selected)
        else:
            # If offline mode is True, provide dialog to select local folder that holds data
            self.folder_line = QtWidgets.QLineEdit()
            self.folder_button = QtWidgets.QToolButton()
            self.folder_button.setText('...')
            self.folder_button.clicked.connect(self.on_folder_selected)

        # Drop down list to choose previous alignments
        self.align_list = QtGui.QStandardItemModel()
        self.align_combobox = QtWidgets.QComboBox()
        self.align_combobox.setModel(self.align_list)
        self.align_combobox.activated.connect(self.on_alignment_selected)

        # Drop down list to select shank
        self.shank_list = QtGui.QStandardItemModel()
        self.shank_combobox = QtWidgets.QComboBox()
        self.shank_combobox.setModel(self.shank_list)
        self.shank_combobox.activated.connect(self.on_shank_selected)

        # Plus button to select alignment file
        self.align_extra = QtWidgets.QToolButton()
        self.align_extra.setText('+')
        self.align_extra.clicked.connect(self.add_alignment_pressed)

        # Button to get data to display in GUI
        self.data_button = QtWidgets.QPushButton('Load')
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

        if self.unity:
            glayout3 = QtWidgets.QGridLayout()
            glayout3.setVerticalSpacing(0)
            self.min_label = QtWidgets.QLabel("0.1")
            self.max_label = QtWidgets.QLabel("1")
            self.max_label.setAlignment(QtCore.Qt.AlignRight)
            self.unity_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            self.unity_slider.setMinimum(1)
            self.unity_slider.setMaximum(10)
            self.unity_slider.setValue(5)
            self.unity_slider.setTickPosition(QtWidgets.QSlider.TicksAbove)
            self.unity_slider.setTickInterval(1)
            self.unity_slider.sliderReleased.connect(self.on_point_size_changed)

            glayout3.addWidget(self.unity_slider, 0, 0, 1, 10)
            glayout3.addWidget(self.min_label, 1, 0, 1, 1)
            glayout3.addWidget(self.max_label, 1, 9, 1, 1)
            self.interaction_layout1.addLayout(glayout3)

        # Group 2
        self.interaction_layout2 = QtWidgets.QHBoxLayout()
        self.interaction_layout2.addWidget(self.reset_button)
        self.interaction_layout2.addWidget(self.complete_button)

        # Group 3 will depend on online/ offline mode
        self.interaction_layout3 = QtWidgets.QHBoxLayout()
        if not self.offline:
            # TODO Doesn't seem to work the stretch here??
            self.interaction_layout3.addWidget(self.subj_combobox, stretch=2)
            self.interaction_layout3.addWidget(self.sess_combobox, stretch=3)
            self.interaction_layout3.addWidget(self.shank_combobox, stretch=1)
            self.interaction_layout3.addWidget(self.align_combobox, stretch=2)
            # self.interaction_layout3.addWidget(self.align_extra, stretch=1)
            self.interaction_layout3.addWidget(self.data_button, stretch=1)
        else:
            self.interaction_layout3.addWidget(self.folder_line, stretch=2)
            self.interaction_layout3.addWidget(self.folder_button, stretch=1)
            self.interaction_layout3.addWidget(self.shank_combobox, stretch=1)
            self.interaction_layout3.addWidget(self.align_combobox, stretch=2)
            self.interaction_layout3.addWidget(self.align_extra, stretch=1)
            self.interaction_layout3.addWidget(self.data_button, stretch=1)


        # Pop up dialog for qc results to datajoint, only for online mode
        if not self.offline:
            align_qc_label = QtWidgets.QLabel("Confidence of alignment")
            self.align_qc = QtWidgets.QComboBox()
            self.align_qc.addItems(["High", "Medium", "Low"])
            ephys_qc_label = QtWidgets.QLabel("QC for ephys recording")
            self.ephys_qc = QtWidgets.QComboBox()
            self.ephys_qc.addItems(["Pass", "Warning", "Critical"])

            self.desc_buttons = QtWidgets.QButtonGroup()
            self.desc_group = QtWidgets.QGroupBox("Describe problem with recording")
            self.desc_layout = QtWidgets.QVBoxLayout()
            self.desc_layout.setSpacing(5)
            self.desc_buttons.setExclusive(False)
            options = CriticalInsertionNote.descriptions_gui
            for i, val in enumerate(options):

                button = QtWidgets.QCheckBox(val)
                button.setCheckState(QtCore.Qt.Unchecked)

                self.desc_buttons.addButton(button, id=i)
                self.desc_layout.addWidget(button)

            self.desc_group.setLayout(self.desc_layout)

            self.qc_dialog = QtWidgets.QDialog(self)
            self.qc_dialog.setWindowTitle('QC assessment')
            self.qc_dialog.resize(300, 150)
            self.qc_dialog.accepted.connect(self.qc_button_clicked)
            buttonBox = QtWidgets.QDialogButtonBox(
                QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
            buttonBox.accepted.connect(self.qc_dialog.accept)
            buttonBox.rejected.connect(self.qc_dialog.reject)
            #
            dialog_layout = QtWidgets.QVBoxLayout()
            dialog_layout.addWidget(align_qc_label)
            dialog_layout.addWidget(self.align_qc)
            dialog_layout.addWidget(ephys_qc_label)
            dialog_layout.addWidget(self.ephys_qc)
            dialog_layout.addWidget(self.desc_group)
            dialog_layout.addWidget(buttonBox)
            self.qc_dialog.setLayout(dialog_layout)


    def init_figures(self):
        """
        Create all figures that will be added to the GUI
        """
        # Lists to store the position of probe top and tip
        self.probe_top_lines = []
        self.probe_tip_lines = []

        # Figures to show ephys data
        # 2D scatter/ image plot
        self.fig_img = pg.PlotItem()
        self.fig_img.setYRange(min=self.probe_tip - self.probe_extra, max=self.probe_top +
                               self.probe_extra, padding=self.pad)
        self.probe_tip_lines.append(self.fig_img.addLine(y=self.probe_tip, pen=self.kpen_dot,
                                                         z=50))
        self.probe_top_lines.append(self.fig_img.addLine(y=self.probe_top, pen=self.kpen_dot,
                                                         z=50))
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
        self.probe_tip_lines.append(self.fig_line.addLine(y=self.probe_tip, pen=self.kpen_dot,
                                                          z=50))
        self.probe_top_lines.append(self.fig_line.addLine(y=self.probe_top, pen=self.kpen_dot,
                                                          z=50))
        self.set_axis(self.fig_line, 'bottom')
        self.set_axis(self.fig_line, 'left', show=False)

        # 2D probe plot
        self.fig_probe = pg.PlotItem()
        self.fig_probe.setMouseEnabled(x=False, y=False)
        self.fig_probe.setMaximumWidth(50)
        self.fig_probe.setYRange(min=self.probe_tip - self.probe_extra, max=self.probe_top +
                                 self.probe_extra, padding=self.pad)
        self.probe_tip_lines.append(self.fig_probe.addLine(y=self.probe_tip, pen=self.kpen_dot,
                                                           z=50))
        self.probe_top_lines.append(self.fig_probe.addLine(y=self.probe_top, pen=self.kpen_dot,
                                                           z=50))
        self.set_axis(self.fig_probe, 'bottom', pen='w')
        self.set_axis(self.fig_probe, 'left', show=False)

        self.fig_probe_cb = pg.PlotItem()
        self.fig_probe_cb.setMouseEnabled(x=False, y=False)
        self.fig_probe_cb.setMaximumHeight(70)
        self.set_axis(self.fig_probe_cb, 'bottom', show=False)
        self.set_axis(self.fig_probe_cb, 'left', pen='w')
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
        self.set_axis(self.fig_hist, 'bottom', pen='w')

        # This is the solution from pyqtgraph people, but doesn't show ticks
        # self.fig_hist.showGrid(False, True, 0)

        replace_axis(self.fig_hist)
        self.ax_hist = self.set_axis(self.fig_hist, 'left', pen=None)
        self.ax_hist.setWidth(0)
        self.ax_hist.setStyle(tickTextOffset=-70)

        self.fig_scale = pg.PlotItem()
        self.fig_scale.setMaximumWidth(50)
        self.fig_scale.setMouseEnabled(x=False)
        self.scale_label = pg.LabelItem(color='k')
        self.set_axis(self.fig_scale, 'bottom', pen='w')
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
        self.set_axis(self.fig_hist_ref, 'bottom', pen='w')
        self.set_axis(self.fig_hist_ref, 'left', show=False)
        replace_axis(self.fig_hist_ref, orientation='right', pos=(2, 2))
        self.ax_hist_ref = self.set_axis(self.fig_hist_ref, 'right', pen=None)
        self.ax_hist_ref.setWidth(0)
        self.ax_hist_ref.setStyle(tickTextOffset=-70)

        self.fig_hist_area = pg.GraphicsLayoutWidget()
        self.fig_hist_area.setMouseTracking(True)
        self.fig_hist_area.scene().sigMouseClicked.connect(self.on_mouse_double_clicked)
        self.fig_hist_area.scene().sigMouseHover.connect(self.on_mouse_hover)

        self.fig_hist_extra_yaxis = pg.PlotItem()
        self.fig_hist_extra_yaxis.setMouseEnabled(x=False, y=False)
        self.fig_hist_extra_yaxis.setMaximumWidth(2)
        self.fig_hist_extra_yaxis.setYRange(min=self.probe_tip - self.probe_extra,
                                            max=self.probe_top + self.probe_extra,
                                            padding=self.pad)

        self.set_axis(self.fig_hist_extra_yaxis, 'bottom', pen='w')
        self.ax_hist2 = self.set_axis(self.fig_hist_extra_yaxis, 'left', pen=None)
        self.ax_hist2.setWidth(10)

        self.fig_hist_layout = pg.GraphicsLayout()
        self.fig_hist_layout.addItem(self.fig_scale_cb, 0, 0, 1, 4)
        self.fig_hist_layout.addItem(self.fig_hist_extra_yaxis, 1, 0)
        self.fig_hist_layout.addItem(self.fig_hist, 1, 1)
        self.fig_hist_layout.addItem(self.fig_scale, 1, 2)
        self.fig_hist_layout.addItem(self.fig_hist_ref, 1, 3)
        self.fig_hist_layout.layout.setColumnStretchFactor(0, 1)
        self.fig_hist_layout.layout.setColumnStretchFactor(1, 4)
        self.fig_hist_layout.layout.setColumnStretchFactor(2, 1)
        self.fig_hist_layout.layout.setColumnStretchFactor(3, 4)
        self.fig_hist_layout.layout.setRowStretchFactor(0, 1)
        self.fig_hist_layout.layout.setRowStretchFactor(1, 10)
        self.fig_hist_area.addItem(self.fig_hist_layout)

        # Figure to show coronal slice through the brain
        self.fig_slice_area = pg.GraphicsLayoutWidget()
        self.fig_slice_layout = pg.GraphicsLayout()
        self.fig_slice_hist_alt = pg.ViewBox()
        self.fig_slice = pg.ViewBox()
        self.fig_slice_layout.addItem(self.fig_slice, 0, 0)
        self.fig_slice_layout.addItem(self.fig_slice_hist_alt, 0, 1)
        self.fig_slice_layout.layout.setColumnStretchFactor(0, 3)
        self.fig_slice_layout.layout.setColumnStretchFactor(1, 1)
        self.fig_slice_area.addItem(self.fig_slice_layout)
        self.slice_item = self.fig_slice_hist_alt

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

        self.lin_fit_option = QtWidgets.QCheckBox('Linear fit', self.fig_fit)
        self.lin_fit_option.setChecked(True)
        self.lin_fit_option.stateChanged.connect(self.lin_fit_option_changed)
        self.on_fig_size_changed()

    def on_fig_size_changed(self):
        # fig_width = self.fig_fit_exporter.getTargetRect().width()
        # fig_height = self.fig_fit_exporter.getTargetRect().width()
        self.lin_fit_option.move(70, 10)



class CheckableComboBox(QtWidgets.QComboBox):
    def __init__(self):
        super(CheckableComboBox, self).__init__()
        self.view().pressed.connect(self.handleItemPressed)
        self.setModel(QtGui.QStandardItemModel(self))

    def handleItemPressed(self, index):
        item = self.model().itemFromIndex(index)
        if item.checkState() == QtCore.Qt.Checked:
            item.setCheckState(QtCore.Qt.Unchecked)
        else:
            item.setCheckState(QtCore.Qt.Checked)


class PopupWindow(QtWidgets.QMainWindow):

    closed = QtCore.pyqtSignal(QtWidgets.QMainWindow)
    moved = QtCore.pyqtSignal()

    def __init__(self, title, parent=None, size=(300, 300), graphics=True):
        super(PopupWindow, self).__init__()
        self.parent = parent
        self.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint | QtCore.Qt.Tool)
        self.resize(size[0], size[1])
        self.move(randrange(30) + 1000, randrange(30) + 200)
        if graphics:
            self.popup_widget = pg.GraphicsLayoutWidget()
        else:
            self.popup_widget = QtWidgets.QWidget()
            self.layout = QtWidgets.QGridLayout()
            self.popup_widget.setLayout(self.layout)
        self.setCentralWidget(self.popup_widget)
        self.setWindowTitle(title)
        self.show()

    def closeEvent(self, event):
        self.closed.emit(self)
        self.close()

    def leaveEvent(self, event):
        self.moved.emit()




def region_lookup_callback(parent):

    if parent.selected_region:
        idx = np.where(parent.hist_regions['left'] == parent.selected_region)[0]
        if not np.any(idx):
            idx = np.where(parent.hist_regions['right'] == parent.selected_region)[0]
        if not np.any(idx):
            idx = np.array([0])

    print('here')
    parent.label_win = RegionLookup._get_or_create('Stucture Info', allen=parent.allen, parent=parent)



class RegionLookup(PopupWindow):

    @staticmethod
    def _instances():
        app = QtWidgets.QApplication.instance()
        return [w for w in app.topLevelWidgets() if isinstance(w, RegionLookup)]

    @staticmethod
    def _get_or_create(title, **kwargs):
        av = next(filter(lambda e: e.isVisible() and e.windowTitle() == title,
                         RegionLookup._instances()), None)
        if av is None:
            av = RegionLookup(title, **kwargs)
        else:
            av.showNormal()
            av.activateWindow()
        return av

    def __init__(self, title, allen=None, parent=None):
        super().__init__(title, parent=parent, size=(500, 700), graphics=False)

        self.struct_list = QtGui.QStandardItemModel()
        self.struct_view = QtWidgets.QTreeView()
        self.struct_view.setModel(self.struct_list)
        self.struct_view.clicked.connect(self.label_pressed)

        allen = allen.drop([0]).reset_index(drop=True)

        # Find the parent path of each structure by removing the structure id from path
        def parent_path(struct_path):
            return struct_path.rsplit('/', 2)[0] + '/'

        allen['parent_path'] = allen['structure_id_path'].apply(parent_path)

        self.struct_view.setHeaderHidden(True)
        unique_levels = np.unique(allen['depth']).astype(int)
        parent_info = {}
        idx = np.where(allen['depth'] == unique_levels[0])[0]
        item = QtGui.QStandardItem(allen['acronym'][idx[0]] + ': ' + allen['name'][idx[0]])
        icon = QtGui.QPixmap(20, 20)
        icon.fill(QtGui.QColor('#' + allen['color_hex_triplet'][idx[0]]))
        item.setIcon(QtGui.QIcon(icon))
        item.setAccessibleText(str(allen['id'][idx[0]]))
        item.setEditable(False)
        self.struct_list.appendRow(item)
        parent_info.update({allen['structure_id_path'][idx[0]]: item})

        for level in unique_levels[1:]:
            idx_levels = np.where(allen['depth'] == level)[0]
            for idx in idx_levels:
                parent = allen['parent_path'][idx]
                parent_item = parent_info[parent]
                item = QtGui.QStandardItem(allen['acronym'][idx] + ': ' + allen['name'][idx])
                icon.fill(QtGui.QColor('#' + allen['color_hex_triplet'][idx]))
                item.setIcon(QtGui.QIcon(icon))
                item.setAccessibleText(str(allen['id'][idx]))
                item.setEditable(False)
                parent_item.appendRow(item)
                parent_info.update({allen['structure_id_path'][idx]: item})

        self.struct_description = QtWidgets.QTextEdit()

        self.layout.addWidget(self.struct_view)
        self.layout.addWidget(self.struct_description)
        self.layout.setRowStretch(0, 7)
        self.layout.setRowStretch(1, 3)


    def label_pressed(self, item):
        idx = int(item.model().itemFromIndex(item).accessibleText())
        description, lookup = self.loaddata.get_region_description(idx)
        item = self.struct_list.findItems(lookup, flags=QtCore.Qt.MatchRecursive)
        model_item = self.struct_list.indexFromItem(item[0])
        self.struct_view.setCurrentIndex(model_item)
        self.struct_description.setText(description)

    def label_selected(self, region):

        description, lookup = self.loaddata.get_region_description(
            self.shank.align.ephysalign.region_id[region[0]][0])
        item = self.struct_list.findItems(lookup, flags=QtCore.Qt.MatchRecursive)
        model_item = self.struct_list.indexFromItem(item[0])
        self.struct_view.collapseAll()
        self.struct_view.scrollTo(model_item)
        self.struct_view.setCurrentIndex(model_item)
        self.struct_description.setText(description)

