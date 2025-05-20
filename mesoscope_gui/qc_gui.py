from qtpy.QtWidgets import (QWidget, QApplication, QSizePolicy, QMainWindow, QVBoxLayout, QSplitter, QSlider,
                            QComboBox, QPushButton, QGraphicsProxyWidget, QGroupBox, QLabel, QFileDialog)
from qtpy.QtCore import Qt, QSettings, QDir
from qtpy.QtGui import QTransform, QStandardItemModel, QStandardItem
import pyqtgraph as pg

import matplotlib
import numpy as np
from pathlib import Path
import pandas as pd
import re
import sparse

from iblutil.util import Bunch
from one.alf.path import get_session_path, folder_parts, ALFPath


class MesoscopeLoader(object):

    def __init__(self, data_path):

        self.session_path = ALFPath(get_session_path(data_path))
        assert self.session_path.is_session_path(), f'Incorrect session path detected {self.session_path}'


        full_path = folder_parts(data_path, as_dict=True)
        assert 'alf/FOV_' in full_path['collection'], 'Must select FOV_** folder'
        self.fov_path = Path(data_path)
        self.fov = int(re.search('alf/FOV_(\\d+)', full_path['collection']).group(1))
        self.plane_path = self.session_path.joinpath('raw_bin_files', f'plane{self.fov}')

        self.fov_info = full_path['collection'].split('/')[-1]
        self.session_info = '/'.join([full_path['subject'], full_path['date'], full_path['number']])


    def load_data(self):

        self.raw_data = False

        # Load in imaging times
        self.imaging_times = self.load_times()

        # Load in cell types
        self.cell_types = self.load_cell_types()

        # Load in cell type names
        self.cell_type_names = self.load_cell_type_names()

        # Load in mean images
        self.mean_image = self.load_mean_image()

        # Load in cell activity
        self.cell_activity = self.load_activity()

        # Load in neuropil activity
        self.neuropil_activity = self.load_neuropil_activity()

        # Load in cell mask
        self.cell_mask = self.load_cell_mask()

        # Load in raw data
        self.imaging_data = self.load_raw()

        # Load in suite2p ops data
        self.ops = self.load_ops()

        # Load in suite2p stat data
        self.stat = self.load_stats()
        self.skew = self.get_skew()

        # Load in bad frames
        self.bad_frames = self.load_bad_frames()

        self.concat_mask = {val: None for val in self.get_cell_types()}

        # Load in frames and times for pc signal (tPc sampled at a subset of the total frames)
        self.tPc_frames = self.get_pc_frames()
        self.tPc_times = self.imaging_times[self.tPc_frames]

    def load_times(self):

        frame_times = np.load(self.fov_path.joinpath('mpci.times.npy'))
        stack_pos = np.load(self.fov_path.joinpath('mpciROIs.stackPos.npy'))
        timeshift = np.load(self.fov_path.joinpath('mpciStack.timeshift.npy'))
        roi_offsets = timeshift[stack_pos[:, len(timeshift.shape)]]
        roi_times = np.tile(frame_times, (roi_offsets.size, 1)) + roi_offsets[np.newaxis, :].T

        return roi_times[self.fov]

    def load_cell_type_names(self):
        cell_type_names = pd.read_csv(self.fov_path.joinpath('mpciROITypes.names.tsv'), sep='\t')
        return cell_type_names

    def load_cell_types(self):
        cell_types = np.load(self.fov_path.joinpath('mpciROIs.mpciROITypes.npy'))
        return cell_types

    def load_mean_image(self):
        mean_image = np.load(self.fov_path.joinpath('mpciMeanImage.images.npy'))
        return mean_image

    def load_activity(self):
        cell_activity = np.load(self.fov_path.joinpath('mpci.ROIActivityF.npy'))
        return cell_activity

    def load_neuropil_activity(self):
        neuropil_activity = np.load(self.fov_path.joinpath('mpci.ROINeuropilActivityF.npy'))
        return neuropil_activity

    def load_cell_mask(self):
        mask = sparse.load_npz(self.fov_path.joinpath('mpciROIs.masks.sparse_npz'))
        return mask

    def load_raw(self):
        if not self.plane_path.joinpath('data.bin').exists():
            self.raw_data = False
            return None

        # TODO read in data type and frame size from metadata
        self.raw_data = True
        raw = np.memmap(self.plane_path.joinpath('data.bin'), dtype=np.int16,
                        shape=(self.imaging_times.size, 512, 512))
        return raw

    def load_ops(self):
        ops = np.load(self.fov_path.joinpath('_suite2p_ROIData.raw.zip'), allow_pickle=True)['ops'].item()
        return ops

    def load_stats(self):
        stat = np.load(self.fov_path.joinpath('_suite2p_ROIData.raw.zip'), allow_pickle=True)['stat']
        return stat

    def load_bad_frames(self):
        frames = np.load(self.fov_path.joinpath('mpci.badFrames.npy'))
        return frames

    def get_ops(self, key):
        return self.ops[key]

    def get_skew(self):
        skew = np.zeros((self.stat.shape))
        for i, stat in enumerate(self.stat):
            skew[i] = stat['skew']

        return skew

    def get_image_at_frame(self, f0):
        return self.imaging_data[f0]

    def get_image_at_time(self, t0):
        return self.imaging_data[self.convert_frame_to_time(t0)]

    def convert_frame_to_time(self, t0):
        return np.searchsorted(self.imaging_times, t0)

    def get_skew_for_cell(self, cell_type=None):
        if cell_type is None:
            return self.skew
        else:
            return self.skew[self.cell_types == cell_type]

    def get_activity_for_cell(self, cell_idx):
        return self.cell_activity[cell_idx]

    def get_neuropil_activity_for_cell(self, cell_idx):
        return self.neuropil_activity[cell_idx]

    def get_average_cell_activity(self, cell_type=None):
        return self._get_average_activity(self.cell_activity, cell_type)

    def get_average_neuropil_activity(self, cell_type=None):
        return self._get_average_activity(self.neuropil_activity, cell_type)

    def get_fluorescence(self, cell_type=None):
        return self.get_average_cell_activity(cell_type) / 0.7 * self.get_average_neuropil_activity(cell_type)

    def _get_average_activity(self, activity, cell_type):
        if cell_type is None:
            return np.mean(activity, axis=1)
        else:
            return np.mean(activity[:, self.cell_types == cell_type], axis=1)

    def _get_mask(self, cell_type):
        if cell_type is None:
            return self.cell_mask
        else:
            return self.cell_mask[np.where(self.cell_types == cell_type)[0], :, :]

    def get_cell_names(self):
        return self.cell_type_names['roi_labels'].values

    def get_cell_types(self):
        return self.cell_type_names['roi_values'].values

    def get_cell_name_from_type(self, cell_type):
        return self.cell_type_names.loc[self.cell_type_names['roi_values'] == cell_type, 'roi_labels'].iloc[0]

    def get_cell_type_from_name(self, cell_name):
        return self.cell_type_names.loc[self.cell_type_names['roi_labels'] == cell_name, 'roi_values'].iloc[0]

    def get_concatenated_mask(self, cell_type=None):

        if self.concat_mask[cell_type] is None:
            mask = self._get_mask(cell_type)
            mask = mask > 0
            weights = np.arange(mask.shape[0])[:, np.newaxis, np.newaxis]
            mask = mask * weights  # Broadcasting in sparse format
            concat = sparse.sum(mask, axis=0)
            concat = sparse.asnumpy(concat).astype(int)
            # concat[concat == 0] = np.nan
            self.concat_mask[cell_type] = concat

        return self.concat_mask[cell_type]

    def get_hist(self, vals, width=None, nbins=100):
        if width is not None:
            nbins = np.arange(np.nanmin(vals), np.nanmax(vals) + width, width)
        counts, bins = np.histogram(vals, bins=nbins)
        #TODO make bin in center
        return counts, bins[:-1]

    def get_npcs(self):
        return self.ops['regPC'].shape[1]

    def get_pc_intervals(self, interval, n_intervals=3):
        assert interval < n_intervals, f'interval must be between 0 and {n_intervals - 1}'
        n_ints = int(self.ops['tPC'].shape[0] / n_intervals)
        return int(n_ints * interval), int(n_ints * (interval + 1))

    def get_pc_frames(self):

        # Taken from https://github.com/main/suite2p/blob/main/suite2p/run_s2p.py#L152
        nsamp = min(2000 if self.ops['nframes'] < 5000 or self.ops['Ly'] > 700 or self.ops['Lx'] > 700 else 5000,
                    self.ops['nframes'])
        inds = np.linspace(0, self.ops['nframes'] - 1, nsamp).astype("int")

        return inds



BUTTON_DEFAULT_STYLE = """ QPushButton {
                            color: #282828;
                            background-color: #C0C0C0;
                            border: 1px solid #A9A9A9;
                            border-radius: 5px;
                            padding: 2px 2px;
                            }
                            QPushButton:hover {
                                background-color: #E5E4E2;
                            }
                        """

BUTTON_MUTED_STYLE = """ QPushButton {
                            color: #A9A9A9;
                            background-color: #C0C0C0;
                            border: 1px solid #A9A9A9;
                            border-radius: 5px;
                            padding: 2px 2px;
                            }
                            QPushButton:hover {
                                background-color: #E5E4E2;
                            }
                        """

SLIDER_STYLE = """ QSlider::groove:horizontal {
                        height: 2px;
                        background: #ddd;
                        border-radius: 1px;
                    }
                    QSlider::handle:horizontal {
                        background: #3498db;
                        border: 1px solid #2980b9;
                        width: 10px;
                        height: 10px;
                        border-radius: 5px;
                        margin: -5px 0px;
                    }
                    QSlider::handle:horizontal:hover {
                        background: #2980b9;
                    }
                    QSlider::sub-page:horizontal {
                        background: #3498db;
                        border-radius: 1px;
                    }
                    QSlider::add-page:horizontal {
                        background: #eee;
                        border-radius: 1px;
                    }
                """

class MesoscopeQcSetup():

    def init_ui(self):

        self.resize(1600, 1000)
        self.setWindowTitle('Mesoscope QC GUI')
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        central_widget = QWidget()
        central_layout = QVBoxLayout(central_widget)
        self.setCentralWidget(central_widget)

        splitter = QSplitter(Qt.Horizontal)
        central_layout.addWidget(splitter)

        splitter1 = QSplitter(Qt.Vertical)
        splitter2 = QSplitter(Qt.Vertical)
        splitter3 = QSplitter(Qt.Vertical)
        splitter4 = QSplitter(Qt.Vertical)

        splitter.addWidget(splitter1)
        splitter.addWidget(splitter2)
        splitter.addWidget(splitter3)
        splitter.addWidget(splitter4)

        def add_layout_with_image_and_lut(title, add_button=None):

            layout = pg.GraphicsLayout()

            label = pg.LabelItem(title)
            layout.addItem(label, 0, 0)

            if add_button is not None:
                button = QPushButton(add_button)
                button_widget = QWidget()
                button_layout = QVBoxLayout(button_widget)
                button_layout.setContentsMargins(0, 0, 0, 0)
                button_layout.addWidget(button)
                button_widget.setStyleSheet("background-color: transparent;")
                # Create a QGraphicsProxyWidget and add to scene
                proxy = QGraphicsProxyWidget()
                proxy.setWidget(button_widget)
                layout.addItem(proxy, 0, 1)

                button.setStyleSheet("background-color : white")
                button.setStyleSheet(BUTTON_DEFAULT_STYLE)
            else:
                button = None

            plot = pg.ViewBox()
            plot.setAspectLocked(True)

            layout.addItem(plot, 1, 0)

            lut = pg.HistogramLUTItem()
            lut.axis.setVisible(False)

            layout.addItem(lut, 1, 1)

            layout.layout.setSpacing(0)
            layout.layout.setContentsMargins(0, 0, 0, 0)
            layout.layout.setColumnStretchFactor(0, 10)
            layout.layout.setColumnStretchFactor(1, 0)
            layout.layout.setColumnMaximumWidth(1, 1)

            return layout, plot, lut, label, button

        # -------------------------------------------------------------------------------------------------
        # Display widgets
        # -------------------------------------------------------------------------------------------------

        self.plot_items = Bunch()
        self.plot_items['raw_image'] = Bunch()
        self.plot_items['mean_image'] = Bunch()
        self.plot_items['ops_diff_image'] = Bunch()
        self.plot_items['ops_merge_image'] = Bunch()
        self.plot_items['ops_raw_image'] = Bunch()
        self.plot_items['fluo_signal'] = Bunch(plot=pg.PlotItem())
        self.plot_items['pc_signal'] = Bunch(plot=pg.PlotItem())
        self.plot_items['xy_signal'] = Bunch(plot=pg.PlotItem())
        self.plot_items['snr_scatter'] = Bunch(plot=pg.PlotItem())
        self.plot_items['skew_hist'] = Bunch(plot=pg.PlotItem())
        self.plot_items['fluo_hist'] = Bunch(plot=pg.PlotItem())
        self.plot_items['pc_hist'] = Bunch(plot=pg.PlotItem())
        self.plot_items['xy_hist'] = Bunch(plot=pg.PlotItem())
        self.plot_items['pc_traces'] = Bunch(plot=pg.PlotItem())

        raw_image_group, self.plot_items['raw_image']['plot'], self.plot_items['raw_image']['lut'], *_  = (
            add_layout_with_image_and_lut('Raw image'))
        self.raw_image_lut = self.plot_items['raw_image']['lut']

        (mean_image_group, self.plot_items['mean_image']['plot'], self.plot_items['mean_image']['lut'], _,
         self.mask_toggle) = (add_layout_with_image_and_lut('Mean image', add_button='Mask'))

        ops_diff_group, self.plot_items['ops_diff_image']['plot'], self.plot_items['ops_diff_image']['lut'], *_ = (
            add_layout_with_image_and_lut('Difference'))

        (self.ops_merge_group, self.plot_items['ops_merge_image']['plot'], self.plot_items['ops_merge_image']['lut_0'],
         *_) = (add_layout_with_image_and_lut('Merged'))
        # We need lut for each layer
        self.plot_items['ops_merge_image']['lut_1'] = pg.HistogramLUTItem()
        self.plot_items['ops_merge_image']['lut_1'].axis.setVisible(False)

        (ops_raw_group, self.plot_items['ops_raw_image']['plot'], self.plot_items['ops_raw_image']['lut'],
         self.layer_info, self.layer_toggle) = (
            add_layout_with_image_and_lut('Top', add_button='Top'))

        # -------------------------------------------------------------------------------------------------
        # Interaction widgets
        # -------------------------------------------------------------------------------------------------

        # Slider to span through the session
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setStyleSheet(SLIDER_STYLE)

        # Label and dropdown to select cell type
        cell_text = QLabel('Select cell type:')
        self.cell_list = QStandardItemModel()
        self.cell_combobox = QComboBox()
        self.cell_combobox.setModel(self.cell_list)

        # Label and dropdown to select pc number
        pc_text = QLabel('Select PC #:')
        self.pc_list = QStandardItemModel()
        self.pc_combobox = QComboBox()
        self.pc_combobox.setModel(self.pc_list)

        # Button to select FOV and labels to display info
        self.session_button = QPushButton('Load FOV')
        self.session_info = QLabel()
        self.fov_info = QLabel()


        # -------------------------------------------------------------------------------------------------
        # Splitter 1, slider, raw_image, mean_image, fluo_signal, pc_signal and xy_signal
        # -------------------------------------------------------------------------------------------------

        splitter1_graphics_layout = pg.GraphicsLayout()
        splitter1_graphics_layout.addItem(raw_image_group, 0, 0)
        splitter1_graphics_layout.addItem(mean_image_group, 0, 1)
        splitter1_graphics_layout.addItem(self.plot_items['fluo_signal']['plot'], 1, 0, colspan=2)
        splitter1_graphics_layout.addItem(self.plot_items['xy_signal']['plot'], 2, 0, colspan=2)
        splitter1_graphics_layout.addItem(self.plot_items['pc_signal']['plot'], 3, 0, colspan=2)
        splitter1_graphics_layout.layout.setContentsMargins(0, 0, 0, 0)
        splitter1_graphics_layout.layout.setRowStretchFactor(0, 2)
        splitter1_graphics_layout.layout.setRowStretchFactor(1, 1)
        splitter1_graphics_layout.layout.setRowStretchFactor(2, 1)
        splitter1_graphics_layout.layout.setRowStretchFactor(3, 1)
        splitter1_graphics_widget = pg.GraphicsLayoutWidget()
        splitter1_graphics_widget.addItem(splitter1_graphics_layout)

        splitter1_widget = QWidget()
        splitter1_widget.setStyleSheet("background-color : black")
        splitter1_layout = QVBoxLayout(splitter1_widget)
        splitter1_layout.addWidget(self.slider)
        splitter1_layout.addWidget(splitter1_graphics_widget)

        splitter1.addWidget(splitter1_widget)

        # -------------------------------------------------------------------------------------------------
        # Splitter 2, snr_scatter, skew_hist, fluo_hist, pc_hist and xy_hist
        # -------------------------------------------------------------------------------------------------

        splitter2_graphics_layout = pg.GraphicsLayout()
        splitter2_graphics_layout.addItem(self.plot_items['snr_scatter']['plot'], row=0, col=0)
        splitter2_graphics_layout.addItem(self.plot_items['skew_hist']['plot'], row=1, col=0)
        splitter2_graphics_layout.addItem(self.plot_items['fluo_hist']['plot'], row=2, col=0)
        splitter2_graphics_layout.addItem(self.plot_items['xy_hist']['plot'], row=3, col=0)
        splitter2_graphics_layout.addItem(self.plot_items['pc_hist']['plot'], row=4, col=0)
        splitter2_graphics_layout.layout.setContentsMargins(0, 0, 0, 0)
        splitter2_graphics_widget = pg.GraphicsLayoutWidget()
        splitter2_graphics_widget.addItem(splitter2_graphics_layout)

        splitter2.addWidget(splitter2_graphics_widget)

        # -------------------------------------------------------------------------------------------------
        # Splitter 3 top: pc_traces, bottom: ops_diff_image, ops_merge_image, ops_raw_image
        # -------------------------------------------------------------------------------------------------
        splitter3_top_graphics_layout = pg.GraphicsLayout()
        splitter3_top_graphics_layout.addItem(self.plot_items['pc_traces']['plot'], 0, 0)
        splitter3_top_graphics_widget = pg.GraphicsLayoutWidget()
        splitter3_top_graphics_widget.addItem(splitter3_top_graphics_layout)

        splitter3_bottom_graphics_layout = pg.GraphicsLayout()
        splitter3_bottom_graphics_layout.addItem(ops_diff_group, row=0, col=0)
        splitter3_bottom_graphics_layout.addItem(self.ops_merge_group, row=1, col=0)
        splitter3_bottom_graphics_layout.addItem(ops_raw_group, row=2, col=0)
        splitter3_bottom_graphics_widget = pg.GraphicsLayoutWidget()
        splitter3_bottom_graphics_widget.addItem(splitter3_bottom_graphics_layout)

        splitter3.addWidget(splitter3_top_graphics_widget)
        splitter3.addWidget(splitter3_bottom_graphics_widget)

        # -------------------------------------------------------------------------------------------------
        # Splitter 4 cell selection dropdown, pc selection dropdown, load data button, data info text
        # -------------------------------------------------------------------------------------------------

        # Add the cell and pc dropdowns into a group
        dropdown_widget = QGroupBox()
        dropdown_widget.setMaximumHeight(180)
        dropdown_layout = QVBoxLayout(dropdown_widget)
        dropdown_layout.addWidget(cell_text)
        dropdown_layout.addWidget(self.cell_combobox)
        dropdown_layout.addWidget(pc_text)
        dropdown_layout.addWidget(self.pc_combobox)

        # Add the session load and session info into a group
        sessions_widget = QGroupBox()
        sessions_widget.setMaximumHeight(150)
        sessions_layout = QVBoxLayout(sessions_widget)
        sessions_layout.addWidget(self.session_button)
        sessions_layout.addWidget(self.session_info)
        sessions_layout.addWidget(self.fov_info)

        splitter4_widget = QWidget()
        splitter4_layout = QVBoxLayout(splitter4_widget)
        splitter4_layout.addWidget(dropdown_widget)
        splitter4_layout.addWidget(sessions_widget)
        splitter4.addWidget(splitter4_widget)

        # -------------------------------------------------------------------------------------------------
        # Format default layout of splitters
        # -------------------------------------------------------------------------------------------------
        splitter.setStretchFactor(0, 8)
        splitter.setStretchFactor(1, 4)
        splitter.setStretchFactor(2, 4)
        splitter.setStretchFactor(3, 3)

        splitter3.setStretchFactor(0, 1)
        splitter3.setStretchFactor(1, 3)


class MesoscopeQcGUI(QMainWindow, MesoscopeQcSetup):
    # -------------------------------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------------------------------

    def __init__(self):
        super().__init__()
        self.init_ui()

        # Load in settings
        self.settings = QSettings("Mesoscope qc gui")

        # Load in some default colours and colourmaps
        self.get_pens()
        self.cmap_mask = matplotlib.colormaps['hsv']
        self.cmap_skew = matplotlib.colormaps['Set1']
        self.cmap_pc = matplotlib.colormaps['YlOrBr']
        self.cmap_fluo = matplotlib.colormaps['PiYG']
        self.cmap_top = self.get_colormap('vanimo', locs=np.arange(128)[::-1])
        self.cmap_bottom = self.get_colormap('vanimo', locs=np.arange(128) + 128)

        # Add connections
        self.session_button.clicked.connect(self.on_load_data)
        self.layer_toggle.clicked.connect(self.toggle_layer)
        self.mask_toggle.clicked.connect(self.toggle_mask)
        self.slider.valueChanged.connect(self.on_slider_moved)
        self.raw_image_lut.sigLevelChangeFinished.connect(self.update_raw_level)
        self.cell_combobox.activated.connect(self.on_cell_selected)
        self.pc_combobox.activated.connect(self.on_pc_selected)

        for key in self.plot_items:
            self.plot_items[key]['plot'].scene().sigMouseClicked.connect(self.on_double_click)

        # Add links between axis
        self.plot_items['fluo_signal']['plot'].setXLink(self.plot_items['xy_signal']['plot'])
        self.plot_items['xy_signal']['plot'].setXLink(self.plot_items['pc_signal']['plot'])

    # -------------------------------------------------------------------------------------------------
    # Entry point
    # -------------------------------------------------------------------------------------------------

    def reset_gui(self):
        """
        Reset gui to initial state
        :return:
        """
        # Clear all plots from previous session
        for items in self.plot_items:
            if isinstance(self.plot_items[items]['plot'], pg.ViewBox):
                self.plot_items[items]['plot'].enableAutoRange(axis=pg.ViewBox.XYAxes, enable=True)
            else:

                self.plot_items[items]['plot'].getViewBox().enableAutoRange(axis=pg.ViewBox.XYAxes, enable=True)

            keys = [k for k in self.plot_items[items].keys() if k not in ['plot', 'lut', 'lut_0', 'lut_1']]
            for key in keys:
                _ = self.plot_items[items].pop(key)
            self.plot_items[items]['plot'].clear()

        # Clear combobox items
        self.cell_list.clear()
        self.pc_list.clear()

        # Reinitialise variables to default values
        self.cell_idx = 1
        self.pc_idx = 0
        self.frame_idx = 0
        self.layer_idx = 0
        self.raw_levels = None

        # Reset the slider to the start
        self.slider.setValue(0)

        # Reset the raw image lut histogram
        self.raw_image_lut.setImageItem(pg.ImageItem(data=np.zeros((256, 256), dtype=np.uint8)))

    def load_gui(self, data_path):
        """
        Load in data and plots
        :param data_path: path to the alf/FOV_** folder to display
        :return:
        """
        # Load in data for the session
        self.ml = MesoscopeLoader(data_path)
        self.ml.load_data()

        # Update the session and FOV info
        self.session_info.setText(self.ml.session_info)
        self.fov_info.setText(self.ml.fov_info)

        # Set the limits of the slider
        self.slider.setTickInterval(1)
        self.slider.setMinimum(0)
        self.slider.setMaximum(self.ml.imaging_times.size - 1)

        # Populate the cell type combobox
        for i, cell in enumerate(self.ml.get_cell_names()):
            if cell == 'cell':
                idx = i
            item = QStandardItem(cell)
            item.setEditable(False)
            self.cell_list.appendRow(item)
        self.cell_combobox.setCurrentIndex(idx)

        # Populate the PC combobox
        for pc in range(self.ml.get_npcs()):
            item = QStandardItem(f'PC{pc}')
            item.setEditable(False)
            self.pc_list.appendRow(item)
        self.pc_combobox.setCurrentIndex(self.pc_idx)

        # Update the plots
        self.init_plots()

        # Initialise the histogram LUT levels
        self.raw_image_lut.autoHistogramRange()
        self.update_raw_level()

    # -------------------------------------------------------------------------------------------------
    # Interactions
    # -------------------------------------------------------------------------------------------------

    def on_load_data(self):
        """
        Load file dialog to select data
        :return:
        """
        prev_path = self.settings.value("lastDir", QDir.homePath())

        data_path = QFileDialog.getExistingDirectory(self, "Select FOV Directory", prev_path)
        if data_path:
            self.reset_gui()
            self.load_gui(data_path)
            self.settings.setValue("lastDir", str(Path(data_path).parent))


    def on_cell_selected(self, cell):
        """
        Update selected cell type and redraw plots
        :param cell: idx of dropdown from cell_combobox
        :return:
        """
        self.cell_idx = cell
        self.update_cell_plots()

    def on_pc_selected(self, pc):
        """
        Update selected pc number and redraw plots
        :param pc: idx of dropdown from cell_combobox
        :return:
        """
        self.pc_idx = pc
        self.update_pc_plots()

    def on_slider_moved(self, frame):
        """
        Update selected frame idx and redraw plots
        :param frame: location of slider
        :return:
        """
        self.frame_idx = frame
        self.raw_image_lut.blockSignals(True)
        self.update_frame_plots()
        self.raw_image_lut.blockSignals(False)

    def toggle_mask(self):
        """
        Toggle layer showing cell ROIs on/ off
        :return:
        """
        plot = self.plot_items['mean_image']
        if plot['mask'].isVisible():
            plot['mask'].setVisible(False)
            self.mask_toggle.setStyleSheet(BUTTON_MUTED_STYLE)
        else:
            plot['mask'].setVisible(True)
            self.mask_toggle.setStyleSheet(BUTTON_DEFAULT_STYLE)

    def toggle_layer(self):
        """
        Toggle ops image showing data from top/ bottom 500 frames
        :return:
        """
        plot = self.plot_items['ops_merge_image']

        if self.layer_idx == 0:
            self.layer_toggle.setText('Bottom')
            self.layer_info.setText('Bottom')
            self.ops_merge_group.removeItem(plot[f'lut_{self.layer_idx}'])
            self.layer_idx = 1
            self.ops_merge_group.addItem(plot[f'lut_{self.layer_idx}'], 1, 1)
            plot[f'img_{self.layer_idx}'].setLookupTable(self.cmap_bottom.getLookupTable())
        else:
            self.layer_toggle.setText('Top')
            self.layer_info.setText('Top')
            self.ops_merge_group.removeItem(plot[f'lut_{self.layer_idx}'])
            self.layer_idx = 0
            self.ops_merge_group.addItem(plot[f'lut_{self.layer_idx}'], 1, 1)
            plot[f'img_{self.layer_idx}'].setLookupTable(self.cmap_top.getLookupTable())

        self.plot_ops_raw()

    def on_double_click(self, event):
        if event.double():
            vb = event.currentItem
            if isinstance(vb, pg.ViewBox):
                vb.enableAutoRange(axis=pg.ViewBox.XYAxes, enable=True)


    # -------------------------------------------------------------------------------------------------
    # Update plots
    # -------------------------------------------------------------------------------------------------

    def update_raw_level(self):
        """
        Update the raw image lut levels
        :return:
        """
        self.raw_levels = self.raw_image_lut.getLevels()

    def update_pc_plots(self):
        """
        Update all plots that change when PC number is changed
        :return:
        """
        self.plot_ops_raw()
        self.plot_ops_diff()
        self.plot_ops_merge()

        self.plot_pc_signal()
        self.plot_pc_signal_point()
        self.plot_pc_hist()

        self.plot_pc_traces()
        self.plot_pc_points()

    def update_cell_plots(self):
        """
        Update all plots that change when PC number is changed
        :return:
        """
        self.plot_fluorescence_signal()
        self.plot_fluorescence_point()
        self.plot_fluorescence_hist()

        self.plot_snr()
        self.plot_mask()

    def update_frame_plots(self):
        """
        Update all plots when slider is moved to change frame
        :return:
        """

        if self.ml.raw_data:
            self.plot_raw_image()
        self.plot_fluorescence_point()
        self.plot_xy_point()
        self.plot_pc_signal_point()

    def init_plots(self):

        if self.ml.raw_data:
            self.plot_raw_image()
        else:
            self.plot_fake_raw_image()

        self.plot_mean_image()
        self.plot_mask()

        self.plot_fluorescence_signal()
        self.plot_bad_frames()
        self.plot_fluorescence_point()
        self.plot_fluorescence_hist()

        self.plot_pc_signal()
        self.plot_pc_signal_point()
        self.plot_pc_hist()

        self.plot_xy_signal()
        self.plot_xy_point()
        self.plot_xy_hist()

        self.plot_snr()
        self.plot_skew_hist()

        self.plot_pc_traces()
        self.plot_pc_points()

        self.plot_ops_diff()
        self.plot_ops_merge()
        self.plot_ops_raw()


    # -------------------------------------------------------------------------------------------------
    # Plotting utils
    # -------------------------------------------------------------------------------------------------

    def get_pens(self):

        self.bpen = pg.mkPen(color='#0aaffd', width=2)

        self.rpen = pg.mkPen(color='r', width=2)
        self.rbrush = pg.mkBrush(color='r')

        self.wpen = pg.mkPen(color='w', width=2)
        self.wbrush = pg.mkBrush(color='w')

        self.ypen = pg.mkPen(color='y', width=2)
        self.ybrush = pg.mkBrush(color='y')

        self.gpen = pg.mkPen(color='g', width=2)
        self.gbrush = pg.mkBrush(color='g')

        self.dashed_pen = pg.mkPen('#808080', width=1, style=pg.QtCore.Qt.DashLine)
        self.dashed_pen_thick = pg.mkPen('#808080', width=2, style=pg.QtCore.Qt.DashLine)

    @staticmethod
    def get_colormap(name, locs=None):
        # Get colormap from Matplotlib in Pyqtgraph format
        cmap = matplotlib.colormaps[name]
        # Extract colors and positions
        if locs is not None:
            colors = [cmap(i) for i in locs]
            positions = np.linspace(0, 1, len(locs))
        else:
            colors = [cmap(i) for i in range(cmap.N)]
            positions = np.linspace(0, 1, cmap.N)
        # Convert to PyQtGraph format (0-255 range for RGBA)
        colors = [(int(r * 255), int(g * 255), int(b * 255), int(a * 255)) for r, g, b, a in colors]
        return pg.ColorMap(positions, colors)


    @staticmethod
    def set_axis_labels(plot, xlabel, ylabel):
        ax_x = plot.getAxis('bottom')
        ax_x.setLabel(xlabel)
        ax_y = plot.getAxis('left')
        ax_y.setLabel(ylabel)

    @staticmethod
    def _plot_image(data, plot, name, opacity=1, set_lut=True):

        img = plot.get(name, None)
        plot_item = plot['plot']
        if img is None:
            img = pg.ImageItem()
            transform = QTransform()
            transform.scale(1, 1)
            transform.translate(0, 0)
            img.setTransform(transform)
            plot_item.addItem(img)

        img.setImage(data)
        img.setOpacity(opacity)
        if set_lut:
            plot['lut'].setImageItem(img)

        return img

    @staticmethod
    def _plot_curve(x, y, plot, name, pen=pg.mkPen('w')):

        curve = plot.get(name, None)
        plot_item = plot['plot']
        if curve is None:
            curve = pg.PlotCurveItem()
            plot_item.addItem(curve)

        curve.setData(x=x, y=y, pen=pen)

        return curve

    @staticmethod
    def _plot_bar(x, y, plot, name, pen=None, brush=pg.mkBrush('w'), width=None):

        bar = plot.get(name, None)
        plot_item = plot['plot']
        if bar is not None:
            plot_item.removeItem(bar)

        width = width or 0.85 * np.median(np.diff(y))
        bar = pg.BarGraphItem(x=y, height=x, width=width, brush=brush, pen=pen)
        plot_item.addItem(bar)

        return bar

    @staticmethod
    def _plot_scatter(x, y, plot, name, pen=pg.mkPen('w'), brush=pg.mkBrush('w'), size=8):

        scatter = plot.get(name, None)
        plot_item = plot['plot']
        if scatter is None:
            scatter = pg.ScatterPlotItem()
            scatter.setZValue(10)
            plot_item.addItem(scatter)

        scatter.setData(x=x, y=y, pen=pen, brush=brush, size=size)

        return scatter

    @staticmethod
    def _plot_rectangle(t0, t1, plot, name, pen=pg.mkPen('w'), brush=pg.mkBrush('w')):

        rectangle = plot.get(name, None)
        plot_item = plot['plot']
        if rectangle is not None:
            plot_item.removeItem(rectangle)

        rectangle = pg.LinearRegionItem(values=(t0, t1),
                                     orientation=pg.LinearRegionItem.Vertical,
                                     brush=brush, pen=pen, movable=False)
        plot_item.addItem(rectangle)

        return rectangle

    # -------------------------------------------------------------------------------------------------
    # Image plots
    # -------------------------------------------------------------------------------------------------

    def plot_raw_image(self):

        plot = self.plot_items['raw_image']
        data = self.ml.get_image_at_frame(self.frame_idx)
        plot['img'] = self._plot_image(data, plot, 'img')
        if self.raw_levels is not None:
           self.raw_image_lut.setLevels(self.raw_levels[0], self.raw_levels[1])

    def plot_fake_raw_image(self):

        plot = self.plot_items['raw_image']
        data = np.zeros_like(self.ml.mean_image)
        plot['img'] = self._plot_image(data, plot, 'img')

    def plot_mean_image(self):
        plot = self.plot_items['mean_image']
        data = self.ml.mean_image
        plot['img'] = self._plot_image(data, plot, 'img')
        plot['lut'].autoHistogramRange()

    def plot_mask(self):
        plot = self.plot_items['mean_image']
        data = self.ml.get_concatenated_mask(self.cell_idx)

        # Get mask to set to nan later
        mask = data == 0

        # Compute size of cells
        idx, weights = np.unique(data[~mask], return_counts=True)
        weights = weights / np.max(weights)

        # Assign random color to each cell
        vals = np.random.default_rng(seed=1).integers(0, 256, int(np.nanmax(data) + 1))
        cols = self.cmap_mask(vals)

        # Assign opacity of each cell according to size of cell
        cols[idx, -1] = weights * 0.7

        # Convert data to rgba format
        data = cols[data].astype(np.float16)
        data[mask] = np.nan

        plot['mask'] = self._plot_image(data, plot, 'mask', set_lut=False)


    def plot_ops_diff(self):
        plot = self.plot_items['ops_diff_image']
        data = self.ml.ops['regPC'][0, self.pc_idx, :, :] - self.ml.ops['regPC'][1, self.pc_idx, :, :]
        plot['img'] = self._plot_image(data, plot, 'img')
        plot['lut'].autoHistogramRange()

    def plot_ops_merge(self):
        plot = self.plot_items['ops_merge_image']
        data1 = self.ml.ops['regPC'][0, self.pc_idx, :, :]
        data2 = self.ml.ops['regPC'][1, self.pc_idx, :, :]
        plot['img_0'] = self._plot_image(data1, plot, 'img_1', opacity=0.5, set_lut=False)
        plot['img_1'] = self._plot_image(data2, plot, 'img_2', opacity=0.5, set_lut=False)
        plot['lut_0'].setImageItem(plot['img_0'])
        plot['lut_1'].setImageItem(plot['img_1'])
        plot['img_0'].setLookupTable(self.cmap_top.getLookupTable())
        plot['img_1'].setLookupTable(self.cmap_bottom.getLookupTable())

    def plot_ops_raw(self):
        plot = self.plot_items['ops_raw_image']
        data = self.ml.ops['regPC'][self.layer_idx, self.pc_idx, :, :]
        plot['img'] = self._plot_image(data, plot, 'img')
        plot['lut'].autoHistogramRange()

    # -------------------------------------------------------------------------------------------------
    # Trace plots
    # -------------------------------------------------------------------------------------------------

    def plot_fluorescence_signal(self):
        plot = self.plot_items['fluo_signal']
        x = self.ml.imaging_times
        y = self.ml.get_fluorescence(self.cell_idx)
        plot['curve'] = self._plot_curve(x, y, plot,'curve', pen=self.bpen)
        self.set_axis_labels(plot['plot'], 'Time (s)', '&Delta;F/ F')

    def plot_pc_signal(self):
        plot = self.plot_items['pc_signal']
        x = self.ml.tPc_times
        y = self.ml.ops['tPC'][:, self.pc_idx]
        plot['curve'] = self._plot_curve(x, y, plot, 'curve', pen=self.wpen)
        self.set_axis_labels(plot['plot'], 'Time (s)', 'Magnitude')

    def plot_xy_signal(self):
        plot = self.plot_items['xy_signal']
        for val, pen in zip(['x', 'y'], [self.ypen, self.gpen]):
            x = self.ml.imaging_times
            y = self.ml.ops[f'{val}off']
            plot[f'curve_{val}'] = self._plot_curve(x, y, plot, f'curve_{val}', pen=pen)

        self.set_axis_labels(plot['plot'], 'Time (s)', 'Shift')

    def plot_pc_traces(self):
        plot = self.plot_items['pc_traces']
        plot['legend'] = pg.LegendItem(offset=(-1, 1))
        plot['legend'].setParentItem(plot['plot'])

        for i, (name, col) in enumerate(zip(['rigid', 'nonrigid', 'nonrigid_max'], ['white', 'orangered', 'slateblue'])):
            x = np.arange(self.ml.ops['regDX'].shape[0])
            y = self.ml.ops['regDX'][:, i]
            pen = pg.mkPen(color=col, width=4)
            plot[f'curve_{name}'] = self._plot_curve(x, y, plot, f'curve_{name}', pen=pen)
            plot['legend'].addItem(plot[f'curve_{name}'], name)

        self.set_axis_labels(plot['plot'], 'PC #', 'Pixel shift')

    # -------------------------------------------------------------------------------------------------
    # Scatter plots
    # -------------------------------------------------------------------------------------------------

    def plot_fluorescence_point(self):
        plot = self.plot_items['fluo_signal']
        x = self.ml.imaging_times[self.frame_idx]
        y = [self.ml.get_fluorescence(self.cell_idx)[self.frame_idx]]
        plot['scatter'] = self._plot_scatter(x, y, plot, 'scatter', pen=self.rpen, brush=self.rbrush)

    def plot_pc_signal_point(self):

        plot = self.plot_items['pc_signal']
        idx = np.searchsorted(self.ml.tPc_frames, self.frame_idx)
        x = [self.ml.tPc_times[idx]]
        y = [self.ml.ops['tPC'][idx, self.pc_idx]]
        plot['scatter'] = self._plot_scatter(x, y, plot, 'scatter', pen=self.rpen, brush=self.rbrush)

    def plot_pc_points(self):

        plot = self.plot_items['pc_traces']
        for i, name in enumerate(['rigid', 'nonrigid', 'nonrigid_max']):
            x = [self.pc_idx]
            y = [self.ml.ops['regDX'][self.pc_idx, i]]
            plot[f'scatter_{name}'] = self._plot_scatter(x, y, plot, f'scatter_{name}',
                                                                          pen=self.wpen, brush=self.wbrush)

    def plot_xy_point(self):

        plot = self.plot_items['xy_signal']
        for val, pen in zip(['x', 'y'], [self.ypen, self.gpen]):
            x = [self.ml.imaging_times[self.frame_idx]]
            y = [self.ml.ops[f'{val}off'][self.frame_idx]]
            plot[f'scatter_{val}'] = self._plot_scatter(x, y, plot, f'scatter_{val}', pen=self.rpen,
                                                        brush=self.rbrush)

    def plot_snr(self):
        plot = self.plot_items['snr_scatter']
        x = self.ml.get_average_neuropil_activity(self.cell_idx)
        y = self.ml.get_average_cell_activity(self.cell_idx)
        plot['scatter'] = self._plot_scatter(x, y, plot, 'scatter', pen=None, brush='slategray', size=2)
        plot['curve'] = self._plot_curve(x, x, plot, 'curve', pen=self.wpen)
        self.set_axis_labels(plot['plot'], 'Avg neuropil activity', 'Avg cell activity')

    # -------------------------------------------------------------------------------------------------
    # Bar plots
    # -------------------------------------------------------------------------------------------------

    def plot_skew_hist(self):

        plot = self.plot_items['skew_hist']
        plot['legend'] = pg.LegendItem(offset=(-1, 1), verSpacing=1)
        plot['legend'].setParentItem(plot['plot'])

        for i, cell_type in enumerate(self.ml.get_cell_types()):
            x, y = self.ml.get_hist(self.ml.get_skew_for_cell(cell_type), nbins=100)
            col = matplotlib.colors.to_hex(self.cmap_skew(i))
            plot[f'bar_{cell_type}'] = self._plot_bar(x, y, plot, f'bar_{cell_type}', brush=pg.mkBrush(color=col))
            plot['legend'].addItem(plot[f'bar_{cell_type}'], self.ml.get_cell_name_from_type(cell_type))

        self.set_axis_labels(plot['plot'], 'Skew', 'Counts')

    def plot_fluorescence_hist(self):

        plot = self.plot_items['fluo_hist']
        plot['legend'] = pg.LegendItem(offset=(-1, 1), verSpacing=1)
        plot['legend'].setParentItem(plot['plot'])

        x, y = self.ml.get_hist(self.ml.get_average_cell_activity(self.cell_idx), nbins=100)
        plot['bar_cell'] = self._plot_bar(x, y, plot, 'bar_cell',
                                          brush=pg.mkBrush(color=matplotlib.colors.to_hex(self.cmap_fluo(0))))
        plot['legend'].addItem(plot[f'bar_cell'], 'cell')

        x, y = self.ml.get_hist(self.ml.get_average_neuropil_activity(self.cell_idx), nbins=100)
        plot['bar_neuropil'] = self._plot_bar(x, y, plot, 'bar_neuropil',
                                              brush=pg.mkBrush(color=matplotlib.colors.to_hex(self.cmap_fluo(255))))
        plot['legend'].addItem(plot[f'bar_neuropil'], 'neuropil')

        self.set_axis_labels(plot['plot'], 'Avg activity', 'Counts')

    def plot_pc_hist(self):

        plot = self.plot_items['pc_hist']
        plot['legend'] = pg.LegendItem(offset=(-1, 1))
        plot['legend'].setParentItem(plot['plot'])
        n_intervals = 3
        for i in range(n_intervals):
            intervals = self.ml.get_pc_intervals(i, n_intervals=n_intervals)
            x, y = self.ml.get_hist(self.ml.ops['tPC'][intervals[0]:intervals[1], self.pc_idx], nbins=50)
            col = np.array(self.cmap_pc(int(i * self.cmap_pc.N / (n_intervals - 1)))) * 255
            plot[f'bar_{i}'] = self._plot_bar(x, y, plot, f'bar_{i}',
                                              brush=pg.mkBrush(color=col))
            plot['legend'].addItem(plot[f'bar_{i}'], f'{i + 1}/{n_intervals}')
            self.plot_pc_signal_intervals(intervals, col, f'interval_{i}')

        self.set_axis_labels(plot['plot'], 'Magnitude', 'Counts')

    def plot_xy_hist(self):
        plot = self.plot_items['xy_hist']
        plot['legend'] = pg.LegendItem(offset=(-1, 1))
        plot['legend'].setParentItem(plot['plot'])

        for val, brush in zip(['x', 'y'], [self.ybrush, self.gbrush]):
            x, y = self.ml.get_hist(self.ml.ops[f'{val}off'], width=1)
            plot[f'bar_{val}'] = self._plot_bar(x, y, plot, f'bar_{val}', brush=brush)
            plot['legend'].addItem(plot[f'bar_{val}'], val)

        for pos in [-10, 0, 10]:
            v_line = pg.InfiniteLine(pos=pos, angle=90, pen=self.dashed_pen_thick)
            plot['plot'].addItem(v_line)

        self.set_axis_labels(plot['plot'], 'Shift', 'Counts')

    # -------------------------------------------------------------------------------------------------
    # Misc plot additions
    # -------------------------------------------------------------------------------------------------

    def plot_pc_signal_intervals(self, intervals, col, name):
        times = self.ml.tPc_times
        plot = self.plot_items['pc_signal']
        col[-1] = 100
        brush = pg.mkBrush(color=col)
        plot[name] = self._plot_rectangle(times[intervals[0]], times[intervals[1]], plot, name, pen=None, brush=brush)

    def plot_bad_frames(self):
        plot = self.plot_items['fluo_signal']
        for fr in np.where(self.ml.bad_frames)[0]:
            v_line = pg.InfiniteLine(pos=self.ml.imaging_times[fr], angle=90,
                                     pen=self.dashed_pen)
            plot['plot'].addItem(v_line)



if __name__ == '__main__':
    app = QApplication([])
    gui = MesoscopeQcGUI()
    gui.show()
    app.exec_()
