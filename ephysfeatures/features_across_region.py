import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets
import sys
import pandas as pd
import numpy as np
from ibllib.atlas import AllenAtlas
import atlaselectrophysiology.ColorBar as cb
from ibllib.pipes.ephys_alignment import EphysAlignment
from atlaselectrophysiology.AdaptedAxisItem import replace_axis
from ephysfeatures.qrangeslider import QRangeSlider
import copy
from one.remote import aws
from one.api import ONE

pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')


# PLOT_TYPES = {'lfp': {'plot_type': 'probe', 'cmap': 'viridis'},
#               'ap':  {'plot_type': 'probe', 'cmap': 'plasma'},
#               'amp': {'plot_type': 'line', 'xlabel': 'Amplitude'},
#               'fr': {'plot_type': 'line', 'xlabel': 'Firing Rate'}}

PLOT_TYPES = {'psd_delta': {'plot_type': 'probe', 'cmap': 'viridis'},
              'psd_theta':  {'plot_type': 'probe', 'cmap': 'viridis'},
              'psd_alpha': {'plot_type': 'probe', 'cmap': 'viridis'},
              'psd_gamma': {'plot_type': 'probe', 'cmap': 'viridis'},
              'rms_ap': {'plot_type': 'probe', 'cmap': 'plasma'},
              'rms_lf': {'plot_type': 'probe', 'cmap': 'inferno'},
              'spike_rate': {'plot_type': 'probe', 'cmap': 'hot'},
              'amps': {'plot_type': 'line', 'xlabel': 'Amplitude'},
              'spike_rate_line': {'plot_type': 'line', 'xlabel': 'Firing Rate'}
              }


class RegionFeatureWindow(QtWidgets.QMainWindow):

    def __init__(self, one, region_ids=None, ba=None, download=True, size=(1600, 800)):
        super(RegionFeatureWindow, self).__init__()

        self.one = one
        # Initialise page counter
        self.page_num = 0
        self.page_idx = 0
        self.step = 10

        self.layout(size)

        self.ba = ba or AllenAtlas()
        br = self.ba.regions

        table_path = self.one.cache_dir.joinpath('bwm_features')
        if download:
            s3, bucket_name = aws.get_s3_from_alyx(alyx=self.one.alyx)
            aws.s3_download_folder("aggregates/bwm", table_path, s3=s3, bucket_name=bucket_name)

        channels = pd.read_parquet(table_path.joinpath('channels.pqt'))
        probes = pd.read_parquet(table_path.joinpath('probes.pqt'))
        features = pd.read_parquet(table_path.joinpath('raw_ephys_features.pqt'))

        df_voltage = pd.merge(features, channels, left_index=True, right_index=True)
        df_voltage = df_voltage.reset_index()
        data = pd.merge(df_voltage, probes, left_on='pid', right_index=True)
        data['rms_ap'] *= 1e6
        data['rms_lf'] *= 1e6

        depths = pd.read_parquet(table_path.joinpath('depths.pqt'))
        depths = depths.reset_index()
        depths = depths.rename(columns={'spike_rate': 'spike_rate_line'})

        self.data = pd.merge(data, depths, left_on=['pid', 'axial_um'], right_on=['pid', 'depths'], how='outer')
        self.data.loc[self.data['histology'] == 'alf', 'histology'] = 'resolved'

        del data, depths, channels, probes, features, df_voltage

        # Initialise region combobox
        if region_ids is not None:
            self.region_ids = region_ids
        else:
            self.region_ids = self.data.atlas_id.unique()
        acronyms = br.id2acronym(self.region_ids)

        for id, acro in enumerate(acronyms):
            item = QtGui.QStandardItem(id)
            item.setText(acro)
            item.setEditable(False)
            self.region_list.appendRow(item)

        # Initialise plot combobox
        for pl in PLOT_TYPES.keys():
            item = QtGui.QStandardItem(id)
            item.setText(pl)
            item.setEditable(False)
            self.plot_list.appendRow(item)

        # Inisitalise normalistion
        self.normalise = False
        self.slider.allowMove(False)
        self.levels = [0, 100]
        self.max_levels = [0, 100]

        # Initialise plot choice
        self.plot_name = 'psd_delta'
        self.plot_type = PLOT_TYPES[self.plot_name]['plot_type']
        self.feature_data = {}

        self.kpen_solid = pg.mkPen(color='k', style=QtCore.Qt.SolidLine, width=2)

    def layout(self, size):

        self.resize(size[0], size[1])
        main_widget = QtWidgets.QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QtWidgets.QVBoxLayout()
        main_widget.setLayout(main_layout)

        # Widget for options and interactions
        options_widget = QtWidgets.QWidget()
        options_layout = QtWidgets.QHBoxLayout()
        options_widget.setLayout(options_layout)

        region_combobox = QtWidgets.QComboBox()
        self.region_list = QtGui.QStandardItemModel()
        region_combobox.setModel(self.region_list)
        region_combobox.activated.connect(self.on_region_chosen)

        self.plot_combobox = QtWidgets.QComboBox()
        self.plot_list = QtGui.QStandardItemModel()
        self.plot_combobox.setModel(self.plot_list)
        self.plot_combobox.activated.connect(self.on_plot_chosen)

        spacer = QtWidgets.QSpacerItem(10, 20, QtWidgets.QSizePolicy.Expanding)

        # TODO
        page_combobox = QtWidgets.QComboBox()
        page_list = QtGui.QStandardItemModel()
        page_combobox.setModel(page_list)


        next_button = QtWidgets.QPushButton('Next')
        next_button.clicked.connect(self.on_next_pressed)
        prev_button = QtWidgets.QPushButton('Previous')
        prev_button.clicked.connect(self.on_prev_pressed)

        self.page_label = QtWidgets.QLabel('1/1')

        options_layout.addWidget(region_combobox)
        options_layout.addWidget(self.plot_combobox)
        options_layout.addItem(spacer)
        options_layout.addWidget(prev_button)
        options_layout.addWidget(next_button)
        options_layout.addWidget(self.page_label)

        options_layout.setStretch(0, 4)
        options_layout.setStretch(1, 4)
        options_layout.setStretch(2, 6)
        options_layout.setStretch(5, 2)

        # Widget that contains all the plots
        plot_widget = QtWidgets.QWidget()
        plot_layout = QtWidgets.QHBoxLayout()
        plot_widget.setLayout(plot_layout)

        self.plots_hist = []
        self.plots_feat = []
        self.plots_cbar = []

        for i in range(self.step):
            fig_area = pg.GraphicsLayoutWidget()
            fig_layout = pg.GraphicsLayout()
            fig_area.addItem(fig_layout)

            fig_hist = pg.PlotItem()
            fig_hist.setContentsMargins(0, 0, 0, 0)
            fig_hist.setMouseEnabled(x=False)
            self.set_axis(fig_hist, 'bottom', pen=None, label='')
            replace_axis(fig_hist)
            ax_hist = self.set_axis(fig_hist, 'left', pen='k')
            ax_hist.setWidth(0)
            ax_hist.setStyle(tickTextOffset=-30)
            self.plots_hist.append(fig_hist)

            fig_feat = pg.PlotItem()
            fig_feat.setYLink(fig_hist)
            fig_feat.setMouseEnabled(x=False)

            self.set_axis(fig_feat, 'bottom', pen='w')
            ax_feat = self.set_axis(fig_feat, 'left', show=False)
            self.plots_feat.append(fig_feat)

            fig_cbar = pg.PlotItem()
            fig_cbar.setMouseEnabled(x=False, y=False)
            fig_cbar.setMaximumHeight(70)
            self.set_axis(fig_cbar, 'bottom', show=False)
            self.set_axis(fig_cbar, 'left', show=False)
            self.set_axis(fig_cbar, 'top', pen='w')
            self.plots_cbar.append(fig_cbar)

            fig_layout.addItem(fig_cbar, 0, 0, 1, 2)
            fig_layout.addItem(fig_hist, 1, 0, 1, 1)
            fig_layout.addItem(fig_feat, 1, 1, 1, 1)
            fig_layout.layout.setRowStretchFactor(0, 1)
            fig_layout.layout.setRowStretchFactor(1, 10)
            plot_layout.addWidget(fig_area)

        # Widget that contains info about the insertion displayed
        info_widget = QtWidgets.QWidget()
        info_layout = QtWidgets.QHBoxLayout()
        info_widget.setLayout(info_layout)
        self.info_labels = []
        for i in range(self.step):
            label = QtWidgets.QTextEdit()
            label.setReadOnly(True)
            label.setLineWrapMode(QtWidgets.QTextEdit.WidgetWidth)
            # label = QtWidgets.QLabel()
            # label.setWordWrap(True)
            info_layout.addWidget(label)
            self.info_labels.append(label)

        # Widget for updating colorbar or axis of plots
        scale_widget = QtWidgets.QWidget()
        scale_layout = QtWidgets.QGridLayout()
        scale_widget.setLayout(scale_layout)

        spacer2 = QtWidgets.QSpacerItem(10, 20, QtWidgets.QSizePolicy.Expanding)

        self.align_button = QtWidgets.QCheckBox('Align Plots')
        self.align_button.setChecked(True)
        self.align_button.clicked.connect(self.on_align_plots)

        self.normalise_button = QtWidgets.QCheckBox('Normalise Plots')
        self.normalise_button.setChecked(False)
        self.normalise_button.clicked.connect(self.on_normalise_plots)

        self.slider_min = QtWidgets.QLabel('Min: 0')
        self.slider_max = QtWidgets.QLabel('Max: 100')
        self.slider_low = QtWidgets.QLabel('Low Val: 0')
        self.slider_high = QtWidgets.QLabel('High Val: 100')
        self.slider = QRangeSlider(QtCore.Qt.Horizontal)
        self.slider.sliderReleased.connect(self.on_slider_moved)

        scale_layout.addWidget(self.align_button, 0, 0, 1, 1)
        scale_layout.addItem(spacer2, 0, 1, 1, 3)
        scale_layout.addWidget(self.normalise_button, 1, 4, 1, 1)
        scale_layout.addWidget(self.slider_min, 0, 5, 1, 1)
        scale_layout.addWidget(self.slider, 1, 5, 1, 5)
        scale_layout.addWidget(self.slider_max, 0, 10, 1, 1)
        scale_layout.addWidget(self.slider_low,  2, 5, 1, 1)
        scale_layout.addWidget(self.slider_high, 2, 10, 1, 1)

        main_layout.addWidget(options_widget)
        main_layout.addWidget(plot_widget)
        main_layout.addWidget(info_widget)
        main_layout.addWidget(scale_widget)
        main_layout.setStretch(0, 1)
        main_layout.setStretch(1, 10)
        main_layout.setStretch(2, 1)
        main_layout.setStretch(3, 1)


    def set_axis(self, fig, ax, show=True, label=None, pen='k', ticks=True):

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

    def on_region_chosen(self, idx):
        self.chosen_id = self.region_ids[idx]
        data = self.data[self.data['atlas_id'] == self.chosen_id]

        pids = data.groupby('pid').atlas_id.count().sort_values()[::-1]
        self.pids = pids.index.values

        self.page_num = np.ceil(self.pids.size / self.step) - 1
        self.max_idx = self.pids.size
        self.page_idx = 0
        self.update_page_label()

        self.region_data, self.probe_info = self.get_region_data(self.pids)
        self.offset_data = self.get_offset_data()
        self.get_feature_data(self.plot_name, self.plot_type)

        self.plot_all()

    def on_plot_chosen(self, idx):
        item = self.plot_list.item(idx)
        self.plot_name = item.text()
        self.plot_type = PLOT_TYPES[self.plot_name]['plot_type']

        self.get_feature_data(self.plot_name, self.plot_type)

        self.plot_all()

    def on_normalise_plots(self):
        self.normalise = self.normalise_button.isChecked()
        self.slider.allowMove(self.normalise)
        self.plot_features()

    def on_align_plots(self):
        self.plot_all()

    def on_slider_moved(self):
        if self.normalise:
            self.levels[0] = self.slider.low()
            self.levels[1] = self.slider.high()
            self.slider_low.setText(str(int(self.levels[0])))
            self.slider_high.setText(str(int(self.levels[1])))
            self.plot_features()

    def on_next_pressed(self):
        if self.page_idx < self.page_num:
            self.page_idx += 1
            self.update_page_label()
            self.plot_all()

    def on_prev_pressed(self):
        if self.page_idx > 0:
            self.page_idx -= 1
            self.update_page_label()
            self.plot_all()

    def init_slider(self):

        min_val = int(self.max_levels[0])
        max_val = int(self.max_levels[1])
        self.slider.setMinimum(min_val)
        self.slider_min.setText(f'Min: {str(min_val)}')
        self.slider.setMaximum(max_val)
        self.slider_max.setText(f'Max: {str(max_val)}')
        self.slider.setLow(min_val)
        self.slider_low.setText(f'Low Val: {str(min_val)}')
        self.slider.setHigh(max_val)
        self.slider_high.setText(f'High Val: {str(max_val)}')

    def update_page_label(self):
        self.page_label.setText(f'{int(self.page_idx) + 1}/{int(self.page_num) + 1}')

    def plot_all(self):

        self.clear_plots()
        self.plot_features()
        self.plot_regions()
        self.set_info()

    def clear_plots(self):
        for fig_hist, fig_feat, fig_cbar in zip(self.plots_hist, self.plots_feat, self.plots_cbar):
            fig_hist.clear()
            fig_feat.clear()
            fig_cbar.clear()
            self.set_axis(fig_cbar, 'top', pen='w')

    def get_idx(self):
        if self.page_idx == self.page_num:
            pid_idx = np.arange(self.page_idx * self.step, self.max_idx)
        else:
            pid_idx = np.arange(self.page_idx * self.step, (self.page_idx * self.step) + self.step)

        return pid_idx

    def set_info(self):
        pid_idx = self.get_idx()
        data = [dat for i, dat in enumerate(self.probe_info) if i in pid_idx]

        for iD, dat in enumerate(data):
            lab = self.info_labels[iD]
            lab.setText(f'{dat["session"]}\n'
                        f'{dat["probe"]}\n'
                        f'{dat["pid"]}')

    def plot_regions(self):

        pid_idx = self.get_idx()

        data = [dat for i, dat in enumerate(self.region_data) if i in pid_idx]
        offsets = [off for i, off in enumerate(self.offset_data) if i in pid_idx]

        for iD, dat in enumerate(data):
            fig = self.plots_hist[iD]
            fig.clear()
            axis = fig.getAxis('left')
            axis.setTicks([])

            if self.align_button.isChecked():
                offset = offsets[iD]
            else:
                offset = 0
            labels = copy.copy(dat['labels'])
            labels[:, 0] = labels[:, 0] - offset

            axis = fig.getAxis('left')
            axis.setTicks([labels])
            axis.setZValue(10)
            axis.setPen('k')

            # Plot each histology region
            for ir, (reg, col, reg_id) in enumerate(zip(dat['regions'], dat['colors'], dat['region_id'])):

                colour = QtGui.QColor(*col)
                if reg_id == self.chosen_id:
                    colour.setAlpha(255)
                else:
                    colour.setAlpha(60)
                region = pg.LinearRegionItem(values=(reg[0] - offset, reg[1] - offset),
                                             orientation=pg.LinearRegionItem.Horizontal,
                                             brush=colour, movable=False)
                # Add a white line at the boundary between regions
                bound = pg.InfiniteLine(pos=reg[0] - offset, angle=0, pen='w')
                fig.addItem(region)
                fig.addItem(bound)

            if self.align_button.isChecked():
                fig.setYRange(min=-1000, max=1000)
            else:
                fig.setYRange(min=0, max=3840)

        if iD < self.step - 1:
            for fig in self.plots_hist[iD + 1:]:
                axis = fig.getAxis('left')
                axis.setTicks([])
                axis.setPen(None)

    def plot_features(self):

        plot_data = self.feature_data[self.plot_name]

        pid_idx = self.get_idx()

        data = [dat for i, dat in enumerate(plot_data['data']) if i in pid_idx]
        offsets = [off for i, off in enumerate(self.offset_data) if i in pid_idx]

        for iD, dat in enumerate(data):
            if self.align_button.isChecked():
                offset = offsets[iD]
            else:
                offset = 0

            if self.plot_type == 'probe':
                fig_probe = self.plots_feat[iD]
                fig_cbar = self.plots_cbar[iD]
                self.plot_probe(fig_probe, fig_cbar, dat, offset)
            elif self.plot_type == 'line':
                fig_line = self.plots_feat[iD]
                fig_cbar = self.plots_cbar[iD]
                self.plot_line(fig_line, fig_cbar, dat, offset)

        if iD < self.step - 1:
            for fig in self.plots_feat[iD + 1:]:
                self.set_axis(fig, 'bottom', pen=None)


    def plot_line(self, fig_line, fig_cbar, data, offset):

        fig_line.clear()
        fig_cbar.clear()
        line = pg.PlotCurveItem()
        line.setData(x=data['x'], y=data['y'] - offset)
        line.setPen(self.kpen_solid)
        fig_line.addItem(line)
        levels = self.levels if self.normalise else data['xrange']
        self.set_axis(fig_line, 'bottom', pen='k', label=data['xaxis'])
        fig_line.setXRange(min=levels[0], max=levels[1], padding=0)

        if self.align_button.isChecked():
            fig_line.setYRange(min=-1000, max=1000)
        else:
            fig_line.setYRange(min=0, max=3840)

        ax = fig_cbar.getAxis('top')
        ax.setPen(None)
        ax.setTextPen('k')
        ax.setStyle(stopAxisAtTick=((False, False)))
        ax.setTicks([])
        ax.setLabel(data['title'])
        ax.setHeight(30)

    def plot_probe(self, fig_probe, fig_cbar, data, offset):

        fig_probe.clear()
        fig_cbar.clear()
        color_bar = cb.ColorBar(data['cmap'])
        lut = color_bar.getColourMap()
        levels = self.levels if self.normalise else data['levels']
        for img, scale, move in zip(data['img'], data['scale'], data['offset']):
            image = pg.ImageItem()
            image.setImage(img)
            transform = [scale[0], 0., 0., 0., scale[1], 0., move[0],
                         move[1] - offset, 1.]
            image.setTransform(QtGui.QTransform(*transform))
            image.setLookupTable(lut)
            image.setLevels((levels[0], levels[1]))
            fig_probe.addItem(image)

        fig_probe.setXRange(min=data['xrange'][0], max=data['xrange'][1], padding=0)
        if self.align_button.isChecked():
            fig_probe.setYRange(min=-1000, max=1000)
        else:
            fig_probe.setYRange(min=0, max=3840)

        cbar = color_bar.makeColourBar(20, 5, fig_cbar, min=levels[0],
                                       max=levels[1], label=data['title'], lim=True)
        fig_cbar.addItem(cbar)
        self.set_axis(fig_probe, 'bottom', pen=None, label='')

    def get_feature_data(self, plot_name, plot_type):

        if plot_type == 'probe':
            data, levels = self.get_probe_data(plot_name, self.pids)
        elif plot_type == 'line':
            data, levels = self.get_line_data(plot_name, self.pids)
        self.feature_data[plot_name] = {'data': data, 'levels': levels}

        self.max_levels = self.feature_data[plot_name]['levels']
        self.init_slider()
        self.levels = np.copy(self.max_levels)

    def get_line_data(self, plot_name, pids):
        all_data = []
        for ip, pid in enumerate(pids):

            df = self.data[self.data['pid'] == pid]
            data = df[plot_name].values
            depths = df['depths'].values

            if ip == 0:
                max_level = np.nanmax(data)
                min_level = np.nanmin(data)
            else:
                max_level = np.nanmax(np.r_[np.nanmax(data), max_level])
                min_level = np.nanmin(np.r_[np.nanmin(data), min_level])

            data_dict = {
                'x': data,
                'y': depths,
                'xrange': np.array([0, np.nanmax(data)]),
                'xaxis': PLOT_TYPES[plot_name]['xlabel'],
                'title': df.iloc[0].histology.upper()
            }
            all_data.append(data_dict)

        return all_data, [min_level, max_level]

    def get_probe_data(self, plot_name, pids):
        all_data = []
        for ip, pid in enumerate(pids):

            df = self.data[self.data['pid'] == pid]
            data = df[plot_name].values
            chn_coords = np.c_[df['lateral_um'].values, df['axial_um'].values]

            BNK_SIZE = 10
            N_BNK = 4
            probe_img, probe_scale, probe_offset = self.arrange_channels2banks(data, chn_coords)
            probe_levels = np.nanquantile(data, [0.1, 0.9])
            if ip == 0:
                max_level = probe_levels[1]
                min_level = probe_levels[0]
            else:
                max_level = np.nanmax(np.r_[probe_levels[1], max_level])
                min_level = np.nanmin(np.r_[probe_levels[0], min_level])

            data_dict = {
                'img': probe_img,
                'scale': probe_scale,
                'offset': probe_offset,
                'levels': probe_levels,
                'cmap': PLOT_TYPES[plot_name]['cmap'],
                'xrange': np.array([0 * BNK_SIZE, (N_BNK) * BNK_SIZE]),
                'title': df.iloc[0].histology.upper()
            }

            all_data.append(data_dict)

        return all_data, [min_level, max_level]

    def get_region_data(self, pids):

        region_data = []
        probe_info = []

        for pid in pids:

            df = self.data[self.data['pid'] == pid]
            mlapdv = np.c_[df['x'].values, df['y'].values, df['z'].values]

            region, region_label, region_colour, region_id = \
                EphysAlignment.get_histology_regions(mlapdv, df['axial_um'].values, brain_atlas=self.ba)

            data = {'regions': region,
                    'labels': region_label,
                    'colors': region_colour,
                    'region_id': region_id
                    }

            d = df.iloc[0]
            info = {'pid': pid,
                    'eid': d.eid,
                    'session': '/'.join(self.one.eid2path(d.eid).parts[-3:]),
                    'probe': d.pname,
                    'histology': d.histology}

            region_data.append(data)
            probe_info.append(info)

        return region_data, probe_info

    def get_offset_data(self):
        offset_data = []
        for data in self.region_data:
            idx_reg = np.where(data['region_id'] == self.chosen_id)[0]

            for i, idx in enumerate(idx_reg):
                if i == 0:
                    regs = data['regions'][idx]
                else:
                    regs = np.r_[regs, data['regions'][idx]]

            offset = np.min(regs) + (np.max(regs) - np.min(regs)) / 2

            offset_data.append(offset)

        return offset_data

    def arrange_channels2banks(self, data, chn_coords):

        bnk_data = []
        N_BNK = len(np.unique(chn_coords[:, 0]))
        BNK_SIZE = 10
        bnk_scale = np.empty((N_BNK, 2))
        bnk_offset = np.empty((N_BNK, 2))
        chn_max = np.max(chn_coords[:, 1])
        chn_min = np.min(chn_coords[:, 1])
        for iX, x in enumerate(np.unique(chn_coords[:, 0])):
            bnk_idx = np.where(chn_coords[:, 0] == x)[0]

            bnk_ycoords = chn_coords[bnk_idx, 1]
            bnk_diff = np.min(np.diff(bnk_ycoords))

            # NP1.0 checkerboard
            bnk_full = np.arange(np.min(bnk_ycoords), np.max(bnk_ycoords) + bnk_diff, bnk_diff)
            _bnk_vals = np.full((bnk_full.shape[0]), np.nan)
            idx_full = np.where(np.isin(bnk_full, bnk_ycoords))
            _bnk_vals[idx_full] = data[bnk_idx]

            # Detect where the nans are, whether it is odd or even
            _bnk_data = _bnk_vals[np.newaxis, :]
            _bnk_yscale = ((chn_max -
                            chn_min) / _bnk_data.shape[1])
            _bnk_xscale = BNK_SIZE / _bnk_data.shape[0]
            _bnk_yoffset = np.min(bnk_ycoords)
            _bnk_xoffset = BNK_SIZE * iX


            bnk_data.append(_bnk_data)
            bnk_scale[iX, :] = np.array([_bnk_xscale, _bnk_yscale])
            bnk_offset[iX, :] = np.array([_bnk_xoffset, _bnk_yoffset])

        return bnk_data, bnk_scale, bnk_offset


if __name__ == '__main__':
    one = ONE()
    app = QtWidgets.QApplication(sys.argv)
    window = RegionFeatureWindow(one)
    window.show()
    app.exec_()
