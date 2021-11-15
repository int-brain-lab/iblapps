from matplotlib import cm
from pathlib import Path
import numpy as np
import one.alf as alf
from brainbox.processing import bincount2D
from brainbox.population.decode import xcorr
from brainbox.task import passive
import scipy
from PyQt5 import QtGui

BNK_SIZE = 10
AUTOCORR_BIN_SIZE = 0.25 / 1000
AUTOCORR_WIN_SIZE = 10 / 1000

FS = 30000
np.seterr(divide='ignore', invalid='ignore')


class PlotData:
    def __init__(self, probe_path, ephys_path, alf_path, shank_idx):

        self.probe_path = probe_path
        self.ephys_path = ephys_path
        self.alf_path = alf_path

        channels = alf.io.load_object(self.probe_path, 'channels')
        self.chn_coords_all = channels['localCoordinates']
        self.chn_ind_all = channels['rawInd'].astype(int)

        self.chn_min = np.min(self.chn_coords_all[:, 1])
        self.chn_max = np.max(self.chn_coords_all[:, 1])
        self.chn_diff = np.min(np.abs(np.diff(np.unique(self.chn_coords_all[:, 1]))))

        self.chn_full = np.arange(self.chn_min, self.chn_max + self.chn_diff, self.chn_diff)

        chn_x = np.unique(self.chn_coords_all[:, 0])
        chn_x_diff = np.diff(chn_x)
        n_shanks = np.sum(chn_x_diff > 100) + 1

        if n_shanks > 1:
            shanks = {}
            for iShank in range(n_shanks):
                shanks[iShank] = [chn_x[iShank * 2], chn_x[(iShank * 2) + 1]]

            shank_chns = np.bitwise_and(self.chn_coords_all[:, 0] >= shanks[shank_idx][0],
                                        self.chn_coords_all[:, 0] <= shanks[shank_idx][1])
            self.chn_coords = self.chn_coords_all[shank_chns, :]
            self.chn_ind = self.chn_ind_all[shank_chns]
        else:
            self.chn_coords = self.chn_coords_all
            self.chn_ind = self.chn_ind_all

        self.N_BNK = len(np.unique(self.chn_coords[:, 0]))
        self.idx_full = np.where(np.isin(self.chn_full, self.chn_coords[:, 1]))[0]

        # See if spike data is available
        try:
            self.spikes = alf.io.load_object(self.probe_path, 'spikes')
            self.spike_data_status = True
        except Exception:
            print('spike data was not found, some plots will not display')
            self.spike_data_status = False

        try:
            self.clusters = alf.io.load_object(self.probe_path, 'clusters')
            shank_spikes = np.isin(self.chn_ind_all[self.clusters.channels[self.spikes.clusters]],
                                   self.chn_ind)
            for key in self.spikes.keys():
                self.spikes[key] = self.spikes[key][shank_spikes]

            self.filter_units('all')
            self.cluster_data_status = True
            self.compute_timescales()

        except Exception as err:
            print(err)
            print('cluster data was not found, some plots will not display')
            self.cluster_data_status = False

        try:
            lfp_spectrum = alf.io.load_object(self.ephys_path, 'ephysSpectralDensityLF',
                                              namespace='iblqc')
            if len(lfp_spectrum) == 2:
                self.lfp_freq = lfp_spectrum.get('freqs')
                self.lfp_power = lfp_spectrum.get('power', [])
                if not np.any(self.lfp_power):
                    self.lfp_power = lfp_spectrum.get('amps')
                self.lfp_data_status = True
            else:
                print('lfp data was not found, some plots will not display')
                self.lfp_data_status = False
        except Exception:
            print('lfp data was not found, some plots will not display')
            self.lfp_data_status = False

        try:
            rf_map_times = alf.io.load_object(self.alf_path, object='passiveRFM',
                                              namespace='ibl')
            # This needs to go into brainbox!!
            rf_map_frames_path = (self.alf_path.parent.
                                  joinpath('raw_passive_data', '_iblrig_RFMapStim.raw.bin'))
            rf_map_frames = np.fromfile(rf_map_frames_path, dtype="uint8")
            y_pix, x_pix = 15, 15
            frames = np.transpose(np.reshape(rf_map_frames, [y_pix, x_pix, -1], order="F"),
                                  [2, 1, 0])

            self.rf_map = dict()
            self.rf_map['times'] = rf_map_times['times']
            self.rf_map['frames'] = frames
            if len(self.rf_map) == 2:
                self.rfmap_data_status = True
            else:
                print('rfmap data was not found, some plots will not display')
                self.rfmap_data_status = False
        except Exception:
            print('rfmp data was not found, some plots will not display')
            self.rfmap_data_status = False

        try:
            self.aud_stim = alf.io.load_object(self.alf_path, object='passiveStims',
                                               namespace='ibl')['table']
            if len(self.aud_stim) > 0:
                self.passive_data_status = True
        except Exception:
            print('passive stim data was not found, some plots will not display')
            self.passive_data_status = False

        try:
            gabor = alf.io.load_object(self.alf_path, object='passiveGabor',
                                       namespace='ibl')['table']
            self.vis_stim = dict()
            self.vis_stim['leftGabor'] = gabor['start'][(gabor['position'] == 35) &
                                                        (gabor['contrast'] > 0.1)]
            self.vis_stim['rightGabor'] = gabor['start'][(gabor['position'] == -35) &
                                                         (gabor['contrast'] > 0.1)]
            self.gabor_data_status = True
        except Exception:
            print('passive gabor data was not found, some plots will not display')
            self.gabor_data_status = False

    def filter_units(self, type):
        if type == 'all':
            self.spike_idx = np.arange(self.spikes['clusters'].size)

        elif type == 'KS good':
            clust = np.where(self.clusters.metrics.ks2_label == 'good')
            self.spike_idx = np.where(np.isin(self.spikes['clusters'], clust))[0]

        elif type == 'KS mua':
            clust = np.where(self.clusters.metrics.ks2_label == 'mua')
            self.spike_idx = np.where(np.isin(self.spikes['clusters'], clust))[0]

        elif type == 'IBL good':
            try:
                clust = np.where(self.clusters.metrics.label == 1)
                self.spike_idx = np.where(np.isin(self.spikes['clusters'], clust))[0]
            except Exception:
                print('IBL metrics not implemented will return ks good units instead')
                clust = np.where(self.clusters.metrics.ks2_label == 'good')
                self.spike_idx = np.where(np.isin(self.spikes['clusters'], clust))[0]

        # Filter for nans in depths and also in amps
        self.kp_idx = np.where(~np.isnan(self.spikes['depths'][self.spike_idx]) &
                               ~np.isnan(self.spikes['amps'][self.spike_idx]))[0]

# Plots that require spike and cluster data
    def get_depth_data_scatter(self):
        if not self.spike_data_status:
            data_scatter = None
            return data_scatter
        else:
            A_BIN = 10
            amp_range = np.quantile(self.spikes['amps'][self.spike_idx][self.kp_idx], [0, 0.9])
            amp_bins = np.linspace(amp_range[0], amp_range[1], A_BIN)
            colour_bin = np.linspace(0.0, 1.0, A_BIN + 1)
            colours = (cm.get_cmap('BuPu')(colour_bin)[np.newaxis, :, :3][0]) * 255
            spikes_colours = np.empty(self.spikes['amps'][self.spike_idx][self.kp_idx].size,
                                      dtype=object)
            spikes_size = np.empty(self.spikes['amps'][self.spike_idx][self.kp_idx].size)
            for iA in range(amp_bins.size):
                if iA == (amp_bins.size - 1):
                    idx = np.where((self.spikes['amps'][self.spike_idx][self.kp_idx] >
                                    amp_bins[iA]))[0]
                    # Make saturated spikes a very dark purple
                    spikes_colours[idx] = QtGui.QColor('#400080')
                else:
                    idx = np.where((self.spikes['amps'][self.spike_idx][self.kp_idx] >
                                    amp_bins[iA]) &
                                   (self.spikes['amps'][self.spike_idx][self.kp_idx] <=
                                    amp_bins[iA + 1]))[0]

                    spikes_colours[idx] = QtGui.QColor(*colours[iA])

                spikes_size[idx] = iA / (A_BIN / 4)

            data_scatter = {
                'x': self.spikes['times'][self.spike_idx][self.kp_idx][0:-1:100],
                'y': self.spikes['depths'][self.spike_idx][self.kp_idx][0:-1:100],
                'levels': amp_range * 1e6,
                'colours': spikes_colours[0:-1:100],
                'pen': None,
                'size': spikes_size[0:-1:100],
                'symbol': np.array('o'),
                'xrange': np.array([np.min(self.spikes['times'][self.spike_idx][self.kp_idx]
                                           [0:-1:100]),
                                    np.max(self.spikes['times'][self.spike_idx][self.kp_idx]
                                           [0:-1:100])]),
                'xaxis': 'Time (s)',
                'title': 'Amplitude (uV)',
                'cmap': 'BuPu',
                'cluster': False
            }

            return data_scatter

    def get_fr_p2t_data_scatter(self):
        if not self.spike_data_status:
            data_fr_scatter = None
            data_p2t_scatter = None
            data_amp_scatter = None
            return data_fr_scatter, data_p2t_scatter, data_amp_scatter
        else:
            (clu,
             spike_depths,
             spike_amps,
             n_spikes) = self.compute_spike_average((self.spikes['clusters'][self.spike_idx]
                                                    [self.kp_idx]), (self.spikes['depths']
                                                    [self.spike_idx][self.kp_idx]),
                                                    (self.spikes['amps'][self.spike_idx]
                                                    [self.kp_idx]))
            spike_amps = spike_amps * 1e6
            fr = n_spikes / np.max(self.spikes['times'])
            fr_levels = np.quantile(fr, [0, 1])

            data_fr_scatter = {
                'x': spike_amps,
                'y': spike_depths,
                'colours': fr,
                'pen': 'k',
                'size': np.array(8),
                'symbol': np.array('o'),
                'levels': fr_levels,
                'xrange': np.array([0.9 * np.min(spike_amps),
                                    1.1 * np.max(spike_amps)]),
                'xaxis': 'Amplitude (uV)',
                'title': 'Firing Rate (Sp/s)',
                'cmap': 'hot',
                'cluster': True
            }

            p2t = self.clusters['peakToTrough'][clu]

            # Define the p2t levels so always same colourbar across sessions
            p2t_levels = [-1.5, 1.5]
            data_p2t_scatter = {
                'x': spike_amps,
                'y': spike_depths,

                'colours': p2t,
                'pen': 'k',
                'size': np.array(8),
                'symbol': np.array('o'),
                'levels': p2t_levels,
                'xrange': np.array([0.9 * np.min(spike_amps),
                                    1.1 * np.max(spike_amps)]),
                'xaxis': 'Amplitude (uV)',
                'title': 'Peak to Trough duration (ms)',
                'cmap': 'RdYlGn',
                'cluster': True
            }

            spike_amps_levels = np.quantile(spike_amps, [0, 1])

            data_amp_scatter = {
                'x': fr,
                'y': spike_depths,

                'colours': spike_amps,
                'pen': 'k',
                'size': np.array(8),
                'symbol': np.array('o'),
                'levels': spike_amps_levels,
                'xrange': np.array([0.9 * np.min(fr),
                                    1.1 * np.max(fr)]),
                'xaxis': 'Firing Rate (Sp/s)',
                'title': 'Amplitude (uV)',
                'cmap': 'magma',
                'cluster': True
            }

            return data_fr_scatter, data_p2t_scatter, data_amp_scatter

    def get_fr_img(self):
        if not self.spike_data_status:
            data_img = None
            return data_img
        else:
            T_BIN = 0.05
            D_BIN = 5
            chn_min = np.min(np.r_[self.chn_min, self.spikes['depths'][self.spike_idx][self.kp_idx]])
            chn_max = np.max(np.r_[self.chn_max, self.spikes['depths'][self.spike_idx][self.kp_idx]])
            n, times, depths = bincount2D(self.spikes['times'][self.spike_idx][self.kp_idx],
                                          self.spikes['depths'][self.spike_idx][self.kp_idx],
                                          T_BIN, D_BIN, ylim=[chn_min, chn_max])
            img = n.T / T_BIN
            xscale = (times[-1] - times[0]) / img.shape[0]
            yscale = (depths[-1] - depths[0]) / img.shape[1]

            data_img = {
                'img': img,
                'scale': np.array([xscale, yscale]),
                'levels': np.quantile(np.mean(img, axis=0), [0, 1]),
                'offset': np.array([0, 0]),
                'xrange': np.array([times[0], times[-1]]),
                'xaxis': 'Time (s)',
                'cmap': 'binary',
                'title': 'Firing Rate'
            }

            return data_img

    def get_fr_amp_data_line(self):
        if not self.spike_data_status:
            data_fr_line = None
            data_amp_line = None
            return data_fr_line, data_amp_line
        else:
            T_BIN = np.max(self.spikes['times'])
            D_BIN = 10
            chn_min = np.min(np.r_[self.chn_min, self.spikes['depths'][self.spike_idx][self.kp_idx]])
            chn_max = np.max(np.r_[self.chn_max, self.spikes['depths'][self.spike_idx][self.kp_idx]])
            nspikes, times, depths = bincount2D(self.spikes['times'][self.spike_idx][self.kp_idx],
                                                self.spikes['depths'][self.spike_idx][self.kp_idx],
                                                T_BIN, D_BIN,
                                                ylim=[chn_min, chn_max])

            amp, times, depths = bincount2D(self.spikes['amps'][self.spike_idx][self.kp_idx],
                                            self.spikes['depths'][self.spike_idx][self.kp_idx],
                                            T_BIN, D_BIN, ylim=[chn_min, chn_max],
                                            weights=self.spikes['amps'][self.spike_idx]
                                            [self.kp_idx])
            mean_fr = nspikes[:, 0] / T_BIN
            mean_amp = np.divide(amp[:, 0], nspikes[:, 0]) * 1e6
            mean_amp[np.isnan(mean_amp)] = 0
            remove_bins = np.where(nspikes[:, 0] < 50)[0]
            mean_amp[remove_bins] = 0

            data_fr_line = {
                'x': mean_fr,
                'y': depths,
                'xrange': np.array([0, np.max(mean_fr)]),
                'xaxis': 'Firing Rate (Sp/s)'
            }

            data_amp_line = {
                'x': mean_amp,
                'y': depths,
                'xrange': np.array([0, np.max(mean_amp)]),

                'xaxis': 'Amplitude (uV)'
            }

            return data_fr_line, data_amp_line

    def get_correlation_data_img(self):
        if not self.spike_data_status:
            data_img = None
            return data_img
        else:
            T_BIN = 0.05
            D_BIN = 40
            chn_min = np.min(np.r_[self.chn_min, self.spikes['depths'][self.spike_idx][self.kp_idx]])
            chn_max = np.max(np.r_[self.chn_max, self.spikes['depths'][self.spike_idx][self.kp_idx]])
            R, times, depths = bincount2D(self.spikes['times'][self.spike_idx][self.kp_idx],
                                          self.spikes['depths'][self.spike_idx][self.kp_idx],
                                          T_BIN, D_BIN, ylim=[chn_min, chn_max])
            corr = np.corrcoef(R)
            corr[np.isnan(corr)] = 0
            scale = (np.max(depths) - np.min(depths)) / corr.shape[0]
            data_img = {
                'img': corr,
                'scale': np.array([scale, scale]),
                'levels': np.array([np.min(corr), np.max(corr)]),
                'offset': np.array([0, 0]),
                'xrange': np.array([self.chn_min, self.chn_max]),
                'cmap': 'viridis',
                'title': 'Correlation',
                'xaxis': 'Distance from probe tip (um)'
            }
            return data_img

    def get_rms_data_img_probe(self, format):
        # Finds channels that are at equivalent depth on probe and averages rms values for each
        # time point at same depth togehter
        try:
            rms_amps = alf.io.load_file_content(Path(self.ephys_path, '_iblqc_ephysTimeRms' +
                                                     format + '.rms.npy'))
        except Exception:
            try:
                rms_amps = alf.io.load_file_content(Path(self.ephys_path, '_iblqc_ephysTimeRms' +
                                                         format + '.amps.npy'))
            except Exception:
                print('rms data was not found, some plots will not display')
                data_img = None
                data_probe = None
                return data_img, data_probe

        try:
            rms_times = alf.io.load_file_content(Path(self.ephys_path, '_iblqc_ephysTimeRms' +
                                                      format + '.timestamps.npy'))
            xaxis = 'Time (s)'
        except Exception:
            rms_times = np.array([0, rms_amps.shape[0]])
            xaxis = 'Time samples'

        _rms = np.take(rms_amps, self.chn_ind, axis=1)
        _, self.chn_depth, chn_count = np.unique(self.chn_coords[:, 1], return_index=True,
                                                 return_counts=True)
        self.chn_depth_eq = np.copy(self.chn_depth)
        self.chn_depth_eq[np.where(chn_count == 2)] += 1

        def avg_chn_depth(a):
            return(np.mean([a[self.chn_depth], a[self.chn_depth_eq]], axis=0))

        def get_median(a):
            return(np.median(a))

        def median_subtract(a):
            return(a - np.median(a))
        img = np.apply_along_axis(avg_chn_depth, 1, _rms * 1e6)
        median = np.mean(np.apply_along_axis(get_median, 1, img))
        # Medium subtract to remove bands, but add back average median so values make sense
        img = np.apply_along_axis(median_subtract, 1, img) + median

        img_full = np.full((img.shape[0], self.chn_full.shape[0]), np.nan)
        img_full[:, self.idx_full] = img

        levels = np.quantile(img, [0.1, 0.9])
        xscale = (rms_times[-1] - rms_times[0]) / img_full.shape[0]
        yscale = (self.chn_max - self.chn_min) / img_full.shape[1]

        if format == 'AP':
            cmap = 'plasma'
        else:
            cmap = 'inferno'

        data_img = {
            'img': img_full,
            'scale': np.array([xscale, yscale]),
            'levels': levels,
            'offset': np.array([0, 0]),
            'cmap': cmap,
            'xrange': np.array([rms_times[0], rms_times[-1]]),
            'xaxis': xaxis,
            'title': format + ' RMS (uV)'
        }

        # Probe data
        rms_avg = (np.mean(rms_amps, axis=0)[self.chn_ind]) * 1e6
        probe_levels = np.quantile(rms_avg, [0.1, 0.9])
        probe_img, probe_scale, probe_offset = self.arrange_channels2banks(rms_avg)

        data_probe = {
            'img': probe_img,
            'scale': probe_scale,
            'offset': probe_offset,
            'levels': probe_levels,
            'cmap': cmap,
            'xrange': np.array([0 * BNK_SIZE, (self.N_BNK) * BNK_SIZE]),
            'title': format + ' RMS (uV)'
        }

        return data_img, data_probe

    def get_lfp_spectrum_data(self):
        freq_bands = np.vstack(([0, 4], [4, 10], [10, 30], [30, 80], [80, 200]))
        data_probe = {}
        if not self.lfp_data_status:
            data_img = None
            for freq in freq_bands:
                lfp_band_data = {f"{freq[0]} - {freq[1]} Hz": None}
                data_probe.update(lfp_band_data)

            return data_img, data_probe
        else:
            # Power spectrum image
            freq_range = [0, 300]
            freq_idx = np.where((self.lfp_freq >= freq_range[0]) &
                                (self.lfp_freq < freq_range[1]))[0]
            _lfp = np.take(self.lfp_power[freq_idx], self.chn_ind, axis=1)
            _lfp_dB = 10 * np.log10(_lfp)
            _, self.chn_depth, chn_count = np.unique(self.chn_coords[:, 1], return_index=True,
                                                     return_counts=True)
            self.chn_depth_eq = np.copy(self.chn_depth)
            self.chn_depth_eq[np.where(chn_count == 2)] += 1

            def avg_chn_depth(a):
                return(np.mean([a[self.chn_depth], a[self.chn_depth_eq]], axis=0))

            img = np.apply_along_axis(avg_chn_depth, 1, _lfp_dB)
            img_full = np.full((img.shape[0], self.chn_full.shape[0]), np.nan)
            img_full[:, self.idx_full] = img

            levels = np.quantile(img, [0.1, 0.9])
            xscale = (freq_range[-1] - freq_range[0]) / img_full.shape[0]
            yscale = (self.chn_max - self.chn_min) / img_full.shape[1]

            data_img = {
                'img': img_full,
                'scale': np.array([xscale, yscale]),
                'levels': levels,
                'offset': np.array([0, 0]),
                'cmap': 'viridis',
                'xrange': np.array([freq_range[0], freq_range[-1]]),
                'xaxis': 'Frequency (Hz)',
                'title': 'PSD (dB)'
            }

            # Power spectrum in bands on probe
            for freq in freq_bands:
                freq_idx = np.where((self.lfp_freq >= freq[0]) & (self.lfp_freq < freq[1]))[0]
                lfp_avg = np.mean(self.lfp_power[freq_idx], axis=0)[self.chn_ind]
                lfp_avg_dB = 10 * np.log10(lfp_avg)
                probe_img, probe_scale, probe_offset = self.arrange_channels2banks(lfp_avg_dB)
                probe_levels = np.quantile(lfp_avg_dB, [0.1, 0.9])

                lfp_band_data = {f"{freq[0]} - {freq[1]} Hz": {
                    'img': probe_img,
                    'scale': probe_scale,
                    'offset': probe_offset,
                    'levels': probe_levels,
                    'cmap': 'viridis',
                    'xaxis': 'Time (s)',
                    'xrange': np.array([0 * BNK_SIZE, (self.N_BNK) * BNK_SIZE]),
                    'title': f"{freq[0]} - {freq[1]} Hz (dB)"}
                }
                data_probe.update(lfp_band_data)

            return data_img, data_probe

    def get_rfmap_data(self):
        data_img = dict()
        if not self.rfmap_data_status:
            return data_img, None
        else:

            (rf_map_times, rf_map_pos,
             rf_stim_frames) = passive.get_on_off_times_and_positions(self.rf_map)

            rf_map, _ = \
                passive.get_rf_map_over_depth(rf_map_times, rf_map_pos, rf_stim_frames,
                                              self.spikes['times'][self.spike_idx][self.kp_idx],
                                              self.spikes['depths'][self.spike_idx][self.kp_idx],
                                              d_bin=160)
            rfs_svd = passive.get_svd_map(rf_map)
            img = dict()
            img['on'] = np.vstack(rfs_svd['on'])
            img['off'] = np.vstack(rfs_svd['off'])
            yscale = ((np.max(self.chn_coords[:, 1]) - np.min(
                self.chn_coords[:, 1])) / img['on'].shape[0])
            xscale = 1
            levels = np.quantile(np.c_[img['on'], img['off']], [0, 1])

            depths = np.linspace(0, 3840, len(rfs_svd['on']) + 1)

            sub_type = ['on', 'off']
            for sub in sub_type:
                sub_data = {sub: {
                    'img': [img[sub].T],
                    'scale': [np.array([xscale, yscale])],
                    'levels': levels,
                    'offset': [np.array([0, 0])],
                    'cmap': 'viridis',
                    'xrange': np.array([0, 15]),
                    'xaxis': 'Position',
                    'title': 'rfmap (dB)'}
                }
                data_img.update(sub_data)

            return data_img, depths

    def get_passive_events(self):
        stim_keys = ['valveOn', 'toneOn', 'noiseOn', 'leftGabor', 'rightGabor']
        data_img = dict()
        if not self.passive_data_status and not self.gabor_data_status:
            return data_img
        elif not self.passive_data_status and self.gabor_data_status:
            stim_types = ['leftGabor', 'rightGabor']
            stims = self.vis_stim
        elif self.passive_data_status and not self.gabor_data_status:
            stim_types = ['valveOn', 'toneOn', 'noiseOn']
            stims = {stim_type: self.aud_stim[stim_type] for stim_type in stim_types}
        else:
            stim_types = stim_keys
            stims = {stim_type: self.aud_stim[stim_type] for stim_type in stim_types[0:3]}
            stims.update(self.vis_stim)

        base_stim = 1
        pre_stim = 0.4
        post_stim = 1
        stim_events = passive.get_stim_aligned_activity(stims, self.spikes['times'][self.spike_idx]
                                                        [self.kp_idx], self.spikes['depths']
                                                        [self.spike_idx][self.kp_idx],
                                                        pre_stim=pre_stim, post_stim=post_stim,
                                                        base_stim=base_stim)

        for stim_type, z_score in stim_events.items():
            xscale = (post_stim + pre_stim) / z_score.shape[1]
            yscale = ((np.max(self.chn_coords[:, 1]) - np.min(
                self.chn_coords[:, 1])) / z_score.shape[0])

            levels = [-10, 10]

            stim_data = {stim_type: {
                'img': z_score.T,
                'scale': np.array([xscale, yscale]),
                'levels': levels,
                'offset': np.array([-1 * pre_stim, 0]),
                'cmap': 'bwr',
                'xrange': [-1 * pre_stim, post_stim],
                'xaxis': 'Time from Stim Onset (s)',
                'title': 'Firing rate (z score)'}
            }
            data_img.update(stim_data)

        return data_img

    def get_autocorr(self, clust_idx):
        idx = np.where(self.spikes['clusters'] == self.clust_id[clust_idx])[0]
        autocorr = xcorr(self.spikes['times'][idx], self.spikes['clusters'][idx],
                         AUTOCORR_BIN_SIZE, AUTOCORR_WIN_SIZE)

        return autocorr[0, 0, :], self.clust_id[clust_idx]

    def get_template_wf(self, clust_idx):
        template_wf = (self.clusters['waveforms'][self.clust_id[clust_idx], :, 0])
        return template_wf * 1e6

    def arrange_channels2banks(self, data):
        bnk_data = []
        bnk_scale = np.empty((self.N_BNK, 2))
        bnk_offset = np.empty((self.N_BNK, 2))
        for iX, x in enumerate(np.unique(self.chn_coords[:, 0])):
            bnk_idx = np.where(self.chn_coords[:, 0] == x)[0]

            bnk_ycoords = self.chn_coords[bnk_idx, 1]
            bnk_diff = np.min(np.diff(bnk_ycoords))

            # NP1.0 checkerboard
            if bnk_diff != self.chn_diff:
                bnk_full = np.arange(np.min(bnk_ycoords), np.max(bnk_ycoords) + bnk_diff, bnk_diff)
                _bnk_vals = np.full((bnk_full.shape[0]), np.nan)
                idx_full = np.where(np.isin(bnk_full, bnk_ycoords))
                _bnk_vals[idx_full] = data[bnk_idx]

                # Detect where the nans are, whether it is odd or even
                _bnk_data = _bnk_vals[np.newaxis, :]

                _bnk_yscale = ((self.chn_max -
                                self.chn_min) / _bnk_data.shape[1])
                _bnk_xscale = BNK_SIZE / _bnk_data.shape[0]

                _bnk_yoffset = np.min(bnk_ycoords)
                _bnk_xoffset = BNK_SIZE * iX

            else:  # NP2.0
                _bnk_vals = np.full((self.chn_full.shape[0]), np.nan)
                idx_full = np.where(np.isin(self.chn_full, bnk_ycoords))
                _bnk_vals[idx_full] = data[bnk_idx]

                _bnk_data = _bnk_vals[np.newaxis, :]

                _bnk_yscale = ((self.chn_max -
                                self.chn_min) / _bnk_data.shape[1])
                _bnk_xscale = BNK_SIZE / _bnk_data.shape[0]
                _bnk_yoffset = 0
                _bnk_xoffset = BNK_SIZE * iX

            bnk_data.append(_bnk_data)
            bnk_scale[iX, :] = np.array([_bnk_xscale, _bnk_yscale])
            bnk_offset[iX, :] = np.array([_bnk_xoffset, _bnk_yoffset])

        return bnk_data, bnk_scale, bnk_offset

    def compute_spike_average(self, spike_clusters, spike_depth, spike_amp):
        clust, inverse, counts = np.unique(spike_clusters, return_inverse=True, return_counts=True)
        _spike_depth = scipy.sparse.csr_matrix((spike_depth, (inverse,
                                                np.zeros(inverse.size, dtype=int))))
        _spike_amp = scipy.sparse.csr_matrix((spike_amp, (inverse,
                                              np.zeros(inverse.size, dtype=int))))
        spike_depth_avg = np.ravel(_spike_depth.toarray()) / counts
        spike_amp_avg = np.ravel(_spike_amp.toarray()) / counts
        self.clust_id = clust
        return clust, spike_depth_avg, spike_amp_avg, counts

    def compute_timescales(self):
        self.t_autocorr = 1e3 * np.arange((AUTOCORR_WIN_SIZE / 2) - AUTOCORR_WIN_SIZE,
                                          (AUTOCORR_WIN_SIZE / 2) + AUTOCORR_BIN_SIZE,
                                          AUTOCORR_BIN_SIZE)
        n_template = self.clusters['waveforms'][0, :, 0].size
        self.t_template = 1e3 * (np.arange(n_template)) / FS

    def normalise_data(self, data, lquant=0, uquant=1):
        levels = np.quantile(data, [lquant, uquant])
        if np.min(data) < 0:
            data = data + np.abs(np.min(data))
        norm_data = data / np.max(data)
        norm_levels = np.quantile(norm_data, [lquant, uquant])
        norm_data[np.where(norm_data < norm_levels[0])] = 0
        norm_data[np.where(norm_data > norm_levels[1])] = 1

        return norm_data, levels
