from pathlib import Path
import sys
from matplotlib import cm
from PyQt5 import QtGui
import scipy
import numpy as np
import alf.io
from brainbox.io.one import load_spike_sorting, load_channel_locations
from oneibl.one import ONE
from brainbox.processing import bincount2D
import random
import matplotlib.pyplot as plt
import ibllib.pipes.histology as histology
import ibllib.atlas as atlas

TIP_SIZE_UM = 200
N_BNK = 4
BNK_SIZE = 10
ONE_BASE_URL = "https://alyx.internationalbrainlab.org"
one = ONE(base_url=ONE_BASE_URL)


def _cumulative_distance(xyz):
    return np.cumsum(np.r_[0, np.sqrt(np.sum(np.diff(xyz, axis=0) ** 2, axis=1))])


class LoadData:
    #def __init__(self, subj, date, sess=None, probe_id=None):
    def __init__(self):
        la = 1

    def get_subjects(self):
        sess_with_hist = one.alyx.rest('trajectories', 'list', provenance='Histology track')
        subjects = [sess['session']['subject'] for sess in sess_with_hist]
        subjects = np.unique(subjects)

        return subjects

    def get_sessions(self, subject):
        self.subj = subject
        self.sess_with_hist = one.alyx.rest('trajectories', 'list', subject=self.subj,
                                            provenance='Histology track')
        session = [(sess['session']['start_time'][:10] + ' ' + sess['probe_name']) for sess in
                   self.sess_with_hist]
        return session
    
    def get_info(self, idx):
        self.n_sess = self.sess_with_hist[idx]['session']['number']
        self.date = self.sess_with_hist[idx]['session']['start_time'][:10]
        self.probe_label = self.sess_with_hist[idx]['probe_name']
        print(self.probe_label)
        print(self.date)
    
    def get_eid(self):
        eids = one.search(subject=self.subj, date=self.date, number=self.n_sess,
                          task_protocol='ephys')
        self.eid = eids[0]
        print(self.eid)

    def get_data(self):
        # Load in all the data required
        dtypes = [
            'spikes.depths',
            'spikes.amps',
            'spikes.times',
            'channels.localCoordinates',
            'channels.rawInd',
            '_iblqc_ephysTimeRms.rms',
            '_iblqc_ephysTimeRms.timestamps',
            '_iblqc_ephysSpectralDensity.freqs',
            '_iblqc_ephysSpectralDensity.amps',
            '_iblqc_ephysSpectralDensity.power'
        ]

        _ = one.load(self.eid, dataset_types=dtypes, download_only=True)
        path = one.path_from_eid(self.eid)
        self.alf_path = path.joinpath('alf', self.probe_label)
        self.ephys_path = path.joinpath('raw_ephys_data', self.probe_label)
        self.spikes = alf.io.load_object(self.alf_path, 'spikes')
        self.chn_coords, self.chn_ind = (alf.io.load_object(self.alf_path, 'channels')).values()
        lfp_spectrum = alf.io.load_object(self.ephys_path, '_iblqc_ephysSpectralDensityLF')
        self.lfp_freq = lfp_spectrum.get('freqs')
        self.lfp_power = lfp_spectrum.get('power', [])
        if not np.any(self.lfp_power):
            self.lfp_power = lfp_spectrum.get('amps')

        #self.alf_path = '/Users/Mayo/Downloads/FlatIron/cortexlab/Subjects/KS014/2019-12-07/001/alf/probe00'
        #self.chn_coords = alf.io.load_file_content(Path(self.alf_path, 'channels.localCoordinates.npy'))
        
        self.brain_atlas = atlas.AllenAtlas(res_um=25)

        insertion = one.alyx.rest('insertions', 'list', session=self.eid, name=self.probe_label)
        xyz_picks = np.array(insertion[0]['json']['xyz_picks']) / 1e6
        # extrapolate to find the brain entry/exit using only the top/bottom 1/4 of picks
        n_picks = np.max([4, round(xyz_picks.shape[0] / 4)])
        traj_entry = atlas.Trajectory.fit(xyz_picks[:n_picks, :])
        entry = (traj_entry.eval_z(self.brain_atlas.bc.zlim))[0, :]
        #entry = atlas.Insertion.get_brain_entry(traj_entry, self.brain_atlas)
        #entry[2] = entry[2] + 200 / 1e6
        
        traj_exit = atlas.Trajectory.fit(xyz_picks[-1 * n_picks:, :])

        #z = self.brain_atlas.bc.zlim[-1]
        #print(z)
        #print(traj_exit.eval_z(z))
        #for m in range(5):
        #    print(m)
        #    xyz = traj_exit.eval_z(z)[0]
        #    print(xyz)
        #    iy = self.brain_atlas.bc.y2i(xyz[1])
        #    print(iy)
        #    ix = self.brain_atlas.bc.x2i(xyz[0])
        #    print(ix)
        #    z = self.brain_atlas.bottom[iy, ix]
        #    print(z)

        exit = atlas.Insertion.get_brain_exit(traj_exit, self.brain_atlas)
        exit[2] = exit[2] - 200 / 1e6

        self.xyz_track = np.r_[exit[np.newaxis, :], xyz_picks, entry[np.newaxis, :]]
        # by convention the deepest point is first
        self.xyz_track = self.xyz_track[np.argsort(self.xyz_track[:, 2]), :]

        # plot on tilted coronal slice for sanity check
        ax = self.brain_atlas.plot_tilted_slice(self.xyz_track, axis=1)
        ax.plot(self.xyz_track[:, 0] * 1e6, self.xyz_track[:, 2] * 1e6, '-*')
        ax.plot(xyz_picks[:, 0] * 1e6, xyz_picks[:, 2] * 1e6, 'r-*')

        plt.show()

        self.max_idx = 10
        self.track_init = [0] * (self.max_idx + 1)
        self.track = [0] * (self.max_idx + 1)
        self.features = [0] * (self.max_idx + 1)

        #tip_distance = _cumulative_distance(self.xyz_track)[2] + TIP_SIZE_UM / 1e6
        #DOUBLE CHECK THIS!!!!!
        tip_distance = _cumulative_distance(self.xyz_track)[1] + TIP_SIZE_UM / 1e6
        track_length = _cumulative_distance(self.xyz_track)[-1]
        self.track_start = np.array([0, track_length]) - tip_distance

        # In case the probe isn't fully in the brain!
        if self.track_start[-1] < 4000 / 1e6:
            print('extending region beyond box')
            self.track_start[-1] = 4000 / 1e6
        self.track_init[0] = np.copy(self.track_start)
        self.track[0] = np.copy(self.track_start)
        self.features[0] = np.copy(self.track_start)


    def feature2track(self, trk, idx):
        fcn = scipy.interpolate.interp1d(self.features[idx], self.track[idx])
        return fcn(trk)

    def track2feature(self, ft, idx):
        fcn = scipy.interpolate.interp1d(self.track[idx], self.features[idx])
        return fcn(ft)

    def get_channels_coordinates(self, idx, depths=None):
        """
        Gets 3d coordinates from a depth along the electrophysiology feature. 2 steps
        1) interpolate from the electrophys features depths space to the probe depth space
        2) interpolate from the probe depth space to the true 3D coordinates
        if depths is not provided, defaults to channels local coordinates depths
        """
        if depths is None:
            depths = self.chn_coords[:, 1] / 1e6
        # nb using scipy here so we can change to cubic spline if needed
        channel_depths_track = self.feature2track(depths, idx) - self.track[idx][0]
        return histology.interpolate_along_track(self.xyz_track, channel_depths_track)

    def get_histology_regions(self, idx):
        """
        Samples at 10um along the trajectory
        :return:
        """
        sampling_trk = np.arange(self.track[idx][0],
                                 self.track[idx][-1] - 10 * 1e-6, 10 * 1e-6)

        xyz_samples = histology.interpolate_along_track(self.xyz_track,
                                                        sampling_trk - sampling_trk[0])

        region_ids = self.brain_atlas.get_labels(xyz_samples)
        region_info = self.brain_atlas.regions.get(region_ids)

        boundaries = np.where(np.diff(region_info.id))[0]
        region = np.empty((boundaries.size + 1, 2))
        region_label = np.empty((boundaries.size + 1, 2), dtype=object)
        region_colour = np.empty((boundaries.size + 1, 3), dtype=int)

        for bound in np.arange(boundaries.size + 1):
            if bound == 0:
                _region = np.array([0, boundaries[bound]])
            elif bound == boundaries.size:
                _region = np.array([boundaries[bound - 1], region_info.id.size - 1])
            else:
                _region = np.array([boundaries[bound - 1], boundaries[bound]])

            _region_colour = region_info.rgb[_region[1]]
            _region_label = region_info.acronym[_region[1]]
            _region = sampling_trk[_region] 
            _region_mean = np.mean(_region)

            region[bound, :] = _region
            region_colour[bound, :] = _region_colour
            region_label[bound, :] = (_region_mean, _region_label)

        region = self.track2feature(region, idx) * 1e6
        region_label[:, 0] = np.int64(self.track2feature(np.float64(region_label[:, 0]),
                                      idx) * 1e6)
        return region, region_label, region_colour

    def get_depth_data_scatter(self):
        A_BIN = 10
        amp_range = np.quantile(self.spikes['amps'], [0.1, 0.9])
        amp_bins = np.linspace(amp_range[0], amp_range[1], A_BIN)
        colour_bin = np.linspace(0.0, 1.0, A_BIN)
        colours = cm.get_cmap('Greys')(colour_bin)[np.newaxis, :, :3][0]
        spikes_colours = np.empty((self.spikes['amps'].size), dtype=object)
        spikes_size = np.empty((self.spikes['amps'].size))
        for iA in range(amp_bins.size - 1):
            idx = np.where((self.spikes['amps'] > amp_bins[iA]) & (self.spikes['amps'] <=
                                                                   amp_bins[iA + 1]))[0]
            spikes_colours[idx] = QtGui.QColor(*colours[iA])
            spikes_size[idx] = iA / (A_BIN / 4)

        scatter = {
            'times': self.spikes['times'][0:-1:100],
            'depths': self.spikes['depths'][0:-1:100],
            'colours': spikes_colours[0:-1:100],
            'size': spikes_size[0:-1:100],
            'xaxis': 'Time (s)'
        }

        return scatter

    def get_depth_data_img(self):
        T_BIN = 0.05
        D_BIN = 5
        R, times, depths = bincount2D(self.spikes['times'], self.spikes['depths'], T_BIN, D_BIN,
                                      ylim=[0, np.max(self.chn_coords[:, 1])])
        img = R.T
        xscale = (times[-1] - times[0]) / img.shape[0]
        yscale = (depths[-1] - depths[0]) / img.shape[1]
        
        depth = {
            'img': img,
            'scale': np.array([xscale, yscale]),
            'levels': np.array([1, 0]),
            'xrange': np.array([times[0], times[-1]]),
            'xaxis': 'Time (s)'
        }

        return depth
    
    def get_fr_data_line(self):
        T_BIN = np.max(self.spikes['times'])
        D_BIN = 5
        R, times, depths = bincount2D(self.spikes['times'], self.spikes['depths'], T_BIN, D_BIN,
                                      ylim=[0, np.max(self.chn_coords[:, 1])])
        R = R / T_BIN
        fr = {
            'x': R[:, 0],
            'y': depths,
            'xrange': np.array([0, np.max(R[:, 0])]),
            'xaxis': 'Firing Rate (Sp/s)'
        }

        return fr

    def get_amp_data_line(self):
        T_BIN = np.max(self.spikes['times'])
        D_BIN = 5
        R, times, depths = bincount2D(self.spikes['amps'], self.spikes['depths'], T_BIN, D_BIN,
                                      ylim=[0, np.max(self.chn_coords[:, 1])])
        amp = {
            'x': R[:, 0],
            'y': depths,
            'xrange': np.array([0, np.max(R[:, 0])]),
            'xaxis': 'Amplitude (uV???)'
        }

        return amp

    
    def get_rms_data_img(self, format):
        # Finds channels that are at equivalent depth on probe and averages rms values for each 
        # time point at same depth togehter
        rms_amps, rms_times = (alf.io.load_object(self.ephys_path, '_iblqc_ephysTimeRms' + format)).values()
        #rms = alf.io.load_object(self.ephys_path, '_iblqc_ephysTimeRmsAP')
        #rms_amps = rms.rms
        _rms = np.take(rms_amps, self.chn_ind, axis=1)
        _, self.chn_depth, chn_count = np.unique(self.chn_coords[:, 1], return_index=True,
                                                 return_counts=True)
        self.chn_depth_eq = np.copy(self.chn_depth)
        self.chn_depth_eq[np.where(chn_count == 2)] += 1

        def avg_chn_depth(a):
            return(np.mean([a[self.chn_depth], a[self.chn_depth_eq]], axis=0))
        
        img = np.apply_along_axis(avg_chn_depth, 1, _rms * 1e6)
        #levels = [np.min(img), np.max(img)]
        levels = np.quantile(img, [0, 0.5])
        xscale = (rms_times[-1] - rms_times[0]) / img.shape[0]
        yscale = (np.max(self.chn_coords[:, 1]) - np.min(self.chn_coords[:, 1])) / img.shape[1]

        # Finds the average rms value on each electrode across all time points
        rms_avg = (np.mean(rms_amps, axis=0)[self.chn_ind]) * 1e6
        probe_levels = np.quantile(rms_avg, [0.1, 0.9])
        probe_img, probe_scale, probe_offset = self.arrange_channels2banks(rms_avg)
        
        X_OFFSET = rms_times[-1] + 150

        rms_data = {
            'img': img,
            'scale': np.array([xscale, yscale]),
            'levels': probe_levels,
            'xrange': np.array([0, X_OFFSET + N_BNK * BNK_SIZE]),
            'cmap': 'plasma',
            'title': format + ' RMS (uV)',
            'axis': ['Time (s)', 'Distance from probe tip (um)'],
            'probe_img': [probe_img],
            'probe_scale': [probe_scale],
            'probe_offset': [probe_offset],
            'probe_level': [probe_levels],
            'probe_cmap': 'plasma',
            'extra_offset': [X_OFFSET],
            'plot_cmap': False
        }

        return rms_data

    def get_rms_data_probe(self, format):
        # Finds channels that are at equivalent depth on probe and averages rms values for each 
        # time point at same depth togehter
        rms_amps, rms_times = (alf.io.load_object(self.ephys_path, '_iblqc_ephysTimeRms' + format)).values()

        # Finds the average rms value on each electrode across all time points
        rms_avg = (np.mean(rms_amps, axis=0)[self.chn_ind]) * 1e6
        probe_levels = np.quantile(rms_avg, [0.1, 0.9])
        probe_img, probe_scale, probe_offset = self.arrange_channels2banks(rms_avg)

        rms_data = {
            'title': format + ' RMS (uV)',
            'xaxis': 'Time (s)',
            'xrange': np.array([0*BNK_SIZE, (N_BNK) * BNK_SIZE]),
            'probe_img': probe_img,
            'probe_scale': probe_scale,
            'probe_offset': probe_offset,
            'probe_level': probe_levels,
            'probe_cmap': 'plasma',
            'plot_cmap': True
        }

        return rms_data

    def get_lfp_spectrum_data(self):
        X_OFFSET = 1000
        freq_bands = np.vstack(([0, 4], [4, 10], [10, 30], [30, 80], [80, 200]))
        probe_img = []
        probe_scale = np.empty((freq_bands.shape[0], 4, 2))
        probe_offset = np.empty((freq_bands.shape[0], 4, 2))
        probe_xoffset = np.empty((freq_bands.shape[0], 1))
        probe_level = np.empty((freq_bands.shape[0], 2))
        probe_title = np.empty((freq_bands.shape[0], 1), dtype=object)

        for iF, freq in enumerate(freq_bands):
            freq_idx = np.where((self.lfp_freq >= freq[0]) & (self.lfp_freq < freq[1]))[0]
            lfp_chns = np.mean(self.lfp_power[freq_idx], axis=0)[self.chn_ind]
            lfp_chns_dB = 10 * np.log10(lfp_chns)
            lfp_bnks, lfp_scale, lfp_offset = self.arrange_channels2banks(lfp_chns_dB)
            lfp_level = np.quantile(lfp_chns_dB, [0.1, 0.9])
            lfp_title = f"{freq[0]} - {freq[1]} Hz (dB)"

            probe_img.append(lfp_bnks)
            probe_scale[iF, :] = lfp_scale
            probe_offset[iF, :] = lfp_offset
            probe_level[iF, :] = lfp_level
            probe_title[iF, :] = lfp_title
            probe_xoffset[iF, :] = iF * X_OFFSET
        
        # A bit of a hack
        img = np.zeros((np.unique(self.chn_coords[:, 1]).size,
                        np.unique(self.chn_coords[:, 1]).size))
        xmax = np.max(probe_xoffset) + N_BNK * BNK_SIZE
        xscale = (xmax - 0) / img.shape[0]
        yscale = (np.max(self.chn_coords[:, 1]) - np.min(self.chn_coords[:, 1]) / img.shape[1])

        lfp_data = {
            'img': img,
            'scale': np.array([xscale, yscale]),
            'xrange': np.array([0, xmax]),
            'axis': ['Time (s)', ''],
            'probe_img': probe_img,
            'probe_scale': probe_scale,
            'probe_offset': probe_offset,
            'probe_level': probe_level,
            'probe_title': probe_title,
            'extra_offset': probe_xoffset,
            'probe_cmap': 'viridis',
            'plot_cmap': True
        }

        return lfp_data

    def get_correlation_data_img(self):

        T_BIN = 0.05
        D_BIN = 40
        R, times, depths = bincount2D(self.spikes['times'], self.spikes['depths'], T_BIN, D_BIN,
                                      ylim=[0, np.max(self.chn_coords[:, 1])])
        corr = np.corrcoef(R)
        corr[np.isnan(corr)] = 0
        scale = (np.max(depths) - np.min(depths)) / corr.shape[0]

        correlation = {
            'img': corr,
            'scale': np.array([scale, scale]),
            'levels': np.array([np.min(corr), np.max(corr)]),
            'xrange': np.array([np.min(self.chn_coords[:, 1]), np.max(self.chn_coords[:, 1])]),
            'cmap': 'viridis',
            'title': 'Correlation',
            'xaxis': 'Distance from probe tip (um)'
        }

        return correlation

    def arrange_channels2banks(self, data):
        Y_OFFSET = 20
        bnk_data = []
        bnk_scale = np.empty((N_BNK, 2))
        bnk_offset = np.empty((N_BNK, 2))
        for iX, x in enumerate(np.unique(self.chn_coords[:, 0])):
            bnk_idx = np.where(self.chn_coords[:, 0] == x)[0]
            bnk_vals = data[bnk_idx]
            _bnk_data = np.reshape(bnk_vals, (bnk_vals.size, 1)).T
            _bnk_yscale = ((np.max(self.chn_coords[bnk_idx, 1]) -
                            np.min(self.chn_coords[bnk_idx, 1])) / _bnk_data.shape[1])
            _bnk_xscale = BNK_SIZE / _bnk_data.shape[0]
            _bnk_yoffset = np.min(self.chn_coords[bnk_idx, 1]) - Y_OFFSET
            _bnk_xoffset = BNK_SIZE * iX

            bnk_data.append(_bnk_data)
            bnk_scale[iX, :] = np.array([_bnk_xscale, _bnk_yscale])
            bnk_offset[iX, :] = np.array([_bnk_xoffset, _bnk_yoffset])

        return bnk_data, bnk_scale, bnk_offset
