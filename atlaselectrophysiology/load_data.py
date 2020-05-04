from matplotlib import cm
from PyQt5 import QtGui
import scipy
import numpy as np
import alf.io
from oneibl.one import ONE
from brainbox.processing import bincount2D
import matplotlib.pyplot as plt
import ibllib.pipes.histology as histology
import ibllib.atlas as atlas

TIP_SIZE_UM = 200
N_BNK = 4
BNK_SIZE = 10
ONE_BASE_URL = "https://dev.alyx.internationalbrainlab.org"
one = ONE(base_url=ONE_BASE_URL)


def _cumulative_distance(xyz):
    return np.cumsum(np.r_[0, np.sqrt(np.sum(np.diff(xyz, axis=0) ** 2, axis=1))])


class LoadData:
    def __init__(self, max_idx):
        self.max_idx = max_idx

    def get_subjects(self):
        """
        Finds all subjects that have a histology track trajectory registered
        :return subjects: list of subjects
        :type subjects: list of strings
        """
        sess_with_hist = one.alyx.rest('trajectories', 'list', provenance='Histology track')
        subjects = [sess['session']['subject'] for sess in sess_with_hist]
        subjects = np.unique(subjects)

        return subjects

    def get_sessions(self, subject):
        """
        Finds all sessions for a particular subject that have a histology track trajectory
        registered
        :param subject: subject name
        :type subject: string
        :return session: list of sessions associated with subject, displayed as date + probe
        :return session: list of strings
        """
        self.subj = subject
        self.sess_with_hist = one.alyx.rest('trajectories', 'list', subject=self.subj,
                                            provenance='Histology track')
        session = [(sess['session']['start_time'][:10] + ' ' + sess['probe_name']) for sess in
                   self.sess_with_hist]
        return session

    def get_info(self, idx):
        """
        """
        self.n_sess = self.sess_with_hist[idx]['session']['number']
        self.date = self.sess_with_hist[idx]['session']['start_time'][:10]
        self.probe_label = self.sess_with_hist[idx]['probe_name']
        print(self.subj)
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
            'spikes.clusters',
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
        #self.clusters = alf.io.load_object(self.alf_path, 'clusters')
        self.chn_coords, self.chn_ind = (alf.io.load_object(self.alf_path, 'channels')).values()
        lfp_spectrum = alf.io.load_object(self.ephys_path, '_iblqc_ephysSpectralDensityLF')
        self.lfp_freq = lfp_spectrum.get('freqs')
        self.lfp_power = lfp_spectrum.get('power', [])
        if not np.any(self.lfp_power):
            self.lfp_power = lfp_spectrum.get('amps')

    def get_probe_track(self):

        self.brain_atlas = atlas.AllenAtlas(res_um=25)

        # Load in user picks for session
        insertion = one.alyx.rest('insertions', 'list', session=self.eid, name=self.probe_label)
        xyz_picks = np.array(insertion[0]['json']['xyz_picks']) / 1e6
        self.probe_id = insertion[0]['id']
        self.get_trajectory()

        # Use the top/bottom 1/4 of picks to compute the entry and exit trajectories of the probe
        n_picks = np.max([4, round(xyz_picks.shape[0] / 4)])
        self.traj_entry = atlas.Trajectory.fit(xyz_picks[:n_picks, :])
        # Force the entry to be on the upper z lim of the atlas to account for cases where channels
        # may be located above the surface of the brain
        entry = (self.traj_entry.eval_z(self.brain_atlas.bc.zlim))[0, :]

        traj_exit = atlas.Trajectory.fit(xyz_picks[-1 * n_picks:, :])
        exit = atlas.Insertion.get_brain_exit(traj_exit, self.brain_atlas)
        exit[2] = exit[2] - 200 / 1e6

        self.xyz_track = np.r_[exit[np.newaxis, :], xyz_picks, entry[np.newaxis, :]]
        # by convention the deepest point is first
        self.xyz_track = self.xyz_track[np.argsort(self.xyz_track[:, 2]), :]

        # plot on tilted coronal slice for sanity check
        # ax = self.brain_atlas.plot_tilted_slice(self.xyz_track, axis=1)
        # ax.plot(self.xyz_track[:, 0] * 1e6, self.xyz_track[:, 2] * 1e6, '-*')
        # ax.plot(xyz_picks[:, 0] * 1e6, xyz_picks[:, 2] * 1e6, 'r-*')
        # plt.show()

        self.track_init = [0] * (self.max_idx + 1)
        self.track = [0] * (self.max_idx + 1)
        self.features = [0] * (self.max_idx + 1)

        tip_distance = _cumulative_distance(self.xyz_track)[1] + TIP_SIZE_UM / 1e6
        track_length = _cumulative_distance(self.xyz_track)[-1]
        self.track_start = np.array([0, track_length]) - tip_distance

        # In some cases need to extend beyond atlas z-lim to accomodate all channels
        if self.track_start[-1] < 4000 / 1e6:
            print('extending region beyond box')
            self.track_start[-1] = 4000 / 1e6
        self.track_init[0] = np.copy(self.track_start)
        self.track[0] = np.copy(self.track_start)
        self.features[0] = np.copy(self.track_start)

        self.hist_data = {
            'region': [0] * (self.max_idx + 1),
            'axis_label': [0] * (self.max_idx + 1),
            'colour': [0] * (self.max_idx + 1),
            'chan_int': 5
        }
        self.get_histology_regions(0)
        self.hist_data_orig = self.hist_data['region'][0]

        self.scale_data = {
            'region': [0] * (self.max_idx + 1),
            'scale': [0] * (self.max_idx + 1),
            'chan_int': 5
        }

        self.get_scale_factor(0)

    def get_trajectory(self):
        self.traj_exists = False
        ephys_traj = one.alyx.rest('trajectories', 'list', probe_insertion=self.probe_id,
                                   provenance='Ephys aligned histology track')
        if len(ephys_traj):
            self.traj_exists = True

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
        self.xyz_channels = histology.interpolate_along_track(self.xyz_track, channel_depths_track)
        return self.xyz_channels

    def upload_channels(self, overwrite=False):
        insertion = atlas.Insertion.from_track(self.xyz_channels, self.brain_atlas)
        brain_regions = self.brain_atlas.regions.get(self.brain_atlas.get_labels(self.xyz_channels))
        brain_regions['xyz'] = self.xyz_channels
        brain_regions['lateral'] = self.chn_coords[:, 0]
        brain_regions['axial'] = self.chn_coords[:, 1]
        assert np.unique([len(brain_regions[k]) for k in brain_regions]).size == 1
        histology.register_aligned_track(self.probe_id, insertion, brain_regions, one=one,
                                         overwrite=overwrite)

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
        region_label[:, 0] = (self.track2feature(np.float64(region_label[:, 0]), idx) * 1e6)
        self.hist_data['region'][idx] = region
        self.hist_data['axis_label'][idx] = region_label
        self.hist_data['colour'][idx] = region_colour

    def get_scale_factor(self, idx):
        scale = []
        for iR, (reg, reg_orig) in enumerate(zip(self.hist_data['region'][idx],
                                                 self.hist_data_orig)):
            scale = np.r_[scale, (reg[1] - reg[0]) / (reg_orig[1] - reg_orig[0])]

        boundaries = np.where(np.diff(np.around(scale, 3)))[0]
        if boundaries.size == 0:
            region = np.array([[self.hist_data['region'][idx][0][0],
                               self.hist_data['region'][idx][-1][1]]])
            region_scale = np.array([1])
        else:

            region = np.empty((boundaries.size + 1, 2))
            region_scale = []
            for bound in np.arange(boundaries.size + 1):
                if bound == 0:
                    _region = np.array([self.hist_data['region'][idx][0][0],
                                       self.hist_data['region'][idx][boundaries[bound]][1]])
                    _region_scale = scale[0]
                elif bound == boundaries.size:
                    _region = np.array([self.hist_data['region'][idx][boundaries[bound - 1]][1],
                                       self.hist_data['region'][idx][-1][1]])
                    _region_scale = scale[-1]
                else:
                    _region = np.array([self.hist_data['region'][idx][boundaries[bound - 1]][1],
                                        self.hist_data['region'][idx][boundaries[bound]][1]])
                    _region_scale = scale[boundaries[bound]]

                region[bound, :] = _region
                region_scale = np.r_[region_scale, _region_scale]

        self.scale_data['region'][idx] = region
        self.scale_data['scale'][idx] = region_scale

    def get_depth_data_scatter(self):
        A_BIN = 10
        amp_range = np.quantile(self.spikes['amps'], [0.1, 0.9])
        amp_bins = np.linspace(amp_range[0], amp_range[1], A_BIN)
        colour_bin = np.linspace(0.0, 1.0, A_BIN)
        colours = (cm.get_cmap('Greys')(colour_bin)[np.newaxis, :, :3][0])*255
        spikes_colours = np.empty((self.spikes['amps'].size), dtype=object)
        spikes_size = np.empty((self.spikes['amps'].size))
        for iA in range(amp_bins.size - 1):
            idx = np.where((self.spikes['amps'] > amp_bins[iA]) & (self.spikes['amps'] <=
                                                                   amp_bins[iA + 1]))[0]
            spikes_colours[idx] = QtGui.QColor(*colours[iA])
            #spikes_colours[idx] = QtGui.QColor((1., 0.96078431, 0.94117647))
           # 1.        , 0.96078431, 0.94117647
            spikes_size[idx] = iA / (A_BIN / 4)

        scatter = {
            'x': self.spikes['times'][0:-1:100],
            'y': self.spikes['depths'][0:-1:100],
            #'colours': self.spikes['amps'][0:-1:100],
            'levels': amp_range,
            'colours': spikes_colours[0:-1:100],
            'pen': None,
            'size': spikes_size[0:-1:100],
            'xrange': np.array([np.min(self.spikes['times'][0:-1:100]),
                                np.max(self.spikes['times'][0:-1:100])]),
            'xaxis': 'Time (s)',
            'title': 'Amplitude (uV)??',
            'cmap': 'Greys'
        }

        return scatter

    def get_peak2trough_data_scatter(self):

        (clu, spike_depths,
         spike_amps, n_spikes) = self.compute_spike_average(self.spikes['clusters'],
                                                            self.spikes['depths'], 
                                                            self.spikes['amps'])

        fr = n_spikes / np.max(self.spikes['times'])

        peak_to_trough = {
            #'x': self.clusters['peakToTrough'][clu],
            'x': spike_amps,
            'y': spike_depths,
            'colours': fr,
            'pen': 'k',
            'size': np.array(5),
            'levels': np.quantile(fr, [0.1, 1]),
            'xrange': np.array([np.min(spike_amps),
                                np.max(spike_amps)]),
            'xaxis': 'Amplitude (uV)??',
            'title': 'Firing Rate',
            'cmap': 'hot'
        }

        return peak_to_trough

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
        rms_amps, rms_times = (alf.io.load_object(self.ephys_path, '_iblqc_ephysTimeRms' +
                                                  format)).values()
        _rms = np.take(rms_amps, self.chn_ind, axis=1)
        _, self.chn_depth, chn_count = np.unique(self.chn_coords[:, 1], return_index=True,
                                                 return_counts=True)
        self.chn_depth_eq = np.copy(self.chn_depth)
        self.chn_depth_eq[np.where(chn_count == 2)] += 1

        def avg_chn_depth(a):
            return(np.mean([a[self.chn_depth], a[self.chn_depth_eq]], axis=0))

        def median_subtract(a):
            return(a - np.median(a))
        img = np.apply_along_axis(avg_chn_depth, 1, _rms * 1e6)
        img = np.apply_along_axis(median_subtract, 1, img)
        levels = np.quantile(img, [0.1, 0.9])
        xscale = (rms_times[-1] - rms_times[0]) / img.shape[0]
        yscale = (np.max(self.chn_coords[:, 1]) - np.min(self.chn_coords[:, 1])) / img.shape[1]

        rms_data = {
            'img': img,
            'scale': np.array([xscale, yscale]),
            'levels': levels,
            'cmap': 'plasma',
            'xrange': np.array([rms_times[0], rms_times[-1]]),
            'xaxis': 'Time (s)',
            'title': format + ' RMS (uV)'
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
            'img': probe_img,
            'scale': probe_scale,
            'offset': probe_offset,
            'level': probe_levels,
            'cmap': 'plasma',
            'xrange': np.array([0*BNK_SIZE, (N_BNK) * BNK_SIZE]),
            'title': format + ' RMS (uV)'
        }

        return rms_data

    def get_lfp_spectrum_data(self):
        freq_bands = np.vstack(([0, 4], [4, 10], [10, 30], [30, 80], [80, 200]))
        lfp_data = {}

        for iF, freq in enumerate(freq_bands):
            freq_idx = np.where((self.lfp_freq >= freq[0]) & (self.lfp_freq < freq[1]))[0]
            lfp_avg = np.mean(self.lfp_power[freq_idx], axis=0)[self.chn_ind]
            lfp_avg_dB = 10 * np.log10(lfp_avg)
            probe_img, probe_scale, probe_offset = self.arrange_channels2banks(lfp_avg_dB)
            probe_levels = np.quantile(lfp_avg_dB, [0.1, 0.9])

            lfp_band_data = {f"{freq[0]} - {freq[1]} Hz": {
                'img': probe_img,
                'scale': probe_scale,
                'offset': probe_offset,
                'level': probe_levels,
                'cmap': 'viridis',
                'xaxis': 'Time (s)',
                'xrange': np.array([0 * BNK_SIZE, (N_BNK) * BNK_SIZE]),
                'title': f"{freq[0]} - {freq[1]} Hz (dB)"}
            }
            lfp_data.update(lfp_band_data)

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

    def compute_spike_average(self, spike_clusters, spike_depth, spike_amp):
        clust, inverse, counts = np.unique(spike_clusters, return_inverse=True, return_counts=True)
        _spike_depth = scipy.sparse.csr_matrix((spike_depth, (inverse, np.zeros(inverse.size, dtype=int))))
        _spike_amp = scipy.sparse.csr_matrix((spike_amp, (inverse, np.zeros(inverse.size, dtype=int))))
        spike_depth_avg = np.ravel(_spike_depth.toarray()) / counts
        spike_amp_avg = np.ravel(_spike_amp.toarray()) / counts
        return clust, spike_depth_avg, spike_amp_avg, counts
