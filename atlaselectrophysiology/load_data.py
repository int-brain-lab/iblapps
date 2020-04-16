
import scipy
import numpy as np
from brainbox.io.one import load_spike_sorting, load_channel_locations
from oneibl.one import ONE
import random

import ibllib.pipes.histology as histology
import ibllib.atlas as atlas

TIP_SIZE_UM = 200
ONE_BASE_URL = "https://alyx.internationalbrainlab.org"
one = ONE(base_url=ONE_BASE_URL)


def _cumulative_distance(xyz):
    return np.cumsum(np.r_[0, np.sqrt(np.sum(np.diff(xyz, axis=0) ** 2, axis=1))])


class LoadData:
    def __init__(self, subj, date, sess=None, probe_id=None):
        if not sess:
            sess = 1
        if not probe_id:
            probe_id = 0

        eids = one.search(subject=subj, date=date, number=sess, task_protocol='ephys')
        self.eid = eids[0]
        print(self.eid)
        self.probe_id = probe_id
        
        #self.brain_atlas, self.probe_coord = self.get_data()
        self.get_data()
        # self.scale_data([1,0])




    def get_data(self):
        #Load in all the data required
        dtypes_extra = [
            'spikes.depths',
            'spikes.amps',
            'clusters.peakToTrough',
            'channels.localCoordinates'
        ]

        spikes, _ = load_spike_sorting(eid=self.eid, one=one, dataset_types=dtypes_extra)
        probe_label = [key for key in spikes.keys() if int(key[-1]) == self.probe_id][0]
        self.channel_coords = one.load_dataset(eid=self.eid, dataset_type='channels.localCoordinates')

        self.spikes = spikes[probe_label]
        self.brain_atlas = atlas.AllenAtlas(res_um=25)

        insertion = one.alyx.rest('insertions', 'list', session=self.eid, name=probe_label)
        xyz_picks = np.array(insertion[0]['json']['xyz_picks']) / 1e6
        # extrapolate to find the brain entry/exit using only the top/bottom 1/4 of picks
        n_picks = round(xyz_picks.shape[0] / 4)
        traj_entry = atlas.Trajectory.fit(xyz_picks[:n_picks, :])
        entry = atlas.Insertion.get_brain_entry(traj_entry, self.brain_atlas)
        traj_exit = atlas.Trajectory.fit(xyz_picks[-1 * n_picks:, :])
        exit = atlas.Insertion.get_brain_exit(traj_exit, self.brain_atlas)

        self.xyz_track = np.r_[exit[np.newaxis, :], xyz_picks, entry[np.newaxis, :]]
        # by convention the deepest point is first
        self.xyz_track = self.xyz_track[np.argsort(self.xyz_track[:, 2]), :]

        # plot on tilted coronal slice for sanity check
        # ax = self.brain_atlas.plot_tilted_slice(self.xyz_track, axis=1)
        # ax.plot(self.xyz_track[:, 0] * 1e6, self.xyz_track[:, 2] * 1e6, '-*')

        tip_distance = _cumulative_distance(self.xyz_track)[2] + TIP_SIZE_UM / 1e6
        shank_height = np.max(self.channel_coords[:, 1]) / 1e6

        self.depths_features = np.array([0, shank_height])
        self.depths_track = np.array([0, shank_height]) + tip_distance


    def scale_data(self, coeff):
        fit = np.poly1d(coeff)
        print(fit)
        self.chan_probe_scale = fit(self.chan_probe)
        self.chan_probe_ext_scale = fit(self.chan_probe_ext)
        self.xyz_channels = np.zeros((self.chan_probe.shape[0], 3))
        self.xyz_channels_ext = np.zeros((self.chan_probe_ext.shape[0], 3))
        for m in np.arange(3):
            self.xyz_channels[:, m] = np.interp(self.chan_probe_scale / 1e6, self.d, self.xyz_ext[:, m])
            self.xyz_channels_ext[:, m] = np.interp(self.chan_probe_ext_scale / 1e6, self.d, self.xyz_ext[:, m])
        
        return self.xyz_channels


    def get_scatter_data(self):
        scatter = {
            'times': self.spikes['times'][0:-1:100],
            'depths': self.spikes['depths'][0:-1:100]
        }

        return scatter

    def feature2track(self, trk):
        fcn = scipy.interpolate.interp1d(self.depths_features, self.depths_track)
        return fcn(trk)

    def track2feature(self, ft):
        fcn = scipy.interpolate.interp1d(self.depths_track, self.depths_features)
        return fcn(ft)

    def get_channels_coordinates(self, depths=None):
        """
        Gets 3d coordinates from a depth along the electrophysiology feature. 2 steps
        1) interpolate from the electrophys features depths space to the probe depth space
        2) interpolate from the probe depth space to the true 3D coordinates
        if depths is not provided, defaults to channels local coordinates depths
        """
        if depths is None:
            depths = self.channel_coords[:, 1] / 1e6
        # nb using scipy here so we can change to cubic spline if needed
        channel_depths_track = self.feature2track(depths)
        return histology.interpolate_along_track(self.xyz_track, channel_depths_track)

    def get_histology_regions(self):
        """
        Samples at 10um along the trajectory
        :return:
        """
        sampling_trk = np.arange(self.depths_track[0], self.depths_track[-1] - 10 * 1e-6, 10 * 1e-6)
        sampling_fts = self.track2feature(sampling_trk)
        xyz_samples = histology.interpolate_along_track(self.xyz_track, sampling_trk)
        region_ids = self.brain_atlas.get_labels(xyz_samples)
        region_info = self.brain_atlas.regions.get(region_ids)

        boundaries = np.where(np.diff(region_info.id))[0]

        region = np.empty((len(boundaries) + 1, 2))
        region_label = np.empty((len(boundaries) + 1, 2), dtype=object)
        region_colour = np.empty((len(boundaries) + 1, 3), dtype=int)

        for idx in np.arange(len(boundaries) + 1):
            if idx == 0:
                _region = np.array([0, boundaries[idx]])
            elif idx == len(boundaries):
                _region = np.array([boundaries[idx - 1], len(region_info.id) - 1])
            else: 
                _region = np.array([boundaries[idx - 1], boundaries[idx]])
            
            _region_colour = region_info.rgb[_region[1]]
            _region_label = region_info.acronym[_region[1]]
            _region = sampling_fts[_region]
            _region_mean = np.mean(_region * 1e6, dtype=int)

            region[idx, :] = _region * 1e6
            region_colour[idx, :] = _region_colour
            region_label[idx, :] = (_region_mean, _region_label)

        return region, region_label, region_colour

    def get_amplitude_data(self):
        
        depths = self.spikes['depths']
        depth_int = 40
        depth_bins = np.arange(0, max(self.channel_coords[:, 1]) + depth_int, depth_int)
        depth_bins_cnt = depth_bins[:-1] + depth_int / 2

        amps = self.spikes['amps'] * 1e6 * 2.54  ##Check that this scaling factor is correct!!
        amp_int = 50
        amp_bins = np.arange(min(amps), max(amps), amp_int)

        times = self.spikes['times']
        time_min = min(times)
        time_max = max(times)
        time_int = 0.01
        time_bins = np.arange(time_min, time_max, time_int)

        depth_amps = []
        depth_fr = []
        depth_amps_fr = []
        depth_hist = []

        for iD in range(len(depth_bins) - 1):
            depth_idx = np.where((depths > depth_bins[iD]) & (depths <= depth_bins[iD + 1]))[0]
            #print(len(depth_idx))
            depth_hist.append(np.histogram(times[depth_idx], time_bins)[0])
            #print(depth_hist)
            #depth_amps_fr.append(np.histogram(amps[depth_idx], amp_bins)[0]/ time_max)
            #depth_amps.append(np.mean(amps[depth_idx]))
            #depth_fr.append(len(depth_idx) / time_max)

        #print(depth_hist)
        corr = np.corrcoef(depth_hist)
        #print(corr)
        corr[np.isnan(corr)] = 0
    
        amplitude = {
            #'amps': depth_amps,
            #'fr': depth_fr,
            #'amps_fr': depth_amps_fr,
            'corr': corr,
            'bins': depth_bins
        }

        return amplitude
        




