
import numpy as np
from brainbox.io.one import load_spike_sorting, load_channel_locations
from oneibl.one import ONE
import random

import ibllib.atlas as atlas

ONE_BASE_URL = "https://alyx.internationalbrainlab.org"
one = ONE(base_url=ONE_BASE_URL)



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
        self.get_data()



    def get_data(self):
        #Load in all the data required
        dtypes_extra = [
            'spikes.depths',
            'spikes.amps',
            'channels.localCoordinates'
        ]

        spikes, _ = load_spike_sorting(eid=self.eid, one=one, dataset_types=dtypes_extra)
        probe_label = [key for key in spikes.keys() if int(key[-1]) == self.probe_id][0]

        self.spikes = spikes[probe_label]

        ses = one.alyx.rest('sessions', 'read', id=self.eid)

        #colours = ba.regions['rgb']
        channels = load_channel_locations(ses, one=one, probe=probe_label)[probe_label]

        brain_atlas = atlas.AllenAtlas(res_um=25)
        xyz = np.c_[channels.x, channels.y, channels.z]
        #Find the unique non-sorted channels
        idx = []
        for pos in np.unique(channels.axial_um):
            idx.append(np.where(channels.axial_um == pos)[0][0])

        xyz = xyz[idx]
        traj = atlas.Trajectory.fit(xyz)
        entry = atlas.Insertion.get_brain_entry(traj, brain_atlas)
        exit = atlas.Insertion.get_brain_exit(traj, brain_atlas)
        tip = xyz[np.argmin(xyz[:, 2]), :]
        top = xyz[np.argmax(xyz[:, 2]), :]

        self.interval = 20

        depth_t, _, _ = atlas.cart2sph(*(entry - top))
        npoints_t = np.ceil(depth_t / self.interval * 1e6).astype(int)
        depth_b, _, _ = atlas.cart2sph(*(tip - exit))
        npoints_b = np.ceil(depth_b / self.interval * 1e6).astype(int)

        xyz_t = np.c_[np.linspace(entry[0], top[0], npoints_t),
                      np.linspace(entry[1], top[1], npoints_t),
                      np.linspace(entry[2], top[2], npoints_t)]
        xyz_b = np.c_[np.linspace(tip[0], exit[0], npoints_b),
                      np.linspace(tip[1], exit[1], npoints_b),
                      np.linspace(tip[2], exit[2], npoints_b)]

        xyz_ext = np.r_[xyz_t[:-1], np.flip(xyz, axis=0), xyz_b[1:]]

        region_ids = brain_atlas.get_labels(np.flip(xyz_ext, axis=0))
        self.region_info = brain_atlas.regions.get(region_ids)
        self.offset = np.argmin(np.linalg.norm((np.flip(xyz_ext, axis=0) - tip), axis=1))

        #cax = self.brain_atlas.plot_cslice(xyz[0, 1], volume='annotation')
        #cax.plot(xyz[:, 0] * 1e6, xyz[:, 2] * 1e6, 'k*')
        #xyz_flip = np.flip(xyz, axis=0)
        region_ids = brain_atlas.get_labels(np.flip(xyz_ext, axis=0))
        self.region_info = brain_atlas.regions.get(region_ids)
    

    def get_scatter_data(self):
        scatter = {
            'times': self.spikes['times'][0:-1:100],
            'depths': self.spikes['depths'][0:-1:100]
        }

        return scatter


    def get_histology_data(self):
        boundaries = np.where(np.diff(self.region_info.id))[0]

        region = np.empty((5, len(boundaries)+ 1,2))
        region_label = np.empty((len(boundaries)+1,1), dtype=object)
        region_axis_label = np.empty((5 ,len(boundaries)+ 1,2), dtype=object)
        region_colour = np.empty((len(boundaries)+1,3), dtype=int)
      

        #region = np.empty((len(boundaries) + 1, 2))
        #region_label = np.empty((len(boundaries) + 1, 2), dtype=object)
        #region_colour = np.empty((len(boundaries) + 1, 3), dtype=int)


        for idx in range(len(boundaries) + 1):
            if idx == 0:
                _region = np.array([0, boundaries[idx]])
            elif idx == len(boundaries):
                _region = np.array([boundaries[idx - 1], len(self.region_info.id) - 1])
            else: 
                _region = np.array([boundaries[idx - 1], boundaries[idx]])

            
            _region_colour = self.region_info.rgb[_region[1]]
            _region_label = self.region_info.acronym[_region[1]]
            _region = (_region * self.interval) - self.offset * self.interval
            _region_mean = np.mean(_region, dtype=int)
        
            region[0,idx,:] = _region
            region_colour[idx,:] = _region_colour 
            region_label[idx] = _region_label
            region_axis_label[0, idx,:] = (_region_mean, _region_label)



            #region[idx,:] = _region
            #region_colour[idx,:] = _region_colour
            #region_label[idx,:] = (_region_mean, _region_label)

        histology = {
            'region': region,
            'axis_label': region_axis_label,
            'label': region_label,
            'colour': region_colour,
            'chan_int': self.interval
        }

        return histology

    def create_random_hex_colours(self, unique_regions):

        ##load the allen structure tree from ibllib/atlas
    
        colour = {}
        random.seed(8)
        for reg in unique_regions:
            colour[reg] = "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
    
        return colour
    
    def get_amplitude_data(self):
        
        depths = self.spikes['depths']
        depth_int = 40
        depth_bins = np.arange(0, max(self.channel_coord[:, 1]) + depth_int, depth_int)
        #depth_bins = np.flip(depth_bins)
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
            depth_amps_fr.append(np.histogram(amps[depth_idx], amp_bins)[0]/ time_max)
            depth_amps.append(np.mean(amps[depth_idx]))
            depth_fr.append(len(depth_idx) / time_max)

        #print(depth_hist)
        corr = np.corrcoef(depth_hist)
        #print(corr)
        corr[np.isnan(corr)] = 0

        amplitude = {
            'amps': depth_amps,
            'fr': depth_fr,
            'amps_fr': depth_amps_fr,
            'corr': corr,
            'bins': depth_bins_cnt
        }

        return amplitude
        




