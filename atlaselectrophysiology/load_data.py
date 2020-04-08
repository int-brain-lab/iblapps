
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
        traj = atlas.Trajectory.fit(xyz)
        top = atlas.Insertion.get_brain_entry(traj, brain_atlas)
        bottom = atlas.Insertion.get_brain_exit(traj, brain_atlas)
        tip = xyz[np.argmin(xyz[:, 2]), :]
        depth, _, _ = atlas.cart2sph(*(top - bottom))
        depth1, _, _ = atlas.cart2sph(*(top - tip))
        depth2, _, _ = atlas.cart2sph(*(tip - bottom))
        
        ## this insertion is the best fit of the channel positions
        npoints = np.ceil(depth / 20 * 1e6).astype(int)
        npoints1 = np.ceil(depth1 / 20 * 1e6).astype(int)
        npoints2 = np.ceil(depth2 / 20 * 1e6).astype(int)
        #xyz_ext = np.c_[np.linspace(*ins.xyz[:, 0], npoints),
        #                np.linspace(*ins.xyz[:, 1], npoints),
        #                np.linspace(*ins.xyz[:, 2], npoints)]

        xyz_ext = np.c_[np.linspace(top[0], bottom[0], npoints),
                        np.linspace(top[1], bottom[1], npoints),
                        np.linspace(top[2], bottom[2], npoints)]
        
        xyz1 = np.c_[np.linspace(top[0], tip[0], npoints1),
                     np.linspace(top[1], tip[1], npoints1),
                     np.linspace(top[2], tip[2], npoints1)]
        xyz2 = np.c_[np.linspace(tip[0], bottom[0], npoints2),
                     np.linspace(tip[1], bottom[1], npoints2),
                     np.linspace(tip[2], bottom[2], npoints2)]

        xyz_ext = np.r_[xyz1[:-1], xyz2]
        xyz_ext = xyz1
        #self.offset = np.argmin(np.linalg.norm((np.flip(xyz_ext, axis=0) - tip), axis=1)) - 2
        #print(self.offset)
        self.offset = len(xyz2)
        print(self.offset)
        self.offset=0
        #self.offset = npoints - np.argmin(np.linalg.norm((xyz_ext - tip), axis=1))
#
        #cax = self.brain_atlas.plot_cslice(xyz[0, 1], volume='annotation')
        #cax.plot(xyz[:, 0] * 1e6, xyz[:, 2] * 1e6, 'k*')
        #xyz_flip = np.flip(xyz, axis=0)
        region_ids = brain_atlas.get_labels(np.flip(xyz_ext, axis=0))
        self.region_info = brain_atlas.regions.get(region_ids)

        self.channels=channels

        self.channel_coord = one.load_dataset(eid=self.eid, dataset_type='channels.localCoordinates')
        #assert np.all(np.c_[self.channels.lateral_um, self.channels.axial_um] == self.channel_coord)
    

    def get_scatter_data(self):
        scatter = {
            'times': self.spikes['times'][0:-1:100],
            'depths': self.spikes['depths'][0:-1:100]
        }

        return scatter


    def get_histology_data2(self):
        boundaries = np.where(np.diff(self.region_info.id))[0]

        #region = np.empty((len(boundaries)+ 1,5,2))
        #region_label = np.empty((5 ,len(boundaries)+ 1,2))
        #region_colour = np.empty((len(boundaries)+1,3), dtype=int)
      

        region = np.empty((len(boundaries)+1,2))
        region_label = np.empty((len(boundaries)+1, 2), dtype=object)
        region_colour = np.empty((len(boundaries)+1,3), dtype=int)


        for idx in range(len(boundaries) + 1):
            if idx == 0:
                _region = np.array([0, boundaries[idx]])
            elif idx == len(boundaries):
                _region = np.array([boundaries[idx - 1], len(self.region_info.id) - 1])
            else: 
                _region = np.array([boundaries[idx - 1], boundaries[idx]])

            
            
            _region_colour = self.region_info.rgb[_region[1]]
            _region_label = self.region_info.acronym[_region[1]]
            #print(_region)
            _region = (_region * 20) - self.offset * 20
            #print(_region)
            _region_mean = np.mean(_region, dtype=int)
        
            #region[idx,0,:] = _region
            #region_colour[idx,:] = _region_colour 
            #region_label[0,idx,:] = (_region_mean, _region_label)

            region[idx,:] = _region
            region_colour[idx,:] = _region_colour
            region_label[idx,:] = (_region_mean, _region_label)

            histology = {
                'boundary': region,
                'boundary_label': region_label,
                'boundary_colour': region_colour,
                'chan_int': 20
            }

        return histology

    
    def get_histology_data(self):
        #acronym = np.flip(self.channels['acronym'])
        acronym = self.channels['acronym']
        colour = self.create_random_hex_colours(np.unique(acronym))  # TODO: ibllib.atlas get colours from Allen

        # Find all boundaries from histology
        #boundaries = np.where(np.diff(np.flip(self.channels.atlas_id)))[0]
        boundaries = np.where(np.diff(self.channels.atlas_id))[0]

        #region = []
        region = np.empty((len(boundaries)+1,2))
        #region_label = []
        region_label = np.empty((len(boundaries)+1, 2), dtype=object)
        #region_colour = []
        #region_colour = np.empty((len(boundaries)+1), dtype = '<U7')
        region_colour = np.empty((len(boundaries)+1,3), dtype=int)

        for idx in range(len(boundaries) + 1):
            if idx == 0:
                _region = [0, boundaries[idx]]
            elif idx == len(boundaries):
                _region = [boundaries[idx - 1], len(self.channel_coord) - 1]
            else: 
                _region = [boundaries[idx - 1], boundaries[idx]]

            _region_label = acronym[_region][1]
            _region_colour = colour[_region_label]
            _region = self.channel_coord[:, 1][_region]
            _region_mean = np.mean(_region, dtype=int)

            region[idx,:] = _region
            region_colour[idx,:] = [ 72, 200,  60]
            region_label[idx,:] = (_region_mean, _region_label)
            #region_label.append((_region_mean, _region_label))
            #region_colour.append(_region_colour)
            #region.append(_region)



        #region_label.insert(0, (-230, 'extra'))
        #region_label.append((4090, 'extra'))
        #region_colour.insert(0, '#808080')
        #region_colour.append('#808080')
        #region.insert(0,[-480, 20])
        #region.append([3840, 3840 + 500])

        histology = {
            'boundary': region,
            'boundary_label': region_label,
            'boundary_colour': region_colour,
            'acronym': acronym,
            'chan_int': 20
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
        




