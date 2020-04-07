
import numpy as np
from brainbox.io.one import load_spike_sorting, load_channel_locations
from oneibl.one import ONE
import random

from ibllib.atlas import AllenAtlas

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
        self.channels = load_channel_locations(ses, one=one, probe=probe_label)[probe_label]



        self.channel_coord = one.load_dataset(eid=self.eid, dataset_type='channels.localCoordinates')
        assert np.all(np.c_[self.channels.lateral_um, self.channels.axial_um] == self.channel_coord)
    

    def get_scatter_data(self):
        scatter = {
            'times': self.spikes['times'][0:-1:100],
            'depths': self.spikes['depths'][0:-1:100]
        }

        return scatter

    
    def get_histology_data(self):
        #acronym = np.flip(self.channels['acronym'])
        acronym = self.channels['acronym']
        colour = self.create_random_hex_colours(np.unique(acronym))  # TODO: ibllib.atlas get colours from Allen

        # Find all boundaries from histology
        #boundaries = np.where(np.diff(np.flip(self.channels.atlas_id)))[0]
        boundaries = np.where(np.diff(self.channels.atlas_id))[0]

        region = []
        region_label = []
        region_colour = []
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

            region_label.append((_region_mean, _region_label))
            region_colour.append(_region_colour)
            region.append(_region)



        region_label.insert(0, (-230, 'extra'))
        region_label.append((4090, 'extra'))
        region_colour.insert(0, '#808080')
        region_colour.append('#808080')
        region.insert(0,[-480, 20])
        region.append([3840, 3840 + 500])

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
        




