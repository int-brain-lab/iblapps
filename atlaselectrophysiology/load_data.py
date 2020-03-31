
import numpy as np
from brainbox.io.one import load_spike_sorting
from oneibl.one import ONE
import random

def get_session(subj, date, sess=None):
    if not sess:
        sess = 1
    one = ONE(base_url="https://dev.alyx.internationalbrainlab.org")
    eids = one.search(subject=subj, date=date, number=sess, task_protocol='ephys')
    eid = eids[0]

    return eid, one


def get_scatter_data(eid, one=None, probe_id=None):
    if not one:
        one = ONE(base_url="https://dev.alyx.internationalbrainlab.org")

    if not probe_id:
        probe_id = 0
       
    probe = 'probe_0' + str(probe_id)
 
    #Extra datasets that we want to download
    dtypes_extra = [
        'spikes.depths',
        'channels.localCoordinates'
    ]

    spikes, _ = load_spike_sorting(eid=eid, one=one, dataset_types=dtypes_extra)

    scatter = {
        'times': spikes[probe]['times'],
        'depths': spikes[probe]['depths']
    }

    return scatter


def get_histology_data(eid, one=None, probe_id=None):
    if not one:
        one = ONE(base_url="https://dev.alyx.internationalbrainlab.org")
 
    if not probe_id:
        probe_id = 0

    ses = one.alyx.rest('sessions', 'read', id=eid)
    te = [te for te in ses['probe_insertion'][probe_id]['trajectory_estimate']
              if te['provenance'] == 'Histology track']
    _channels = te[0]['channels']
    channels = {
        'atlas_id': np.array([ch['brain_region']['id'] for ch in _channels]),
        'acronym': np.array([ch['brain_region']['acronym'] for ch in _channels]),
        'x': np.array([ch['x'] for ch in _channels]) / 1e6,
        'y': np.array([ch['y'] for ch in _channels]) / 1e6,
        'z': np.array([ch['z'] for ch in _channels]) / 1e6,
    }

    #Load coordinates of channels
    channel_coord = one.load_dataset(eid=eid, dataset_type='channels.localCoordinates')
    acronym = np.flip(channels['acronym'])
    
    colour = create_random_hex_colours(np.unique(acronym))

    #Find all boundaries from histology
    boundaries = np.where((acronym[1:] == acronym[:-1]) == False)[0]
  
    region = []
    region_label = []
    region_colour = []
    for idx in range(len(boundaries) + 1):
        if idx == 0:
            _region = [0, boundaries[idx]]
        elif idx == len(boundaries):
            _region = [boundaries[idx - 1], len(channel_coord) - 1]
        else: 
            _region = [boundaries[idx - 1], boundaries[idx]]

        _region_label = acronym[_region][1]
        _region_colour = colour[_region_label]
        _region = channel_coord[:, 1][_region]
        _region_mean = np.mean(_region, dtype=int)

        region_label.append((_region_mean, _region_label))
        region_colour.append(_region_colour)
        region.append(_region)

    histology = {
        'boundary': region,
        'boundary_label': region_label,
        'boundary_colour': region_colour,
        'acronym': acronym,
        'chan_int': 20
    }

    return histology


def create_random_hex_colours(unique_regions):

    colour = {}
    random.seed(8)
    for reg in unique_regions:
        colour[reg] = "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])

    return colour
