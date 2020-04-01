
import numpy as np
from brainbox.io.one import load_spike_sorting, load_channel_locations
from oneibl.one import ONE
import random

ONE_BASE_URL = "https://dev.alyx.internationalbrainlab.org"


def get_session(subj, date, sess=None):
    if not sess:
        sess = 1
    one = ONE(base_url=ONE_BASE_URL)
    eids = one.search(subject=subj, date=date, number=sess, task_protocol='ephys')
    eid = eids[0]

    return eid, one


def get_scatter_data(eid, one=None, probe_id=None):
    if not one:
        one = ONE(base_url=ONE_BASE_URL)

    if not probe_id:
        probe_id = 0

    #Extra datasets that we want to download
    dtypes_extra = [
        'spikes.depths',
        'channels.localCoordinates'
    ]

    spikes, _ = load_spike_sorting(eid=eid, one=one, dataset_types=dtypes_extra)

    probe_label = [key for key in spikes.keys() if int(key[-1]) == probe_id][0]
    
    scatter = {
        'times': spikes[probe_label]['times'][0:-1:100],
        'depths': spikes[probe_label]['depths'][0:-1:100]
    }



    return scatter, probe_label


def get_histology_data(eid, probe_label, one=None):
    if not one:
        one = ONE(base_url=ONE_BASE_URL)

    ses = one.alyx.rest('sessions', 'read', id=eid)
    #probe_label = ses['probe_insertion'][0]['name'] 

    channels = load_channel_locations(ses, one=one, probe=probe_label)[probe_label]

    # Load coordinates of channels
    channel_coord = one.load_dataset(eid=eid, dataset_type='channels.localCoordinates')
    assert np.all( np.c_[channels.lateral_um, channels.axial_um ] == channel_coord)

    acronym = np.flip(channels['acronym'])
    colour = create_random_hex_colours(np.unique(acronym))  # TODO: ibllib.atlas get colours from Allen

    # Find all boundaries from histology
    boundaries = np.where(np.diff(np.flip(channels.atlas_id)))[0]
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
