import numpy as np

from brainbox.io.one import load_spike_sorting
from oneibl.one import ONE

one = ONE(base_url="https://dev.alyx.internationalbrainlab.org")

eids = one.search(subject='ZM_2407', task_protocol='ephys')

#
ses = one.alyx.rest('sessions', 'read', id=eids[0])
iprobe = 0
_channels = ses['probe_insertion'][iprobe]['trajectory_estimate'][0]['channels']
channels = {
    'atlas_id': np.array([ch['brain_region']['id'] for ch in _channels]),
    'acronym': np.array([ch['brain_region']['acronym'] for ch in _channels]),
    'x': np.array([ch['x'] for ch in _channels]) / 1e6,
    'y': np.array([ch['y'] for ch in _channels]) / 1e6,
    'z': np.array([ch['z'] for ch in _channels]) / 1e6,
}

spikes, clusters = load_spike_sorting(eids[0], one=one)
