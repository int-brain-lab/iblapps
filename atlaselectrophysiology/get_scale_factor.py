'''
Extract channel locations from reference points of previous alignments saved in json field of
trajectory object
Create plot showing histology regions which channels pass through as well as coronal slice with
channel locations shown
'''

# import modules
from oneibl.one import ONE
from ibllib.pipes.ephys_alignment import EphysAlignment
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ibllib.atlas as atlas


# Instantiate brain atlas and one
brain_atlas = atlas.AllenAtlas(25)
one = ONE()

# Find eid of interest
subject = 'CSH_ZAD_001'

# Find the ephys aligned trajectory for eid probe combination
trajectory = one.alyx.rest('trajectories', 'list', provenance='Ephys aligned histology track',
                           subject=subject)

# Load in channels.localCoordinates dataset type
chn_coords = one.load(trajectory[0]['session']['id'],
                      dataset_types=['channels.localCoordinates'])[0]
depths = chn_coords[:, 1]

subject_summary = pd.DataFrame(columns={'Session', 'User', 'Scale Factor', 'Avg Scale Factor'})
sf = []
sess = []
user = []
for traj in trajectory:
    alignments = traj['json']

    insertion = one.alyx.rest('insertions', 'list', session=traj['session']['id'],
                              name=traj['probe_name'])
    xyz_picks = np.array(insertion[0]['json']['xyz_picks']) / 1e6

    session_info = traj['session']['start_time'][:10] + '_' + traj['probe_name']

    for iK, key in enumerate(alignments):
        # Location of reference lines used for alignmnet
        feature = np.array(alignments[key][0])
        track = np.array(alignments[key][1])
        user = key[20:]
        # Instantiate EphysAlignment object
        ephysalign = EphysAlignment(xyz_picks, depths, track_prev=track, feature_prev=feature)
        region_scaled, _ = ephysalign.scale_histology_regions(feature, track)
        _, scale_factor = ephysalign.get_scale_factor(region_scaled)

        if np.all(np.round(np.diff(scale_factor), 3) == 0):
            scale_factor = np.array([1])
            avg_sf = 1
        else:
            if feature.size > 4:
                avg_sf = scale_factor[0]
            else:
                avg_sf = np.mean(scale_factor[1:-1])

        for iS, sf in enumerate(scale_factor):
            if iS == 0:
                subject_summary = subject_summary.append(
                    {'Session': session_info, 'User': user, 'Scale Factor': sf,
                     'Avg Scale Factor': avg_sf}, ignore_index=True)
            else:
                subject_summary = subject_summary.append(
                    {'Session': session_info, 'User': user, 'Scale Factor': sf,
                     'Avg Scale Factor': np.NAN}, ignore_index=True)

fig, ax = plt.subplots()
s1 = sns.swarmplot(x='Session', y='Scale Factor', hue='User', data=subject_summary, ax=ax)
s1.legend_.remove()
s2 = sns.swarmplot(x='Session', y='Avg Scale Factor', hue='User', size=8, linewidth=1,
                  data=subject_summary, ax=ax)
plt.show()
