from oneibl.one import ONE
from atlaselectrophysiology.load_data import LoadData
from ibllib.pipes.ephys_alignment import EphysAlignment
from ibllib.pipes.misc import create_alyx_probe_insertions
from ibllib.pipes.histology import register_track
import numpy as np

one = ONE(username='test_user', password='TapetesBloc18',
               base_url='https://test.alyx.internationalbrainlab.org')


cluster_chns = np.load('iblapps/atlaselectrophysiology/clusters.channels.npy')
alignments_stored = {'2020-07-26T17:06:58_alejandro': [[-1.0016980364099322,
   0.00047877496991576273,
   0.001330986933492519,
   0.002099550451500421,
   0.002601841155234657,
   0.0030670974729241885,
   0.003141966305655836,
   1.0051440753904748],
  [-1.153578623891749,
   0.0006501531566169838,
   0.001548329723225031,
   0.002287397004769017,
   0.0029173598074608924,
   0.0035415451180881596,
   0.003815011822220497,
   1.1575841141993106]],
 '2020-09-14T15:44:56_nate': [[-1.0016980364099322,
   8.609448818897726e-05,
   0.0018834173228346462,
   0.003100188976377953,
   0.0032035748031496065,
   1.0051440753904748],
  [-1.1355431294812974,
   0.000221118646247076,
   0.0021617637795274813,
   0.003545543307086614,
   0.0038238897637795266,
   1.1396363475848257]],
 '2020-06-12T11:33:02_guido': [[-1.0016980364099322,
   0.00047877496991576273,
   0.0012435066185318888,
   0.0018371095066185312,
   0.002601841155234657,
   0.0030670974729241885,
   0.003141966305655836,
   1.0051440753904748],
  [-1.138461638877023,
   0.0006501531566169838,
   0.001548329723225031,
   0.0021740192539109497,
   0.0029173598074608924,
   0.0035415451180881596,
   0.003815011822220497,
   1.1425738337640625]],
 '2020-09-14T15:42:22_guido': [[-1.0016980364099322,
   0.00047877496991576273,
   0.0012435066185318888,
   0.0018371095066185312,
   0.002601841155234657,
   0.0030670974729241885,
   0.0032064811858881124,
   1.0051440753904748],
  [-1.125796592594711,
   0.0006501531566169838,
   0.001548329723225031,
   0.0021740192539109497,
   0.0029173598074608924,
   0.0035415451180881596,
   0.003815011822220497,
   1.1298931711501976]]}


xyz_picks = np.array([[-2188, -2175, -118],
                     [-2163, -2175, -193],
                     [-2138, -2199, -292],
                     [-2113, -2224, -367],
                     [-2138, -2250, -443],
                     [-2113, -2250, -543],
                     [-2113, -2250, -618],
                     [-2113, -2250, -718],
                     [-2088, -2300, -793],
                     [-2063, -2300, -868],
                     [-2063, -2300, -942],
                     [-2063, -2300, -1017],
                     [-2038, -2400, -1068],
                     [-1988, -2400, -1092],
                     [-1963, -2400, -1193],
                     [-1963, -2400, -1268],
                     [-1938, -2400, -1367],
                     [-1938, -2400, -1468],
                     [-1938, -2400, -1568],
                     [-1913, -2450, -1668],
                     [-1913, -2450, -1743],
                     [-1888, -2450, -1843],
                     [-1863, -2450, -1942],
                     [-1863, -2450, -1993],
                     [-1863, -2475, -2093],
                     [-1838, -2475, -2193],
                     [-1813, -2475, -2293],
                     [-1788, -2475, -2392],
                     [-1764, -2475, -2467],
                     [-1738, -2475, -2568],
                     [-1713, -2499, -2643],
                     [-1688, -2525, -2668],
                     [-1688, -2525, -2718],
                     [-1688, -2525, -2768],
                     [-1663, -2549, -2843],
                     [-1638, -2549, -2918],
                     [-1614, -2549, -2967],
                     [-1513, -2600, -3343],
                     [-1513, -2575, -3392],
                     [-1488, -2600, -3418],
                     [-1464, -2624, -3493],
                     [-1438, -2624, -3568],
                     [-1413, -2624, -3617],
                     [-1413, -2624, -3668],
                     [-1388, -2650, -3792],
                     [-1363, -2700, -3868],
                     [-1314, -2700, -4018],
                     [-1288, -2700, -4067],
                     [-1288, -2700, -4118],
                     [-1238, -2700, -4242],
                     [-1213, -2724, -4367],
                     [-1213, -2724, -4392],
                     [-1189, -2750, -4468],
                     [-1164, -2724, -4543],
                     [-1113, -2724, -4618]]) / 1e6


#traj = one.alyx.rest('trajectories', 'create', data=trajectory_data)
#traj_id = traj['id']

# Now we have our test datasets we can do some stuff

#build it up from no alignments to 4 alignments

session = 'b1c968ad-4874-468d-b2e4-5ffa9b9964e9'
model = '3B2'
probe = ['probe00']

create_alyx_probe_insertions(session_path=session, model=model, labels=probe, one=one, force=True)
insertion = one.alyx.rest('insertions', 'list', session=session, name=probe[0])[0]
assert(insertion['json']['qc'] == 'NOT_SET')
assert(len(insertion['json']['extended_qc']) == 0)

probe_id = insertion['id']
# Do this at the end
# register_track(probe_id, picks=None, one=one, overwrite=True, channels=False)
# assert(insertion['json']['qc'] == 'CRITICAL')
# assert(insertion['json']['extended_qc']['_tracing_exists'] == 0)

# TODO remember to remove histology track on teardown

register_track(probe_id, picks=xyz_picks, one=one, overwrite=True, channels=False)
insertion = one.alyx.rest('insertions', 'read', id=probe_id)
# TODO figure this one out
#assert(np.all(np.ravel(np.array(insertion['json']['xyz_picks'])) == np.ravel((xyz_picks * 1e6))))
assert(insertion['json']['qc'] == 'NOT_SET')
assert(insertion['json']['extended_qc']['_tracing_exists'] == 1)


# Now lets look at the alignments
ld = LoadData(one=one, testing=True, probe_id=probe_id)
_ = ld.get_xyzpicks()
ld.cluster_chns = cluster_chns
prev_align = ld.get_previous_alignments()
assert(len(prev_align) == 1)
assert(prev_align[0] == 'original')
feature, track = ld.get_starting_alignment(0)
assert(not feature)
assert(not track)
assert(not ld.alignments)
assert(ld.resolved == 0)

# Now add an alignment
ephysalign = EphysAlignment(ld.xyz_picks, ld.chn_depths, brain_atlas=ld.brain_atlas)
key1 = '2020-07-26T17:06:58_alejandro'
feature = alignments_stored[key1][0]
track = alignments_stored[key1][1]
xyz_channels = ephysalign.get_channel_locations(feature, track)
ld.upload_data(xyz_channels, channels=False)
ld.update_alignments(np.array(feature), np.array(track), key_info=key1)
prev_align = ld.get_previous_alignments()
_ = ld.get_starting_alignment(0)

traj = one.alyx.rest('trajectories', 'list', probe_id=probe_id,
                     provenance='Ephys aligned histology track')
prev_traj_id = traj[0]['id']
assert(ld.current_align == key1)
assert(len(traj[0]['json']) == 1)
ld.update_qc()
insertion = one.alyx.rest('insertions', 'read', id=probe_id)
assert(insertion['json']['extended_qc']['_alignment_number'] == 1)
assert(insertion['json']['extended_qc']['_alignment_stored'] == key1)
assert(insertion['json']['qc'] == 'NOT_SET')
assert(ld.resolved == 0)

key2 = '2020-08-26T17:06:58_alejandro'
ld.upload_data(xyz_channels, channels=False)
ld.update_alignments(np.array(feature), np.array(track), key_info=key2)
prev_align = ld.get_previous_alignments()
_ = ld.get_starting_alignment(0)
traj = one.alyx.rest('trajectories', 'list', probe_id=probe_id,
                     provenance='Ephys aligned histology track')
traj_id = traj[0]['id']
assert(ld.current_align == key2)
assert(len(traj[0]['json'] ) == 1)
ld.update_qc()
insertion = one.alyx.rest('insertions', 'read', id=probe_id)
assert(insertion['json']['extended_qc']['_alignment_number'] == 1)
assert(insertion['json']['extended_qc']['_alignment_stored'] == key2)
assert(insertion['json']['qc'] == 'NOT_SET')
assert(ld.resolved == 0)

assert(traj_id != prev_traj_id)
prev_traj_id = traj_id


key3 = '2020-09-14T15:42:22_guido'
feature = alignments_stored[key3][0]
track = alignments_stored[key3][1]
xyz_channels = ephysalign.get_channel_locations(feature, track)
ld.upload_data(xyz_channels, channels=False)
ld.update_alignments(np.array(feature), np.array(track), key_info=key3)
prev_align = ld.get_previous_alignments()
_ = ld.get_starting_alignment(0)

assert(ld.current_align == key3)

traj = one.alyx.rest('trajectories', 'list', probe_id=probe_id,
                     provenance='Ephys aligned histology track')
traj_id = traj[0]['id']
assert(len(traj[0]['json']) == 2)
# Also assert all the keys match
assert(traj_id != prev_traj_id)
prev_traj_id = traj_id

ld.update_qc()
insertion = one.alyx.rest('insertions', 'read', id=probe_id)
assert(insertion['json']['qc'] == 'WARNING')
assert(insertion['json']['extended_qc']['_alignment_stored'] == key3)
assert(insertion['json']['extended_qc']['_alignment_resolved'] == 0)
assert(insertion['json']['extended_qc']['_alignment_qc'] < 0.8)
assert(ld.resolved == 0)

# Now let's add another one
key4 = '2020-09-14T15:44:56_nate'
feature = alignments_stored[key4][0]
track = alignments_stored[key4][1]
xyz_channels = ephysalign.get_channel_locations(feature, track)
ld.upload_data(xyz_channels, channels=False)
ld.update_alignments(np.array(feature), np.array(track), key_info=key4)
prev_align = ld.get_previous_alignments()
_ = ld.get_starting_alignment(0)

assert(ld.current_align == key4)

traj = one.alyx.rest('trajectories', 'list', probe_id=probe_id,
                     provenance='Ephys aligned histology track')
traj_id = traj[0]['id']
assert(len(traj[0]['json']) == 3)
assert(traj_id != prev_traj_id)
prev_traj_id = traj_id

ld.update_qc()
insertion = one.alyx.rest('insertions', 'read', id=probe_id)
assert(insertion['json']['qc'] == 'PASS')
assert(insertion['json']['extended_qc']['_alignment_stored'] == key4)
assert(insertion['json']['extended_qc']['_alignment_resolved'] == 1)
assert(insertion['json']['extended_qc']['_alignment_qc'] > 0.8)


# Now try to add an extra alignment
key5 = '2020-09-16T15:44:56_mayo'
ld.upload_data(xyz_channels, channels=False)
ld.update_alignments(np.array(feature), np.array(track), key_info=key5)
prev_align = ld.get_previous_alignments()
_ = ld.get_starting_alignment(0)
traj = one.alyx.rest('trajectories', 'list', probe_id=probe_id,
                     provenance='Ephys aligned histology track')
traj_id = traj[0]['id']
assert(len(traj[0]['json']) == 4)
# insertion should not have changed
assert(traj_id == prev_traj_id)

ld.update_qc()
insertion = one.alyx.rest('insertions', 'read', id=probe_id)
assert(insertion['json']['qc'] == 'PASS')
assert(insertion['json']['extended_qc']['_alignment_stored'] == key4)
assert(insertion['json']['extended_qc']['_alignment_resolved'] == 1)
assert(insertion['json']['extended_qc']['_alignment_number'] == 4)

# Now try to add an extra alignment with same user name
key6 = '2020-10-14T15:44:56_nate'
ld.upload_data(xyz_channels, channels=False)
ld.update_alignments(np.array(feature), np.array(track), key_info=key6)
prev_align = ld.get_previous_alignments()
_ = ld.get_starting_alignment(0)

assert(ld.current_align == key6)

traj = one.alyx.rest('trajectories', 'list', probe_id=probe_id,
                     provenance='Ephys aligned histology track')
traj_id = traj[0]['id']
assert(len(traj[0]['json']) == 5)
assert(traj_id == prev_traj_id)

ld.update_qc()
insertion = one.alyx.rest('insertions', 'read', id=probe_id)
assert(insertion['json']['qc'] == 'PASS')
assert(insertion['json']['extended_qc']['_alignment_stored'] == key4)
assert(insertion['json']['extended_qc']['_alignment_resolved'] == 1)
assert(insertion['json']['extended_qc']['_alignment_number'] == 5)

# Now we need to look into deletions



# next look at case where the alignments are already there

# Now delete anyways and strip down