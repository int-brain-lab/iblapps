import numpy as np
import ibllib.pipes.histology as histology
import ibllib.atlas as atlas
from oneibl.one import ONE
from mayavi import mlab
from atlaselectrophysiology import rendering

brain_atlas = atlas.AllenAtlas(25)
ONE_BASE_URL = "https://alyx.internationalbrainlab.org"

one = ONE()

all_hist = one.alyx.rest('trajectories', 'list', provenance='Histology track')
# Some do not have tracing, exclude these ones
sess_with_hist = [sess for sess in all_hist if sess['x'] is not None]
trajectories = [atlas.Insertion.from_dict(sess) for sess in sess_with_hist]
traj_ids = [sess['id'] for sess in sess_with_hist]

depths = np.arange(200, 1100, 20) / 1e6

traj_id = sess_with_hist[200]['id']

traj_coords = np.empty((len(traj_ids), len(depths), 3))

for iT, traj in enumerate(trajectories):
    traj_coords[iT, :] = histology.interpolate_along_track(np.vstack([traj.tip, traj.entry]),
                                                           depths)
# traj_coords_mlap = traj_coords[:, :, (0, 1)]
chosen_traj = traj_ids.index(traj_id)
avg_dist = np.mean(np.sqrt(np.sum((traj_coords - traj_coords[chosen_traj]) ** 2, axis=2)), axis=1)

closest_traj = np.argsort(avg_dist)

close_sessions = []
fig = rendering.figure(grid=False)
for iSess, sess_idx in enumerate(closest_traj[0:10]):

    mlapdv = brain_atlas.xyz2ccf(traj_coords[sess_idx])
    if iSess == 0:
        mlab.plot3d(mlapdv[:, 1], mlapdv[:, 2], mlapdv[:, 0],
                    line_width=1, tube_radius=10, color=(0, 0, 0))
    elif iSess < 11:
        mlab.plot3d(mlapdv[:, 1], mlapdv[:, 2], mlapdv[:, 0],
                    line_width=1, tube_radius=10, color=(0.0, 0.4, 0.5))
    else:
        mlab.plot3d(mlapdv[:, 1], mlapdv[:, 2], mlapdv[:, 0],
                    line_width=1, tube_radius=10, color=(0.0, 1, 0.5))

    mlab.text3d(mlapdv[0, 1], mlapdv[0, 2], mlapdv[0, 0], str(iSess),
                line_width=4, color=(0, 0, 0), figure=fig, scale=150)

    close_sessions.append((sess_with_hist[sess_idx]['session']['subject'] + ' ' +
                           sess_with_hist[sess_idx]['session']['start_time'][:10] +
                           ' ' + sess_with_hist[sess_idx]['probe_name']))

print(avg_dist[closest_traj[0:10]] * 1e6)
print(close_sessions)
