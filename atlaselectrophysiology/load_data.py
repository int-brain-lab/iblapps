
import scipy
import numpy as np
import alf.io
from oneibl.one import ONE
# import matplotlib.pyplot as plt
import ibllib.pipes.histology as histology
import ibllib.atlas as atlas

TIP_SIZE_UM = 200
ONE_BASE_URL = "https://dev.alyx.internationalbrainlab.org"
one = ONE(base_url=ONE_BASE_URL)


def _cumulative_distance(xyz):
    return np.cumsum(np.r_[0, np.sqrt(np.sum(np.diff(xyz, axis=0) ** 2, axis=1))])


class LoadData:
    def __init__(self, max_idx):
        self.max_idx = max_idx

    def get_subjects(self):
        """
        Finds all subjects that have a histology track trajectory registered
        :return subjects: list of subjects
        :type subjects: list of strings
        """
        sess_with_hist = one.alyx.rest('trajectories', 'list', provenance='Histology track')
        subjects = [sess['session']['subject'] for sess in sess_with_hist]
        subjects = np.unique(subjects)

        return subjects

    def get_sessions(self, subject):
        """
        Finds all sessions for a particular subject that have a histology track trajectory
        registered
        :param subject: subject name
        :type subject: string
        :return session: list of sessions associated with subject, displayed as date + probe
        :return session: list of strings
        """
        self.subj = subject
        self.sess_with_hist = one.alyx.rest('trajectories', 'list', subject=self.subj,
                                            provenance='Histology track')
        session = [(sess['session']['start_time'][:10] + ' ' + sess['probe_name']) for sess in
                   self.sess_with_hist]
        return session

    def get_info(self, idx):
        """
        """
        self.n_sess = self.sess_with_hist[idx]['session']['number']
        self.date = self.sess_with_hist[idx]['session']['start_time'][:10]
        self.probe_label = self.sess_with_hist[idx]['probe_name']

    def get_eid(self):
        eids = one.search(subject=self.subj, date=self.date, number=self.n_sess,
                          task_protocol='ephys')
        self.eid = eids[0]
        print(self.subj)
        print(self.probe_label)
        print(self.date)
        print(self.eid)

    def get_data(self):
        # Load in all the data required
        dtypes = [
            'spikes.depths',
            'spikes.amps',
            'spikes.times',
            'spikes.clusters',
            'channels.localCoordinates',
            'channels.rawInd',
            'clusters.metrics',
            'clusters.peakToTrough',
            'clusters.waveforms',
            '_iblqc_ephysTimeRms.rms',
            '_iblqc_ephysTimeRms.timestamps',
            '_iblqc_ephysSpectralDensity.freqs',
            '_iblqc_ephysSpectralDensity.amps',
            '_iblqc_ephysSpectralDensity.power'
        ]

        _ = one.load(self.eid, dataset_types=dtypes, download_only=True)
        path = one.path_from_eid(self.eid)
        self.alf_path = path.joinpath('alf', self.probe_label)
        self.ephys_path = path.joinpath('raw_ephys_data', self.probe_label)
        self.chn_coords, self.chn_ind = (alf.io.load_object(self.alf_path, 'channels')).values()

        sess = one.alyx.rest('sessions', 'read', id=self.eid)
        if sess['notes']:
            sess_notes = sess['notes'][0]['text']
        else:
            sess_notes = 'No notes for this session'

        return self.alf_path, self.ephys_path, sess_notes

    def get_probe_track(self):

        self.brain_atlas = atlas.AllenAtlas(res_um=25)

        # Load in user picks for session
        insertion = one.alyx.rest('insertions', 'list', session=self.eid, name=self.probe_label)
        xyz_picks = np.array(insertion[0]['json']['xyz_picks']) / 1e6
        self.probe_id = insertion[0]['id']
        self.get_trajectory()

        # Use the top/bottom 1/4 of picks to compute the entry and exit trajectories of the probe
        n_picks = np.max([4, round(xyz_picks.shape[0] / 4)])
        traj_entry = atlas.Trajectory.fit(xyz_picks[:n_picks, :])
        # Force the entry to be on the upper z lim of the atlas to account for cases where channels
        # may be located above the surface of the brain
        entry = (traj_entry.eval_z(self.brain_atlas.bc.zlim))[0, :]

        traj_exit = atlas.Trajectory.fit(xyz_picks[-1 * n_picks:, :])
        exit = atlas.Insertion.get_brain_exit(traj_exit, self.brain_atlas)
        # exit = (traj_exit.eval_z(self.brain_atlas.bc.zlim))[1, :]
        exit[2] = exit[2] - 200 / 1e6

        self.xyz_track = np.r_[exit[np.newaxis, :], xyz_picks, entry[np.newaxis, :]]
        # by convention the deepest point is first
        self.xyz_track = self.xyz_track[np.argsort(self.xyz_track[:, 2]), :]

        # plot on tilted coronal slice for sanity check
        # ax = self.brain_atlas.plot_tilted_slice(self.xyz_track, axis=1)
        # ax.plot(self.xyz_track[:, 0] * 1e6, self.xyz_track[:, 2] * 1e6, '-*')
        # ax.plot(xyz_picks[:, 0] * 1e6, xyz_picks[:, 2] * 1e6, 'r-*')
        # plt.show()

        self.track = [0] * (self.max_idx + 1)
        self.features = [0] * (self.max_idx + 1)

        # ORIG
        tip_distance = _cumulative_distance(self.xyz_track)[1] + TIP_SIZE_UM / 1e6
        track_length = _cumulative_distance(self.xyz_track)[-1]
        self.track_init = np.array([0, track_length]) - tip_distance

        self.track[0] = np.copy(self.track_init)
        self.features[0] = np.copy(self.track_init)

        self.hist_data = {
            'region': [0] * (self.max_idx + 1),
            'axis_label': [0] * (self.max_idx + 1),
            'colour': [0] * (self.max_idx + 1)
        }
        self.get_histology_regions(0)
        self.scale_histology_regions(0)

        self.scale_data = {
            'region': [0] * (self.max_idx + 1),
            'scale': [0] * (self.max_idx + 1)
        }

        self.get_scale_factor(0)

    def get_trajectory(self):
        self.traj_exists = False
        ephys_traj = one.alyx.rest('trajectories', 'list', probe_insertion=self.probe_id,
                                   provenance='Ephys aligned histology track')
        if len(ephys_traj):
            self.traj_exists = True

    def feature2track_lin(self, trk, idx):
        if self.features[idx].size >= 5:
            fcn_lin = np.poly1d(np.polyfit(self.features[idx][1:-1], self.track[idx][1:-1], 1))
            lin_fit = fcn_lin(trk)
        else:
            lin_fit = 0
            fcn_lin = 0
        return lin_fit

    def feature2track(self, trk, idx):

        fcn = scipy.interpolate.interp1d(self.features[idx], self.track[idx],
                                         fill_value="extrapolate")
        return fcn(trk)

    def track2feature(self, ft, idx):

        fcn = scipy.interpolate.interp1d(self.track[idx], self.features[idx],
                                         fill_value="extrapolate")
        return fcn(ft)

    def get_channels_coordinates(self, idx, depths=None):
        """
        Gets 3d coordinates from a depth along the electrophysiology feature. 2 steps
        1) interpolate from the electrophys features depths space to the probe depth space
        2) interpolate from the probe depth space to the true 3D coordinates
        if depths is not provided, defaults to channels local coordinates depths
        """
        if depths is None:
            depths = self.chn_coords[:, 1] / 1e6
        # nb using scipy here so we can change to cubic spline if needed
        channel_depths_track = self.feature2track(depths, idx) - self.track_init[0]
        self.xyz_channels = histology.interpolate_along_track(self.xyz_track, channel_depths_track)
        return self.xyz_channels

    def upload_channels(self, overwrite=False):
        insertion = atlas.Insertion.from_track(self.xyz_channels, self.brain_atlas)
        # NEEED TO ADD TIP TO DEPTH?
        brain_regions = self.brain_atlas.regions.get(self.brain_atlas.get_labels
                                                     (self.xyz_channels))
        brain_regions['xyz'] = self.xyz_channels
        brain_regions['lateral'] = self.chn_coords[:, 0]
        brain_regions['axial'] = self.chn_coords[:, 1]
        assert np.unique([len(brain_regions[k]) for k in brain_regions]).size == 1
        histology.register_aligned_track(self.probe_id, insertion, brain_regions, one=one,
                                         overwrite=overwrite)

    def scale_histology_regions(self, idx):

        region_label = np.copy(self.region_label)
        region = self.track2feature(self.region, idx) * 1e6
        region_label[:, 0] = (self.track2feature(np.float64(region_label[:, 0]), idx) * 1e6)

        self.hist_data['region'][idx] = region
        self.hist_data['axis_label'][idx] = region_label
        self.hist_data['colour'][idx] = self.region_colour

    def get_histology_regions(self, idx):
        """
        Samples at 10um along the trajectory
        :return:
        """
        sampling_trk = np.arange(self.track_init[0],
                                 self.track_init[-1] - 10 * 1e-6, 10 * 1e-6)
        xyz_samples = histology.interpolate_along_track(self.xyz_track,
                                                        sampling_trk - sampling_trk[0])

        region_ids = self.brain_atlas.get_labels(xyz_samples)
        region_info = self.brain_atlas.regions.get(region_ids)
        boundaries = np.where(np.diff(region_info.id))[0]
        self.region = np.empty((boundaries.size + 1, 2))
        self.region_label = np.empty((boundaries.size + 1, 2), dtype=object)
        self.region_colour = np.empty((boundaries.size + 1, 3), dtype=int)

        for bound in np.arange(boundaries.size + 1):
            if bound == 0:
                _region = np.array([0, boundaries[bound]])
            elif bound == boundaries.size:
                _region = np.array([boundaries[bound - 1], region_info.id.size - 1])
            else:
                _region = np.array([boundaries[bound - 1], boundaries[bound]])

            _region_colour = region_info.rgb[_region[1]]
            _region_label = region_info.acronym[_region[1]]
            _region = sampling_trk[_region]
            _region_mean = np.mean(_region)

            self.region[bound, :] = _region
            self.region_colour[bound, :] = _region_colour
            self.region_label[bound, :] = (_region_mean, _region_label)

    def get_scale_factor(self, idx):
        scale = []
        for iR, (reg, reg_orig) in enumerate(zip(self.hist_data['region'][idx],
                                                 self.region * 1e6)):
            scale = np.r_[scale, (reg[1] - reg[0]) / (reg_orig[1] - reg_orig[0])]

        boundaries = np.where(np.diff(np.around(scale, 3)))[0]
        if boundaries.size == 0:
            region = np.array([[self.hist_data['region'][idx][0][0],
                               self.hist_data['region'][idx][-1][1]]])
            region_scale = np.array([1])
        else:

            region = np.empty((boundaries.size + 1, 2))
            region_scale = []
            for bound in np.arange(boundaries.size + 1):
                if bound == 0:
                    _region = np.array([self.hist_data['region'][idx][0][0],
                                       self.hist_data['region'][idx][boundaries[bound]][1]])
                    _region_scale = scale[0]
                elif bound == boundaries.size:
                    _region = np.array([self.hist_data['region'][idx][boundaries[bound - 1]][1],
                                       self.hist_data['region'][idx][-1][1]])
                    _region_scale = scale[-1]
                else:
                    _region = np.array([self.hist_data['region'][idx][boundaries[bound - 1]][1],
                                        self.hist_data['region'][idx][boundaries[bound]][1]])
                    _region_scale = scale[boundaries[bound]]

                region[bound, :] = _region
                region_scale = np.r_[region_scale, _region_scale]

        self.scale_data['region'][idx] = region
        self.scale_data['scale'][idx] = region_scale
