import scipy
import numpy as np
from datetime import datetime
import ibllib.pipes.histology as histology
import ibllib.atlas as atlas
from oneibl.one import ONE
from atlaselectrophysiology.load_histology import download_histology_data
brain_atlas = atlas.AllenAtlas(25)
TIP_SIZE_UM = 200
ONE_BASE_URL = "https://dev.alyx.internationalbrainlab.org"
one = ONE(base_url=ONE_BASE_URL)

def _cumulative_distance(xyz):
    return np.cumsum(np.r_[0, np.sqrt(np.sum(np.diff(xyz, axis=0) ** 2, axis=1))])

class EphysAlignment:

    def __init__(self, xyz_track, chn_depths, track_extent, track_init,
                 feature_init, probe_id=None):

        self.xyz_track = xyz_track
        self.chn_depths = chn_depths
        self.track_extent = track_extent
        self.track_init = track_init
        self.feature_init = feature_init
        self.probe_id = probe_id
        self.region, self.region_label, self.region_colour = self.get_histology_regions()

    def get_track_and_feature(self):
        return self.feature_init, self.track_init, self.xyz_track

    @staticmethod
    def feature2track(trk, feature, track):
        fcn = scipy.interpolate.interp1d(feature, track, fill_value="extrapolate")
        return fcn(trk)

    @staticmethod
    def track2feature(ft, feature, track):
        fcn = scipy.interpolate.interp1d(track, feature, fill_value="extrapolate")
        return fcn(ft)

    @staticmethod
    def feature2track_lin(trk, feature, track):
        if feature.size >= 5:
            fcn_lin = np.poly1d(np.polyfit(feature[1:-1], track[1:-1], 1))
            lin_fit = fcn_lin(trk)
        else:
            lin_fit = 0
        return lin_fit

    @staticmethod
    def adjust_extremes_uniform(feature, track):
        diff = np.diff(feature - track)
        track[0] -= diff[0]
        track[-1] += diff[-1]
        return track

    def adjust_extremes_linear(self, feature, track, extend_feature):
        feature[0] = self.track_extent[0] - extend_feature
        feature[-1] = self.track_extent[-1] + extend_feature
        extend_track = self.feature2track_lin(feature[[0, -1]], feature, track)
        track[0] = extend_track[0]
        track[-1] = extend_track[-1]
        return feature, track

    def scale_histology_regions(self, feature, track):
        region_label = np.copy(self.region_label)
        region = self.track2feature(self.region, feature, track) * 1e6
        region_label[:, 0] = (self.track2feature(np.float64(region_label[:, 0]), feature, track) * 1e6)
        return region, region_label

    def get_histology_regions(self):
        sampling_trk = np.arange(self.track_extent[0],
                                 self.track_extent[-1] - 10 * 1e-6, 10 * 1e-6)
        xyz_samples = histology.interpolate_along_track(self.xyz_track,
                                                        sampling_trk - sampling_trk[0])
        region_ids = brain_atlas.get_labels(xyz_samples)
        region_info = brain_atlas.regions.get(region_ids)
        boundaries = np.where(np.diff(region_info.id))[0]
        region = np.empty((boundaries.size + 1, 2))
        region_label = np.empty((boundaries.size + 1, 2), dtype=object)
        region_colour = np.empty((boundaries.size + 1, 3), dtype=int)
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
            region[bound, :] = _region
            region_colour[bound, :] = _region_colour
            region_label[bound, :] = (_region_mean, _region_label)

        return region, region_label, region_colour

    def get_scale_factor(self, region):
        scale = []
        for iR, (reg, reg_orig) in enumerate(zip(region, self.region * 1e6)):
            scale = np.r_[scale, (reg[1] - reg[0]) / (reg_orig[1] - reg_orig[0])]
        boundaries = np.where(np.diff(np.around(scale, 3)))[0]
        if boundaries.size == 0:
            scaled_region = np.array([[region[0][0], region[-1][1]]])
            scale_factor = np.unique(scale)
        else:
            scaled_region = np.empty((boundaries.size + 1, 2))
            scale_factor = []
            for bound in np.arange(boundaries.size + 1):
                if bound == 0:
                    _scaled_region = np.array([region[0][0],
                                        region[boundaries[bound]][1]])
                    _scale_factor = scale[0]
                elif bound == boundaries.size:
                    _scaled_region = np.array([region[boundaries[bound - 1]][1],
                                        region[-1][1]])
                    _scale_factor = scale[-1]
                else:
                    _scaled_region = np.array([region[boundaries[bound - 1]][1],
                                        region[boundaries[bound]][1]])
                    _scale_factor = scale[boundaries[bound]]
                scaled_region[bound, :] = _scaled_region
                scale_factor = np.r_[scale_factor, _scale_factor]
        return scaled_region, scale_factor

    def get_channel_locations(self, feature, track, depths=None):
        """
        Gets 3d coordinates from a depth along the electrophysiology feature. 2 steps
        1) interpolate from the electrophys features depths space to the probe depth space
        2) interpolate from the probe depth space to the true 3D coordinates
        if depths is not provided, defaults to channels local coordinates depths
        """
        if depths is None:
            depths = self.chn_depths / 1e6
        # nb using scipy here so we can change to cubic spline if needed
        channel_depths_track = self.feature2track(depths, feature, track) - self.track_extent[0]
        xyz_channels = histology.interpolate_along_track(self.xyz_track, channel_depths_track)
        return xyz_channels

    def get_brain_locations(self, xyz_channels):
        brain_regions = brain_atlas.regions.get(brain_atlas.get_labels(xyz_channels))
        return brain_regions

    def get_perp_vector(self, feature_lines, feature, track):

        slice_lines = []
        for line in feature_lines[1:-1]:
            depths = np.array([line, line + 10 / 1e6])
            xyz = self.get_channel_locations(feature, track, depths)

            extent = 500e-6
            vector = np.diff(xyz, axis=0)[0]
            point = xyz[0, :]
            vector_perp = np.array([1, 0, -1 * vector[0] / vector[2]])
            xyz_per = np.r_[[point + (-1 * extent * vector_perp)],
                        [point + (extent * vector_perp)]]
            slice_lines.append(xyz_per)

        return slice_lines







class EphysAlignmentFromAlyx(EphysAlignment):

    def __init__(self, eid, probe_label, chn_depths, use_previous=False, track_previous=None, feature_previous=None):

        # Load in user picks for session
        insertion = one.alyx.rest('insertions', 'list', session=eid, name=probe_label)
        xyz_picks = np.array(insertion[0]['json']['xyz_picks']) / 1e6
        probe_id = insertion[0]['id']

        # Use the top/bottom 1/4 of picks to compute the entry and exit trajectories of the probe
        n_picks = np.max([4, round(xyz_picks.shape[0] / 4)])
        traj_entry = atlas.Trajectory.fit(xyz_picks[:n_picks, :])
        traj_exit = atlas.Trajectory.fit(xyz_picks[-1 * n_picks:, :])
        # Force the entry to be on the upper z lim of the atlas to account for cases where channels
        # may be located above the surface of the brain
        entry = (traj_entry.eval_z(brain_atlas.bc.zlim))[0, :]
        exit = atlas.Insertion.get_brain_exit(traj_exit, brain_atlas)
        exit[2] = exit[2] - 200 / 1e6

        xyz_track = np.r_[exit[np.newaxis, :], xyz_picks, entry[np.newaxis, :]]
        # by convention the deepest point is first
        xyz_track = xyz_track[np.argsort(xyz_track[:, 2]), :]

        tip_distance = _cumulative_distance(xyz_track)[1] + TIP_SIZE_UM / 1e6
        track_length = _cumulative_distance(xyz_track)[-1]
        track_extent = np.array([0, track_length]) - tip_distance

        if use_previous:
            track_init = track_previous
            feature_init = feature_previous
        else:
            track_init = np.copy(track_extent)
            feature_init = np.copy(track_extent)

        super().__init__(xyz_track, chn_depths, track_extent, track_init, feature_init, probe_id)



class EphysAlignmentFromLocal(EphysAlignment):

    def __init__(self, xyz_picks, chn_depths, use_previous=False, track_previous=None, feature_previous=None):

        # Use the top/bottom 1/4 of picks to compute the entry and exit trajectories of the probe
        n_picks = np.max([4, round(xyz_picks.shape[0] / 4)])
        traj_entry = atlas.Trajectory.fit(xyz_picks[:n_picks, :])
        traj_exit = atlas.Trajectory.fit(xyz_picks[-1 * n_picks:, :])
        # Force the entry to be on the upper z lim of the atlas to account for cases where channels
        # may be located above the surface of the brain
        entry = (traj_entry.eval_z(brain_atlas.bc.zlim))[0, :]
        exit = atlas.Insertion.get_brain_exit(traj_exit, brain_atlas)
        exit[2] = exit[2] - 200 / 1e6

        xyz_track = np.r_[exit[np.newaxis, :], xyz_picks, entry[np.newaxis, :]]
        # by convention the deepest point is first
        xyz_track = xyz_track[np.argsort(xyz_track[:, 2]), :]

        tip_distance = _cumulative_distance(xyz_track)[1] + TIP_SIZE_UM / 1e6
        track_length = _cumulative_distance(xyz_track)[-1]
        track_extent = np.array([0, track_length]) - tip_distance

        if use_previous:
            track_init = track_previous
            feature_init = feature_previous
        else:
            track_init = np.copy(track_extent)
            feature_init = np.copy(track_extent)

        super().__init__(xyz_track, chn_depths, track_extent, track_init, feature_init, probe_id=None)



class LoadData:
    def __init__(self):
        self.eid = []
        self.lab = []
        self.n_sess = []
        self.probe_label = []
        self.probe_id = []
        self.date = []
        self.subj = []
        self.chn_coords = []

    def get_subjects(self):
        """
        Finds all subjects that have a histology track trajectory registered
        :return subjects: list of subjects
        :type subjects: list of strings
        """
        sess_with_hist = one.alyx.rest('trajectories', 'list', provenance='Histology track')
        subjects = [sess['session']['subject'] for sess in sess_with_hist]
        self.subjects = np.unique(subjects)

        return self.subjects

    def get_sessions(self, idx):
        """
        Finds all sessions for a particular subject that have a histology track trajectory
        registered
        :param subject: subject name
        :type subject: string
        :return session: list of sessions associated with subject, displayed as date + probe
        :return session: list of strings
        """
        self.subj = self.subjects[idx]
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
        self.probe_id = self.sess_with_hist[idx]['probe_insertion']
        self.lab = self.sess_with_hist[idx]['session']['lab']

        ephys_traj_prev = one.alyx.rest('trajectories', 'list', probe_insertion=self.probe_id,
                                        provenance='Ephys aligned histology track')

        if ephys_traj_prev:
            self.alignments = ephys_traj_prev[0]['json']
            self.prev_align = []
            if self.alignments:
                self.prev_align = [*self.alignments.keys()]
            # To make sure they are ordered by date added, default to latest fit
            self.prev_align.reverse()
            self.prev_align.append('original')
        else:
            self.prev_align = ['original']

        return self.prev_align

    def get_eid(self):
        eids = one.search(subject=self.subj, date=self.date, number=self.n_sess,
                          task_protocol='ephys')
        self.eid = eids[0]
        print(self.subj)
        print(self.probe_label)
        print(self.date)
        print(self.eid)


    def get_starting_alignment(self, idx):
        align = self.prev_align[idx]

        if align == 'original':
            feature = None
            track = None
        else:
            feature = np.array(self.alignments[align][0])
            track = np.array(self.alignments[align][1])

        return feature, track

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
        alf_path = path.joinpath('alf', self.probe_label)
        ephys_path = path.joinpath('raw_ephys_data', self.probe_label)
        self.chn_coords = np.load(path.joinpath(alf_path, 'channels.localCoordinates.npy'))
        chn_depths = self.chn_coords[:, 1]

        sess = one.alyx.rest('sessions', 'read', id=self.eid)
        if sess['notes']:
            sess_notes = sess['notes'][0]['text']
        else:
            sess_notes = 'No notes for this session'

        return alf_path, ephys_path, chn_depths, sess_notes

    def get_slice_images(self, xyz_channels):
        hist_path = download_histology_data(self.subj, self.lab)
        ccf_slice, width, height, _ = brain_atlas.tilted_slice(xyz_channels, axis=1)
        ccf_slice = np.swapaxes(np.flipud(ccf_slice), 0, 1)
        label_slice, _, _, _ = brain_atlas.tilted_slice(xyz_channels, volume='annotation', axis=1)
        label_slice = np.swapaxes(np.flipud(label_slice), 0, 1)

        if hist_path:
            hist_atlas = atlas.AllenAtlas(hist_path=hist_path)
            hist_slice, _, _, _ = hist_atlas.tilted_slice(xyz_channels, axis=1)
            hist_slice = np.swapaxes(np.flipud(hist_slice), 0, 1)
        else:
            print('Could not find histology image for this subject')
            hist_slice = np.copy(ccf_slice)

        slice_data = {
            'hist': hist_slice,
            'ccf': ccf_slice,
            'label': label_slice,
            'scale': np.array([(width[-1] - width[0])/hist_slice.shape[0],
                               (height[-1] - height[0])/hist_slice.shape[1]]),
            'offset': np.array([width[0], height[0]])
        }

        return slice_data


    def upload_data(self, feature, track, xyz_channels, overwrite=False):

        if overwrite:
            # Get the original stored trajectory
            ephys_traj_prev = one.alyx.rest('trajectories', 'list', probe_insertion=self.probe_id,
                                       provenance='Ephys aligned histology track')
            # Save the json field in memory
            original_json = ephys_traj_prev[0]['json']

            # Create new trajectory and overwrite previous one
            insertion = atlas.Insertion.from_track(xyz_channels, brain_atlas)
            # NEEED TO ADD TIP TO DEPTH?
            brain_regions = brain_atlas.regions.get(brain_atlas.get_labels
                                                         (xyz_channels))
            brain_regions['xyz'] = xyz_channels
            brain_regions['lateral'] = self.chn_coords[:, 0]
            brain_regions['axial'] = self.chn_coords[:, 1]
            assert np.unique([len(brain_regions[k]) for k in brain_regions]).size == 1
            histology.register_aligned_track(self.probe_id, insertion, brain_regions, one=one,
                                             overwrite=overwrite)

            # Get  the new trajectoru
            ephys_traj = one.alyx.rest('trajectories', 'list', probe_insertion=self.probe_id,
                                       provenance='Ephys aligned histology track')

            name = one._par.ALYX_LOGIN
            date = datetime.now().replace(microsecond=0).isoformat()
            data = {date + '_' + name: [feature.tolist(), track.tolist()]}
            if original_json:
                original_json.update(data)
            else:
                original_json = data
            patch_dict = {'json': original_json}
            one.alyx.rest('trajectories', 'partial_update', id=ephys_traj[0]['id'], data=patch_dict)


