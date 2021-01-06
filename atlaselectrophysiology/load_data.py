import numpy as np
from datetime import datetime
import ibllib.pipes.histology as histology
from ibllib.ephys.neuropixel import SITES_COORDINATES
import ibllib.atlas as atlas
from ibllib.qc.alignment_qc import AlignmentQC
from oneibl.one import ONE
from pathlib import Path
import alf.io
import glob
import os
from atlaselectrophysiology.load_histology import download_histology_data, tif2nrrd

ONE_BASE_URL = "https://alyx.internationalbrainlab.org"


class LoadData:
    def __init__(self, one=None, brain_atlas=None, testing=False, probe_id=None):
        self.one = one or ONE(base_url=ONE_BASE_URL)
        self.brain_atlas = brain_atlas or atlas.AllenAtlas(25)

        if testing:
            self.probe_id = probe_id
            self.chn_coords = SITES_COORDINATES
            self.chn_depths = SITES_COORDINATES[:, 1]
        else:
            from atlaselectrophysiology import qc_table
            self.qc = qc_table.EphysQC()
            self.brain_regions = self.one.alyx.rest('brain-regions', 'list')
            self.chn_coords = None
            self.chn_depths = None

        # Initialise all variables that get assigned
        self.sess_with_hist = None
        self.traj_ids = None
        self.traj_coords = None
        self.subjects = None
        self.sess = None
        self.eid = None
        self.lab = None
        self.n_sess = None
        self.probe_label = None
        self.traj_id = None
        self.date = None
        self.subj = None
        self.alignments = {}
        self.prev_align = None
        self.sess_path = None
        self.allen_id = None
        self.cluster_chns = None
        self.resolved = None
        self.alyx_str = None

    def get_subjects(self):
        """
        Finds all subjects that have a histology track trajectory registered
        :return subjects: list of subjects
        :type: list of strings
        """
        # All sessions that have a histology track
        all_hist = self.one.alyx.rest('trajectories', 'list', provenance='Histology track')
        # Some do not have tracing, exclude these ones
        self.sess_with_hist = [sess for sess in all_hist if sess['x'] is not None]
        self.subj_with_hist = [sess['session']['subject'] for sess in self.sess_with_hist]

        # For all histology tracks find the trajectory and coordinates of active part (where
        # electrodes are located). Used in get_nearby trajectories to find sessions close-by
        # insertions
        depths = np.arange(200, 4100, 20) / 1e6
        trajectories = [atlas.Insertion.from_dict(sess) for sess in self.sess_with_hist]
        self.traj_ids = [sess['id'] for sess in self.sess_with_hist]
        self.traj_coords = np.empty((len(self.traj_ids), len(depths), 3))
        for iT, traj in enumerate(trajectories):
            self.traj_coords[iT, :] = (histology.interpolate_along_track
                                       (np.vstack([traj.tip, traj.entry]), depths))

        self.subjects = np.unique(self.subj_with_hist)

        return self.subjects

    def get_sessions(self, idx):
        """
        Finds all sessions for a particular subject that have a histology track trajectory
        registered
        :param idx: index of chosen subject from drop-down list
        :type idx: int
        :return session: list of sessions associated with subject, displayed as date + probe
        :type: list of strings
        """
        self.subj = self.subjects[idx]
        sess_idx = [i for i, e in enumerate(self.subj_with_hist) if e == self.subj]
        self.sess = [self.sess_with_hist[idx] for idx in sess_idx]
        session = [(sess['session']['start_time'][:10] + ' ' + sess['probe_name']) for sess in
                   self.sess]
        return session

    def get_info(self, idx):
        """
        Reads in all information about the chosen sessions and looks to see if there are any
        previous alignments associated with the session
        :param idx: index of chosen session from drop-down list
        :type idx: int
        :return prev_align: list of previous alignments associated with session, if none just has
        option for 'original' alignment
        :type: list of strings
        """
        self.n_sess = self.sess[idx]['session']['number']
        self.date = self.sess[idx]['session']['start_time'][:10]
        self.probe_label = self.sess[idx]['probe_name']
        self.traj_id = self.sess[idx]['id']
        self.probe_id = self.sess[idx]['probe_insertion']
        self.lab = self.sess[idx]['session']['lab']
        self.eid = self.sess[idx]['session']['id']

        return self.get_previous_alignments()

    def get_previous_alignments(self):

        # Looks for any previous alignments
        ephys_traj_prev = self.one.alyx.rest('trajectories', 'list', probe_insertion=self.probe_id,
                                             provenance='Ephys aligned histology track')

        if ephys_traj_prev:
            self.alignments = ephys_traj_prev[0]['json'] or {}
            self.prev_align = [*self.alignments.keys()]
            self.prev_align = sorted(self.prev_align, reverse=True)
            self.prev_align.append('original')
        else:
            self.alignments = {}
            self.prev_align = ['original']

        return self.prev_align

    def get_starting_alignment(self, idx):
        """
        Finds all sessions for a particular subject that have a histology track trajectory
        registered
        :param idx: index of chosen alignment from drop-down list
        :type idx: int
        :return feature: reference points in feature space
        :type: np.array
        :return track: reference points in track space
        :type: np.array
        """
        self.current_align = self.prev_align[idx]

        if self.current_align == 'original':
            feature = None
            track = None
        else:
            feature = np.array(self.alignments[self.current_align][0])
            track = np.array(self.alignments[self.current_align][1])

        return feature, track

    def get_nearby_trajectories(self):
        """
        Find sessions that have trajectories close to the currently selected session
        :return close_session: list of nearby sessions ordered by absolute distance, displayed as
        subject + date + probe
        :type: list of strings
        :return close_dist: absolute distance to nearby sessions
        :type: list of float
        :return close_dist_mlap: absolute distance to nearby sessions, only using ml and ap
        directions
        :type: list of float
        """

        chosen_traj = self.traj_ids.index(self.traj_id)
        avg_dist = np.mean(np.sqrt(np.sum((self.traj_coords - self.traj_coords[chosen_traj]) ** 2,
                                          axis=2)), axis=1)
        avg_dist_mlap = np.mean(np.sqrt(np.sum((self.traj_coords[:, :, 0:2] -
                                                self.traj_coords[chosen_traj][:, 0:2]) ** 2,
                                        axis=2)), axis=1)

        closest_traj = np.argsort(avg_dist)
        close_dist = avg_dist[closest_traj[0:10]] * 1e6
        close_dist_mlap = avg_dist_mlap[closest_traj[0:10]] * 1e6

        close_sessions = []
        for sess_idx in closest_traj[0:10]:
            close_sessions.append((self.sess_with_hist[sess_idx]['session']['subject'] + ' ' +
                                  self.sess_with_hist[sess_idx]['session']['start_time'][:10] +
                                   ' ' + self.sess_with_hist[sess_idx]['probe_name']))

        return close_sessions, close_dist, close_dist_mlap

    def get_data(self):
        """
        Load/ Download all data and info associated with session
        :return alf_path: path to folder containing alf format files
        :type: Path
        :return ephys_path: path to folder containing ephys files
        :type: Path
        :return chn_depths: depths of electrode channel on probe relative to tip
        :type: np.array(384,1)
        :return sess_notes: user notes associated with session
        :type: str
        """
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
            'clusters.channels',
            '_iblqc_ephysTimeRms.rms',
            '_iblqc_ephysTimeRms.timestamps',
            '_iblqc_ephysSpectralDensity.freqs',
            '_iblqc_ephysSpectralDensity.power',
            '_iblqc_ephysSpectralDensity.amps',
            '_ibl_passiveGabor.table',
            '_ibl_passivePeriods.intervalsTable',
            '_ibl_passiveRFM.frames',
            '_ibl_passiveRFM.times',
            '_ibl_passiveStims.table'
        ]

        print(self.subj)
        print(self.probe_label)
        print(self.date)
        print(self.eid)

        _ = self.one.load(self.eid, dataset_types=dtypes, download_only=True)
        self.sess_path = self.one.path_from_eid(self.eid)

        alf_path = Path(self.sess_path, 'alf', self.probe_label)
        ephys_path = Path(self.sess_path, 'raw_ephys_data', self.probe_label)

        cluster_file_old = alf_path.joinpath('clusters.metrics.csv')
        if cluster_file_old.exists():
            os.remove(cluster_file_old)

        try:
            self.chn_coords = np.load(Path(alf_path, 'channels.localCoordinates.npy'))
            self.chn_depths = self.chn_coords[:, 1]
            self.cluster_chns = np.load(Path(alf_path, 'clusters.channels.npy'))
        except Exception:
            print('Could not download alf data for this probe - gui will not work')
            return [None] * 4

        sess = self.one.alyx.rest('sessions', 'read', id=self.eid)
        sess_notes = None
        if sess['notes']:
            sess_notes = sess['notes'][0]['text']
        if not sess_notes:
            sess_notes = sess['narrative']
        if not sess_notes:
            sess_notes = 'No notes for this session'

        return alf_path, ephys_path, self.chn_depths, sess_notes

    def get_allen_csv(self):
        """
        Load in allen csv file
        :return allen: dataframe containing all information in csv file
        :type: pd.Dataframe
        """
        allen_path = Path(Path(atlas.__file__).parent, 'allen_structure_tree.csv')
        allen = alf.io.load_file_content(allen_path)

        self.allen_id = allen['id']

        return allen

    def get_xyzpicks(self):
        """
        Load in user chosen track tracing points
        :return xyz_picks: 3D coordinates of points relative to bregma
        :type: np.array(n_picks, 3)
        """
        insertion = self.one.alyx.rest('insertions', 'read', id=self.probe_id)
        self.xyz_picks = np.array(insertion['json']['xyz_picks']) / 1e6
        self.resolved = (insertion.get('json', {'temp': 0}).get('extended_qc', {'temp': 0}).
                         get('alignment_resolved', False))

        return self.xyz_picks

    def get_slice_images(self, xyz_channels):
        # First see if the histology file exists before attempting to connect with FlatIron and
        # download
        hist_dir = Path(self.sess_path.parent.parent, 'histology')
        hist_path_rd = None
        hist_path_gr = None
        if hist_dir.exists():
            path_to_rd_image = glob.glob(str(hist_dir) + '/*RD.tif')
            if path_to_rd_image:
                hist_path_rd = tif2nrrd(Path(path_to_rd_image[0]))
            else:
                files = download_histology_data(self.subj, self.lab)
                if files is not None:
                    hist_path_rd = files[1]

            path_to_gr_image = glob.glob(str(hist_dir) + '/*GR.tif')
            if path_to_gr_image:
                hist_path_gr = tif2nrrd(Path(path_to_gr_image[0]))
            else:
                files = download_histology_data(self.subj, self.lab)
                if files is not None:
                    hist_path_gr = files[0]

        else:
            files = download_histology_data(self.subj, self.lab)
            if files is not None:
                hist_path_gr = files[0]
                hist_path_rd = files[1]

        index = self.brain_atlas.bc.xyz2i(xyz_channels)[:, self.brain_atlas.xyz2dims]
        ccf_slice = self.brain_atlas.image[index[:, 0], :, index[:, 2]]
        ccf_slice = np.swapaxes(ccf_slice, 0, 1)

        label_slice = self.brain_atlas._label2rgb(self.brain_atlas.label[index[:, 0], :,
                                                  index[:, 2]])
        label_slice = np.swapaxes(label_slice, 0, 1)

        width = [self.brain_atlas.bc.i2x(0), self.brain_atlas.bc.i2x(456)]
        height = [self.brain_atlas.bc.i2z(index[0, 2]), self.brain_atlas.bc.i2z(index[-1, 2])]

        if hist_path_rd:
            hist_atlas_rd = atlas.AllenAtlas(hist_path=hist_path_rd)
            hist_slice_rd = hist_atlas_rd.image[index[:, 0], :, index[:, 2]]
            hist_slice_rd = np.swapaxes(hist_slice_rd, 0, 1)
        else:
            print('Could not find red histology image for this subject')
            hist_slice_rd = np.copy(ccf_slice)

        if hist_path_gr:
            hist_atlas_gr = atlas.AllenAtlas(hist_path=hist_path_gr)
            hist_slice_gr = hist_atlas_gr.image[index[:, 0], :, index[:, 2]]
            hist_slice_gr = np.swapaxes(hist_slice_gr, 0, 1)
        else:
            print('Could not find green histology image for this subject')
            hist_slice_gr = np.copy(ccf_slice)

        slice_data = {
            'hist_rd': hist_slice_rd,
            'hist_gr': hist_slice_gr,
            'ccf': ccf_slice,
            'label': label_slice,
            'scale': np.array([(width[-1] - width[0]) / ccf_slice.shape[0],
                               (height[-1] - height[0]) / ccf_slice.shape[1]]),
            'offset': np.array([width[0], height[0]])
        }

        return slice_data

    def get_region_description(self, region_idx):
        struct_idx = np.where(self.allen_id == region_idx)[0][0]
        description = self.brain_regions[struct_idx]['description']
        region_lookup = (self.brain_regions[struct_idx]['acronym'] +
                         ': ' + self.brain_regions[struct_idx]['name'])

        if region_lookup == 'void: void':
            region_lookup = 'root: root'

        if not description:
            description = region_lookup + '\nNo information available on Alyx for this region'
        else:
            description = region_lookup + '\n' + description

        return description, region_lookup

    def upload_data(self, xyz_channels, channels=True):
        if not self.resolved:
            channel_upload = True
            # Create new trajectory and overwrite previous one
            histology.register_aligned_track(self.probe_id, xyz_channels,
                                             chn_coords=self.chn_coords, one=self.one,
                                             overwrite=True, channels=channels)
        else:
            channel_upload = False

        return channel_upload

    def update_alignments(self, feature, track, key_info=None, user_eval=None):
        if not key_info:
            user = self.one._par.ALYX_LOGIN
            date = datetime.now().replace(microsecond=0).isoformat()
            data = {date + '_' + user: [feature.tolist(), track.tolist(), self.alyx_str]}
        else:
            user = key_info[20:]
            if user_eval:
                data = {key_info: [feature.tolist(), track.tolist(), user_eval]}
            else:
                data = {key_info: [feature.tolist(), track.tolist()]}

        old_user = [key for key in self.alignments.keys() if user in key]
        # Only delete duplicated if trajectory is not resolved
        if len(old_user) > 0 and not self.resolved:
            self.alignments.pop(old_user[0])

        self.alignments.update(data)
        self.update_json(self.alignments)

    def update_json(self, json_data):
        # Get the new trajectory
        ephys_traj = self.one.alyx.rest('trajectories', 'list', probe_insertion=self.probe_id,
                                        provenance='Ephys aligned histology track')
        patch_dict = {'json': json_data}
        self.one.alyx.rest('trajectories', 'partial_update', id=ephys_traj[0]['id'],
                           data=patch_dict)

    def upload_dj(self, align_qc, ephys_qc, ephys_desc):
        # Upload qc results to datajoint table
        user = self.one._par.ALYX_LOGIN
        if len(ephys_desc) == 0:
            ephys_desc_str = 'None'
            ephys_dj_str = None
        else:
            ephys_desc_str = ", ".join(ephys_desc)
            ephys_dj_str = ephys_desc_str

        self.qc.insert1(dict(probe_insertion_uuid=self.probe_id, user_name=user,
                             alignment_qc=align_qc, ephys_qc=ephys_qc,
                             ephys_qc_description=ephys_dj_str),
                        allow_direct_insert=True, replace=True)
        self.alyx_str = ephys_qc.upper() + ': ' + ephys_desc_str

    def update_qc(self, upload_alyx=True, upload_flatiron=True):
        # if resolved just update the alignment_number
        align_qc = AlignmentQC(self.probe_id, one=self.one, brain_atlas=self.brain_atlas)
        align_qc.load_data(prev_alignments=self.alignments, xyz_picks=self.xyz_picks,
                           depths=self.chn_depths, cluster_chns=self.cluster_chns)
        results = align_qc.run(update=True, upload_alyx=upload_alyx,
                               upload_flatiron=upload_flatiron)
        align_qc.update_experimenter_evaluation(prev_alignments=self.alignments)

        self.resolved = results['alignment_resolved']

        return self.resolved
