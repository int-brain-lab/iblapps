import scipy
import numpy as np
from datetime import datetime
import ibllib.pipes.histology as histology
import ibllib.atlas as atlas
from oneibl.one import ONE
from pathlib import Path
import alf.io
import glob
from atlaselectrophysiology.load_histology import download_histology_data, tif2nrrd
# from atlaselectrophysiology import qc_table
brain_atlas = atlas.AllenAtlas(25)
ONE_BASE_URL = "https://alyx.internationalbrainlab.org"


class LoadData:
    def __init__(self):

        self.one = ONE(base_url=ONE_BASE_URL)
        self.brain_regions = self.one.alyx.rest('brain-regions', 'list')

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
        self.probe_id = None
        self.date = None
        self.subj = None
        self.alignments = None
        self.prev_align = None
        self.chn_coords = None
        self.sess_path = None
        self.allen_id = None

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
        subjects = [sess['session']['subject'] for sess in self.sess_with_hist]

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

        self.subjects = np.unique(subjects)

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
        self.sess = self.one.alyx.rest('trajectories', 'list', subject=self.subj,
                                       provenance='Histology track')
        session = [(sess['session']['start_time'][:10] + ' ' + sess['probe_name']) for sess in
                   self.sess if sess['x']]
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
        self.eid = self.one.search(subject=self.subj, date=self.date, number=self.n_sess,
                                   task_protocol='ephys')[0]

        # Looks for any previous alignments
        ephys_traj_prev = self.one.alyx.rest('trajectories', 'list', probe_insertion=self.probe_id,
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
        align = self.prev_align[idx]

        if align == 'original':
            feature = None
            track = None
        else:
            feature = np.array(self.alignments[align][0])
            track = np.array(self.alignments[align][1])

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
            '_iblqc_ephysTimeRms.rms',
            '_iblqc_ephysTimeRms.timestamps',
            '_iblqc_ephysSpectralDensity.freqs',
            '_iblqc_ephysSpectralDensity.power'
        ]

        print(self.subj)
        print(self.probe_label)
        print(self.date)
        print(self.eid)

        _ = self.one.load(self.eid, dataset_types=dtypes, download_only=True)
        self.sess_path = self.one.path_from_eid(self.eid)
        alf_path = Path(self.sess_path, 'alf', self.probe_label)
        ephys_path = Path(self.sess_path, 'raw_ephys_data', self.probe_label)
        self.chn_coords = np.load(Path(alf_path, 'channels.localCoordinates.npy'))
        chn_depths = self.chn_coords[:, 1]

        sess = self.one.alyx.rest('sessions', 'read', id=self.eid)
        sess_notes = None
        if sess['notes']:
            sess_notes = sess['notes'][0]['text']
        if not sess_notes:
            sess_notes = sess['narrative']
        if not sess_notes:
            sess_notes = 'No notes for this session'

        return alf_path, ephys_path, chn_depths, sess_notes

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
        insertion = self.one.alyx.rest('insertions', 'list', session=self.eid,
                                       name=self.probe_label)
        self.insertion_id = insertion[0]['id']
        xyz_picks = np.array(insertion[0]['json']['xyz_picks']) / 1e6

        return xyz_picks

    def get_slice_images(self, xyz_channels):
        # First see if the histology file exists before attempting to connect with FlatIron and
        # download
        hist_dir = Path(self.sess_path.parent.parent, 'histology')
        if hist_dir.exists():
            path_to_rd_image = glob.glob(str(hist_dir) + '/*RD.tif')
            if path_to_rd_image:
                hist_path_rd = tif2nrrd(Path(path_to_rd_image[0]))
            else:
                _, hist_path_rd = download_histology_data(self.subj, self.lab)

            path_to_gr_image = glob.glob(str(hist_dir) + '/*GR.tif')
            if path_to_gr_image:
                hist_path_gr = tif2nrrd(Path(path_to_gr_image[0]))
            else:
                hist_path_gr, _ = download_histology_data(self.subj, self.lab)
        else:
            hist_path_gr, hist_path_rd = download_histology_data(self.subj, self.lab)

        index = brain_atlas.bc.xyz2i(xyz_channels)[:, brain_atlas.xyz2dims]
        ccf_slice = brain_atlas.image[index[:, 0], :, index[:, 2]]
        ccf_slice = np.swapaxes(ccf_slice, 0, 1)

        label_slice = brain_atlas._label2rgb(brain_atlas.label[index[:, 0], :, index[:, 2]])
        label_slice = np.swapaxes(label_slice, 0, 1)

        width = [brain_atlas.bc.i2x(0), brain_atlas.bc.i2x(456)]
        height = [brain_atlas.bc.i2z(index[0, 2]), brain_atlas.bc.i2z(index[-1, 2])]

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
            print('Could not find histology image for this subject')
            hist_slice_gr = np.copy(ccf_slice)

        slice_data = {
            'hist_rd': hist_slice_rd,
            'hist_gr': hist_slice_gr,
            'ccf': ccf_slice,
            'label': label_slice,
            'scale': np.array([(width[-1] - width[0])/ccf_slice.shape[0],
                               (height[-1] - height[0])/ccf_slice.shape[1]]),
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

    def upload_data(self, feature, track, xyz_channels, overwrite=False):

        if overwrite:
            # Get the original stored trajectory
            ephys_traj_prev = self.one.alyx.rest('trajectories', 'list',
                                                 probe_insertion=self.probe_id,
                                                 provenance='Ephys aligned histology track')
            # Save the json field in memory
            original_json = None
            if np.any(ephys_traj_prev):
                original_json = ephys_traj_prev[0]['json']

            # Create new trajectory and overwrite previous one
            insertion = atlas.Insertion.from_track(xyz_channels, brain_atlas)
            # NEEED TO ADD TIP TO DEPTH?
            brain_regions = brain_atlas.regions.get(brain_atlas.get_labels(xyz_channels))
            brain_regions['xyz'] = xyz_channels
            brain_regions['lateral'] = self.chn_coords[:, 0]
            brain_regions['axial'] = self.chn_coords[:, 1]
            assert np.unique([len(brain_regions[k]) for k in brain_regions]).size == 1
            histology.register_aligned_track(self.probe_id, insertion, brain_regions, one=self.one,
                                             overwrite=overwrite)

            # Get the new trajectoru
            ephys_traj = self.one.alyx.rest('trajectories', 'list', probe_insertion=self.probe_id,
                                            provenance='Ephys aligned histology track')

            name = self.one._par.ALYX_LOGIN
            date = datetime.now().replace(microsecond=0).isoformat()
            data = {date + '_' + name: [feature.tolist(), track.tolist()]}
            if original_json:
                original_json.update(data)
            else:
                original_json = data
            patch_dict = {'json': original_json}
            self.one.alyx.rest('trajectories', 'partial_update', id=ephys_traj[0]['id'],
                               data=patch_dict)

    def upload_dj(self, align_qc, ephys_qc, ephys_desc):
        # Upload qc results to datajoint table

        user = self.one._par.ALYX_LOGIN
        qc = qc_table.EphysQC()
        if ephys_desc == 'None':
            ephys_desc = None
        qc.insert1(dict(probe_insertion_uuid=self.insertion_id, user_name=user,
                        alignment_qc=align_qc, ephys_qc=ephys_qc, ephys_qc_description=ephys_desc),
                   allow_direct_insert=True, replace=True)

