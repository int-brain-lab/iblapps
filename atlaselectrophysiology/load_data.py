import logging
import numpy as np
from datetime import datetime
import ibllib.pipes.histology as histology
from ibllib.ephys.neuropixel import SITES_COORDINATES
import ibllib.atlas as atlas
from ibllib.qc.alignment_qc import AlignmentQC
from one.api import ONE
from pathlib import Path
import one.alf as alf
from one import params
import glob
from atlaselectrophysiology.load_histology import download_histology_data, tif2nrrd
import ibllib.qc.critical_reasons as usrpmt

logger = logging.getLogger('ibllib')
ONE_BASE_URL = "https://alyx.internationalbrainlab.org"


class LoadData:
    def __init__(self, one=None, brain_atlas=None, testing=False, probe_id=None,
                 load_histology=True, spike_collection=None, mode='auto'):
        self.one = one or ONE(base_url=ONE_BASE_URL, mode=mode)
        self.brain_atlas = brain_atlas or atlas.AllenAtlas(25)
        self.download_hist = load_histology  # whether or not to look for the histology files
        self.spike_collection = spike_collection

        if testing:
            self.probe_id = probe_id
            self.chn_coords = SITES_COORDINATES
            self.chn_depths = SITES_COORDINATES[:, 1]
            self.probe_collection = None
        else:
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
        self.sr = None

        if probe_id is not None:
            self.sess = self.one.alyx.rest('insertions', 'list', id=probe_id)

    def get_subjects(self):
        """
        Finds all subjects that a probe insertions with spike data
        :return subjects: list of subjects
        :type: list of strings
        """

        self.sess_ins = self.one.alyx.rest('insertions', 'list', dataset_type='spikes.times')
        self.subj_ins = [sess['session_info']['subject'] for sess in self.sess_ins]

        self.subjects = np.unique(self.subj_ins)

        return self.subjects

    def get_sessions(self, idx):
        """
        Finds all sessions for a particular subject
        :param idx: index of chosen subject from drop-down list
        :type idx: int
        :return session: list of sessions associated with subject, displayed as date + probe
        :type: list of strings
        """
        subj = self.subjects[idx]
        sess_idx = [i for i, e in enumerate(self.subj_ins) if e == subj]
        self.sess = [self.sess_ins[idx] for idx in sess_idx]
        session = [(sess['session_info']['start_time'][:10] + ' ' + sess['name']) for sess in
                   self.sess]
        return session

    def get_info(self, idx):
        """
        Reads in all information about the chosen sessions and looks to see if there are any
        previous alignments associated with the session, also sees if we have histology or not
        :param idx: index of chosen session from drop-down list
        :type idx: int
        :return prev_align: list of previous alignments associated with session, if none just has
        option for 'original' alignment
        :type: list of strings
        """
        self.n_sess = self.sess[idx]['session_info']['number']
        self.date = self.sess[idx]['session_info']['start_time'][:10]
        self.probe_label = self.sess[idx]['name']
        self.probe_id = self.sess[idx]['id']
        self.lab = self.sess[idx]['session_info']['lab']
        self.eid = self.sess[idx]['session']
        self.subj = self.sess[idx]['session_info']['subject']

        histology_exists = False

        # Make sure there is a histology track
        hist_traj = self.one.alyx.rest('trajectories', 'list', probe_insertion=self.probe_id,
                                       provenance='Histology track')
        if len(hist_traj) == 1:
            if hist_traj[0]['x'] is not None:
                histology_exists = True
                self.traj_id = hist_traj[0]['id']

        return self.get_previous_alignments(), histology_exists

    def get_previous_alignments(self):
        """
        Find out if there are any previous alignments associated with probe insertion
        :return:
        """

        # Looks for any previous alignments
        ephys_traj_prev = self.one.alyx.rest('trajectories', 'list', probe_insertion=self.probe_id,
                                             provenance='Ephys aligned histology track',
                                             no_cache=True)

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

        if self.traj_ids is None:
            self.sess_with_hist = self.one.alyx.rest('trajectories', 'list',
                                                     provenance='Histology track',
                                                     django='x__isnull,False,probe_insertion__'
                                                     'datasets__name__icontains,spikes.times')
            # Some do not have tracing, exclude these ones

            depths = np.arange(200, 4100, 20) / 1e6
            trajectories = [atlas.Insertion.from_dict(sess) for sess in self.sess_with_hist]
            self.traj_ids = [sess['id'] for sess in self.sess_with_hist]
            self.traj_coords = np.empty((len(self.traj_ids), len(depths), 3))
            for iT, traj in enumerate(trajectories):
                self.traj_coords[iT, :] = (histology.interpolate_along_track
                                           (np.vstack([traj.tip, traj.entry]), depths))

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

        self.sess_path = self.one.eid2path(self.eid)

        # THIS IS STUPID
        if self.spike_collection == '':
            self.probe_collection = f'alf/{self.probe_label}'
            probe_path = Path(self.sess_path, 'alf', self.probe_label)
        elif self.spike_collection:
            self.probe_collection = f'alf/{self.probe_label}/{self.spike_collection}'
            probe_path = Path(self.sess_path, 'alf', self.probe_label, self.spike_collection)
        else:
            # Pykilosort is default, if not present look for normal kilosort
            # Find all collections
            all_collections = self.one.list_collections(self.eid)

            if f'alf/{self.probe_label}/pykilosort' in all_collections:
                self.probe_collection = f'alf/{self.probe_label}/pykilosort'
                probe_path = Path(self.sess_path, 'alf', self.probe_label, 'pykilosort')
            else:
                self.probe_collection = f'alf/{self.probe_label}'
                probe_path = Path(self.sess_path, 'alf', self.probe_label)

        try:
            _ = self.one.load_object(self.eid, 'spikes', collection=self.probe_collection,
                                     attribute=['depths', 'amps', 'times', 'clusters'],
                                     download_only=True)

            _ = self.one.load_object(self.eid, 'clusters', collection=self.probe_collection,
                                     attribute=['metrics', 'peakToTrough', 'waveforms',
                                                'channels'],
                                     download_only=True)

            _ = self.one.load_object(self.eid, 'channels', collection=self.probe_collection,
                                     attribute=['rawInd', 'localCoordinates'], download_only=True)
        except alf.exceptions.ALFObjectNotFound:
            logger.error(f'Could not load spike sorting for probe insertion {self.probe_id}, GUI'
                         f' will not work')
            return [None] * 5

        dtypes_raw = [
            '_iblqc_ephysTimeRmsAP.rms.npy',
            '_iblqc_ephysTimeRmsAP.timestamps.npy',
            '_iblqc_ephysSpectralDensityAP.freqs.npy',
            '_iblqc_ephysSpectralDensityAP.power.npy',
            '_iblqc_ephysTimeRmsLF.rms.npy',
            '_iblqc_ephysTimeRmsLF.timestamps.npy',
            '_iblqc_ephysSpectralDensityLF.freqs.npy',
            '_iblqc_ephysSpectralDensityLF.power.npy',
        ]

        collection_raw = [f'raw_ephys_data/{self.probe_label}'] * len(dtypes_raw)

        dtypes_alf = [
            '_ibl_passiveGabor.table.csv',
            '_ibl_passivePeriods.intervalsTable.csv',
            '_ibl_passiveRFM.times.npy',
            '_ibl_passiveStims.table.csv']

        collection_alf = ['alf'] * len(dtypes_alf)

        dtypes_passive = [
            '_iblrig_RFMapStim.raw.bin']

        collection_passive = ['raw_passive_data'] * len(dtypes_passive)

        dtypes = dtypes_raw + dtypes_alf + dtypes_passive
        collections = collection_raw + collection_alf + collection_passive

        print(self.subj)
        print(self.probe_label)
        print(self.date)
        print(self.eid)
        print(self.probe_collection)

        _ = self.one.load_datasets(self.eid, datasets=dtypes, collections=collections,
                                   download_only=True, assert_present=False)

        ephys_path = Path(self.sess_path, 'raw_ephys_data', self.probe_label)
        alf_path = Path(self.sess_path, 'alf')

        self.chn_coords = np.load(Path(probe_path, 'channels.localCoordinates.npy'))
        self.chn_depths = self.chn_coords[:, 1]
        self.cluster_chns = np.load(Path(probe_path, 'clusters.channels.npy'))

        sess = self.one.alyx.rest('sessions', 'read', id=self.eid)
        sess_notes = None
        if sess['notes']:
            sess_notes = sess['notes'][0]['text']
        if not sess_notes:
            sess_notes = sess['narrative']
        if not sess_notes:
            sess_notes = 'No notes for this session'

        return probe_path, ephys_path, alf_path, self.chn_depths, sess_notes

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
        insertion = self.one.alyx.rest('insertions', 'read', id=self.probe_id, no_cache=True)
        self.xyz_picks = np.array(insertion['json']['xyz_picks']) / 1e6
        self.resolved = (insertion.get('json', {'temp': 0}).get('extended_qc', {'temp': 0}).
                         get('alignment_resolved', False))

        return self.xyz_picks

    def get_slice_images(self, xyz_channels):

        index = self.brain_atlas.bc.xyz2i(xyz_channels)[:, self.brain_atlas.xyz2dims]
        ccf_slice = self.brain_atlas.image[index[:, 0], :, index[:, 2]]
        ccf_slice = np.swapaxes(ccf_slice, 0, 1)

        label_slice = self.brain_atlas._label2rgb(self.brain_atlas.label[index[:, 0], :,
                                                  index[:, 2]])
        label_slice = np.swapaxes(label_slice, 0, 1)

        width = [self.brain_atlas.bc.i2x(0), self.brain_atlas.bc.i2x(456)]
        height = [self.brain_atlas.bc.i2z(index[0, 2]), self.brain_atlas.bc.i2z(index[-1, 2])]

        hist_path_rd = None
        hist_path_gr = None
        # First see if the histology file exists before attempting to connect with FlatIron and
        # download
        if self.download_hist:
            hist_dir = Path(self.sess_path.parent.parent, 'histology')
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
                                             overwrite=True, channels=channels, brain_atlas=self.brain_atlas)
        else:
            channel_upload = False

        return channel_upload

    def update_alignments(self, feature, track, key_info=None, user_eval=None):
        if not key_info:
            user = params.get().ALYX_LOGIN
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
                                        provenance='Ephys aligned histology track', no_cache=True)
        patch_dict = {'json': json_data}
        self.one.alyx.rest('trajectories', 'partial_update', id=ephys_traj[0]['id'],
                           data=patch_dict)

    def upload_dj(self, align_qc, ephys_qc, ephys_desc):
        # Upload qc results to datajoint table
        # Check the FTP patcher credentials are in the params, so we can upload filed
        self.check_FTP_patcher_credentials()

        if len(ephys_desc) == 0:
            ephys_desc_str = 'None'
        else:
            ephys_desc_str = ", ".join(ephys_desc)

        self.alyx_str = ephys_qc.upper() + ': ' + ephys_desc_str

        if ephys_qc.upper() == 'CRITICAL':
            usrpmt.main_gui(eid=self.probe_id, reasons_selected=ephys_desc, one=self.one)

    def update_qc(self, upload_alyx=True, upload_flatiron=True):
        # if resolved just update the alignment_number
        align_qc = AlignmentQC(self.probe_id, one=self.one, brain_atlas=self.brain_atlas,
                               collection=self.probe_collection)
        align_qc.load_data(prev_alignments=self.alignments, xyz_picks=self.xyz_picks,
                           depths=self.chn_depths, cluster_chns=self.cluster_chns,
                           chn_coords=self.chn_coords)
        results = align_qc.run(update=True, upload_alyx=upload_alyx,
                               upload_flatiron=upload_flatiron)
        align_qc.update_experimenter_evaluation(prev_alignments=self.alignments)

        self.resolved = results['alignment_resolved']

        return self.resolved

    def check_FTP_patcher_credentials(self):
        par = self.one.alyx._par
        if not params._get_current_par('FTP_DATA_SERVER_LOGIN', par) \
                or not params._get_current_par('FTP_DATA_SERVER_PWD', par):
            par_dict = par.as_dict()
            par_dict['FTP_DATA_SERVER_LOGIN'] = 'iblftp'
            par_dict['FTP_DATA_SERVER_PWD'] = params._get_current_par('HTTP_DATA_SERVER_PWD', par)

            from one.params import _PAR_ID_STR  # noqa
            from iblutil.io import params as iopar  # noqa
            iopar.write(f'{_PAR_ID_STR}/{params._key_from_url(self.one.alyx.base_url)}', par_dict)
            self.one.alyx._par = params.get(client=self.one.alyx.base_url)
