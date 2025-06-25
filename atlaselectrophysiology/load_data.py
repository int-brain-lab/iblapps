import logging
import traceback
import numpy as np
from datetime import datetime
import ibllib.pipes.histology as histology

from neuropixel import trace_header
import iblatlas.atlas as atlas
from ibllib.qc.alignment_qc import AlignmentQC
from iblutil.numerical import ismember
from iblutil.util import Bunch
from one.api import ONE

from one.alf.exceptions import ALFObjectNotFound
from one import params
import json

from atlaselectrophysiology.loaders.histology_loader import NrrdSliceLoader, download_histology_data
from atlaselectrophysiology.plot_data import PlotData
import ibllib.qc.critical_reasons as usrpmt
from datetime import timedelta
import re

from atlaselectrophysiology.utils.align import AlignData

logger = logging.getLogger('ibllib')
ONE_BASE_URL = "https://alyx.internationalbrainlab.org"


class LoadData:
    def __init__(self, one=None, brain_atlas=None, testing=False, probe_id=None,
                 load_histology=True, spike_collection=None):
        self.one = one or ONE(base_url=ONE_BASE_URL)
        self.brain_atlas = brain_atlas or atlas.AllenAtlas(25)
        # self.franklin_atlas = atlas.FranklinPaxinosAtlas()
        self.franklin_atlas = None
        self.download_hist = load_histology  # whether or not to look for the histology files
        self.spike_collection = spike_collection

        if testing:
            self.probe_id = probe_id
            refch_3a = np.array([36, 75, 112, 151, 188, 227, 264, 303, 340, 379])
            th = trace_header(version=1)
            SITES_COORDINATES = np.delete(np.c_[th['x'], th['y']], refch_3a, axis=0)
            self.chn_coords = SITES_COORDINATES
            self.chn_depths = SITES_COORDINATES[:, 1]
            self.probe_collection = None
        else:
            self.brain_regions = self.one.alyx.rest('brain-regions', 'list', expires=timedelta(days=1))
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
        self.probe_path = None
        self.slice_loader = None

        if probe_id is not None:
            self.sess = self.one.alyx.rest('insertions', 'list', id=probe_id)

    def get_subjects(self):
        """
        Finds all subjects that a probe insertions with spike data
        :return subjects: list of subjects
        :type: list of strings
        """

        self.sess_ins = self.one.alyx.rest('insertions', 'list', dataset_type='spikes.times', expires=timedelta(days=1))
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
        self.sessions = [self.get_session_probe_name(sess) for sess in self.sess]
        self.sessions = np.unique(self.sessions)

        # idx = np.argsort(session)
        # #self.sess = np.array(self.sess)[idx]
        # session = np.array(session)[idx]

        return self.sessions

    def get_shanks(self, idx):

        sess = self.sessions[idx]
        sess_idx = [i for i, e in enumerate(self.sess) if self.get_session_probe_name(e) == sess]
        self.shanks = [self.sess[idx] for idx in sess_idx]
        shanks = [s['name'] for s in self.shanks]
        # TODO better way to do this pleasee!
        idx = np.argsort(shanks)
        self.shanks = np.array(self.shanks)[idx]
        shanks = np.array(shanks)[idx]


        self.probes = Bunch()
        #insertions = [self.sess[idx]] + self.get_other_shanks()
        for ins in self.shanks:
            self.probes[ins['name']] = ProbeLoader(ins, self.one, self.brain_atlas)

        return list(shanks) + ['all']


    @staticmethod
    def normalize_probe_name(probe_name):
        match = re.match(r'(probe\d+)', probe_name)
        return match.group(1) if match else probe_name


    def get_session_probe_name(self, ins):
        return ins['session_info']['start_time'][:10] + ' ' + self.normalize_probe_name(ins['name'])

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

        self.n_sess = self.shanks[idx]['session_info']['number']
        self.date = self.shanks[idx]['session_info']['start_time'][:10]
        self.probe_label = self.shanks[idx]['name']
        self.probe_id = self.shanks[idx]['id']
        self.lab = self.shanks[idx]['session_info']['lab']
        self.eid = self.shanks[idx]['session']
        self.subj = self.shanks[idx]['session_info']['subject']


        print(self.subj)
        print(self.probe_label)
        print(self.date)
        print(self.eid)



    def get_starting_alignment(self, idx):
        self.probes[self.probe_label].get_starting_alignment(idx)

    def get_previous_alignments(self):
        return self.probes[self.probe_label].get_previous_alignments()

    def get_selected_probe(self):
        return self.probes[self.probe_label]

    def download_histology(self):
        _, hist_path = download_histology_data(self.subj, self.lab)
        self.slice_loader = NrrdSliceLoader(hist_path, self.brain_atlas)

    # TODO BETTER NAMING
    def load_data(self):
        # Should this be here?
        if self.slice_loader is None:
            self.download_histology()
        for probe in self.probes.keys():
            self.probes[probe].load_data(self.slice_loader)


    def add_extra_alignments(self, file_path):

        with open(file_path, "r") as f:
            extra_alignments = json.load(f)
        self.extra_alignments = {}
        # Add the username to the keys, required if combining and offline and online alignment
        user = params.get().ALYX_LOGIN
        for key, val in extra_alignments.items():
            if len(key) == 19:
                self.extra_alignments[key + '_' + user] = val
            else:
                self.extra_alignments[key] = val

        if len(self.alignments) > 0:
            self.alignments.update(self.extra_alignments)
        else:
            self.alignments = self.extra_alignments
        self.prev_align = [*self.alignments.keys()]
        self.prev_align = sorted(self.prev_align, reverse=True)
        self.prev_align.append('original')

        return self.prev_align


    def get_other_shanks(self):

        insertions = self.one.alyx.rest('insertions', 'list', session=self.eid)
        # TODO IMPROVE HANDLING OF THIS
        insertions = [ins for ins in insertions if self.probe_label[:7] in ins['name'] and
                      self.probe_label != ins['name']]

        return insertions








class ProbeLoader:
    def __init__(self, insertion, one, brain_atlas, spike_collection=None):
        self.one = one
        self.spike_collection = spike_collection
        self.brain_atlas = brain_atlas
        self.insertion = insertion
        self.probe_id = self.insertion['id']
        self.session_path = self.one.eid2path(self.insertion['session'])
        self.franklin_atlas = None
        self.download_hist = True
        self.subj = self.insertion['session_info']['subject']
        self.lab = self.insertion['session_info']['lab']
        self.eid = self.insertion['session']
        self.date = self.insertion['session_info']['start_time'][:10]
        self.probe_label = self.insertion['name']

        hist_traj = self.one.alyx.rest('trajectories', 'list', probe_insertion=self.probe_id,
                                       provenance='Histology track')
        if len(hist_traj) == 1:
            if hist_traj[0]['x'] is not None:
                self.traj_id = hist_traj[0]['id']


        self.histology = self.insertion['json'].get('extended_qc', {}).get('tracing_exists', False)
        if self.histology:
            self.xyz_picks = np.array(self.insertion['json']['xyz_picks']) / 1e6
            self.resolved = self.insertion['json'].get('extended_qc', {}).get('alignment_resolved', False)
            self.align_stored = self.insertion['json'].get('extended_qc', {}).get('alignment_stored', None)
            self.get_previous_alignments()
            self.get_starting_alignment(0)

        self.data_loaded = False

    @property
    def xyz_channels(self):
        return self.align.xyz_channels

    @property
    def xyz_track(self):
        return self.align.ephysalign.xyz_track


    def load_data(self, slice_loader):
        # TODO cleaner way to do this
        if not self.data_loaded:
            self.raw_data = DataLoader(self.one, self.eid, self.insertion['name'],
                                   spike_collection=self.spike_collection)

            self.chn_coords = self.raw_data.data['channels']['localCoordinates']
            self.chn_depths = self.chn_coords[:, 1]
            self.cluster_chns = self.raw_data.data['clusters']['channels']

            # Generate plots
            self.plotdata = PlotData(self.raw_data.probe_path, self.raw_data.data, 0)
            self.img_plots = dict()
            self.scatter_plots = dict()
            self.probe_plots = dict()
            self.line_plots = dict()

            self.img_plots.update(self.plotdata.get_fr_img())
            self.img_plots.update(self.plotdata.get_correlation_data_img())
            rms_img, rms_probe = self.plotdata.get_rms_data_img_probe('AP')
            self.img_plots.update(rms_img)
            self.probe_plots.update(rms_probe)
            rms_img, rms_probe = self.plotdata.get_rms_data_img_probe('LF')
            self.img_plots.update(rms_img)
            self.probe_plots.update(rms_probe)
            rms_img, rms_probe = self.plotdata.get_lfp_spectrum_data()
            self.img_plots.update(rms_img)
            self.probe_plots.update(rms_probe)
            self.img_plots.update(self.plotdata.get_raw_data_image(self.probe_id, one=self.one))
            self.img_plots.update(self.plotdata.get_passive_events())

            self.probe_plots.update(self.plotdata.get_rfmap_data())

            self.scatter_plots.update(self.plotdata.get_depth_data_scatter())
            self.scatter_plots.update(self.plotdata.get_fr_p2t_data_scatter())

            self.line_plots.update(self.plotdata.get_fr_amp_data_line())

            self.data_loaded = True

            self.align = AlignData(self.xyz_picks, self.chn_depths, self.brain_atlas)

            self.slice_plots = slice_loader.get_slices(self.align.xyz_samples)


        # Set the featuer to the current chosen alignment
        self.align.set_init_feature_track(self.feature_prev, self.track_prev)
        # TODO better handling
        self.probe_path = self.raw_data.probe_path
        self.probe_collection = self.raw_data.probe_collection

    def filter_plots(self):

        # self.scat_drift_data = self.plotdata.get_depth_data_scatter()
        # (self.scat_fr_data, self.scat_p2t_data,
        #  self.scat_amp_data) = self.plotdata.get_fr_p2t_data_scatter()
        # self.img_corr_data = self.plotdata.get_correlation_data_img()
        # self.img_fr_data = self.plotdata.get_fr_img()
        # self.line_fr_data, self.line_amp_data = self.plotdata.get_fr_amp_data_line()
        # self.probe_rfmap, self.rfmap_boundaries = self.plotdata.get_rfmap_data()
        # self.img_stim_data = self.plotdata.get_passive_events()

        # TODO need to make this better so that we don't get the raw ephys plots again

        self.img_plots = dict()
        self.scatter_plots = dict()
        self.probe_plots = dict()
        self.line_plots = dict()

        self.img_plots['Firing Rate'] = self.plotdata.get_fr_img()['Firing Rate']
        self.img_plots.update(self.plotdata.get_fr_img())
        self.img_plots.update(self.plotdata.get_correlation_data_img())
        rms_img, rms_probe = self.plotdata.get_rms_data_img_probe('AP')
        self.img_plots.update(rms_img)
        self.probe_plots.update(rms_probe)
        rms_img, rms_probe = self.plotdata.get_rms_data_img_probe('LF')
        self.img_plots.update(rms_img)
        self.probe_plots.update(rms_probe)
        rms_img, rms_probe = self.plotdata.get_lfp_spectrum_data()
        self.img_plots.update(rms_img)
        self.probe_plots.update(rms_probe)
        self.img_plots.update(self.plotdata.get_raw_data_image(self.probe_id, one=self.one))
        self.img_plots.update(self.plotdata.get_passive_events())

        self.probe_plots.update(self.plotdata.get_rfmap_data())

        self.scatter_plots.update(self.plotdata.get_depth_data_scatter())
        self.scatter_plots.update(self.plotdata.get_fr_p2t_data_scatter())

        self.line_plots.update(self.plotdata.get_fr_amp_data_line())


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

        # TODO highlight the resolved alignment in black

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

        # TODO by default load the resolved alignment

        self.current_align = self.prev_align[idx]
        start_lims = 6000 / 1e6
        self.feature_prev = np.array(self.alignments[self.current_align][0]) if self.current_align != 'original' else np.array([-1 * start_lims, start_lims])
        self.track_prev = np.array(self.alignments[self.current_align][1]) if self.current_align != 'original' else np.array([-1 * start_lims, start_lims])


    def upload_data(self):

        # Upload channels
        channels = self.upload_channels()
        # Upload alignments
        self.update_alignments()
        # Update alignment qc
        resolved = self.update_qc()
        # Update probe qc

        upload_info = self.get_upload_info(channels, resolved)

        return upload_info
    @staticmethod
    def get_upload_info(channels, resolved):
        if channels and resolved == 0:
            # Channels saved alignment not resolved
            info = "Channels locations saved to Alyx. \nAlignment not resolved"

        if channels and resolved == 1:
            # channels saved alignment resolved, writen to flatiron
            info = "Channel locations saved to Alyx. \nAlignment resolved and channels datasets written to flatiron"

        if not channels and resolved == 1:
            # alignment already resolved, save alignment but channels not written
            info = ("Channel locations not saved to Alyx as alignment has already been resolved. "
                    "\nNew user reference lines have been saved")

        return info

    def upload_channels(self, channels=True):
        if not self.resolved:

            channel_upload = True
            # Create new trajectory and overwrite previous one
            histology.register_aligned_track(self.probe_id, self.align.xyz_channels, chn_coords=self.chn_coords, one=self.one,
                                             overwrite=True, channels=channels, brain_atlas=self.brain_atlas)
        else:
            channel_upload = False

        return channel_upload

    def update_alignments(self, key_info=None, user_eval=None):

        feature = self.align.feature.tolist()
        track = self.align.track.tolist()

        if not key_info:
            user = params.get().ALYX_LOGIN
            date = datetime.now().replace(microsecond=0).isoformat()
            data = {date + '_' + user: [feature, track, self.qc_str, self.confidence_str]}
        else:
            user = key_info[20:]
            if user_eval:
                data = {key_info: [feature, track, user_eval]}
            else:
                data = {key_info: [feature, track]}

        old_user = [key for key in self.alignments.keys() if user in key]
        # Only delete duplicated if trajectory is not resolved
        if len(old_user) > 0 and not self.resolved:
            for old in old_user:
                self.alignments.pop(old)

        self.alignments.update(data)
        self.write_alignments_to_disk(self.alignments)
        self.update_json(self.alignments)

    def write_alignments_to_disk(self, data):
        prev_align_filename = 'prev_alignments.json'
        if self.probe_path is not None:
            with open(self.probe_path.joinpath(prev_align_filename), "w") as f:
                json.dump(data, f, indent=2, separators=(',', ': '))

    def update_json(self, json_data):
        # Get the new trajectory
        ephys_traj = self.one.alyx.rest('trajectories', 'list', probe_insertion=self.probe_id,
                                        provenance='Ephys aligned histology track', no_cache=True)
        patch_dict = {'probe_insertion': self.probe_id, 'json': json_data}
        self.one.alyx.rest('trajectories', 'partial_update', id=ephys_traj[0]['id'],
                           data=patch_dict)

    def get_qc_string(self, align_qc, ephys_qc, ephys_desc):

        if len(ephys_desc) == 0:
            ephys_desc_str = 'None'
        else:
            ephys_desc_str = ", ".join(ephys_desc)

        self.qc_str = ephys_qc.upper() + ': ' + ephys_desc_str
        self.confidence_str = f'Confidence: {align_qc}'


        if ephys_qc.upper() == 'CRITICAL':
            usrpmt.main_gui(self.probe_id, reasons_selected=ephys_desc, alyx=self.one.alyx)

    def update_qc(self, upload_alyx=True, upload_flatiron=False):
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







class DataLoader:
    def __init__(self, one, eid, probe_label, spike_collection=None):
        """
        Load and process data for a specific session/probe.
        """

        self.one = one
        self.eid = eid
        self.session_path = one.eid2path(eid)
        self.probe_label = probe_label
        self.spike_collection = spike_collection
        self.probe_path = self.get_spikesorting_path()
        self.probe_collection = str(self.probe_path.relative_to(self.session_path))
        self.data = self.get_data()
        self.load_method = self.one.load_object

    def get_data(self):
        """
        Load/ Download all data associated with probe
        """

        data = Bunch()
        # Load in spike sorting data
        data['spikes'], data['clusters'], data['channels'] = self.get_spikesorting_data()
        # Load in rms AP data
        data['rms_AP'] = self.get_rms_data(band='AP')
        # Load in rms LF data
        data['rms_LF'] = self.get_rms_data(band='LF')
        # Load in psd LF data
        data['psd_lf'] = self.get_psd_data(band='LF')
        # Load in passive data TODO make this cleverer, it should be shared across the probes
        data['rf_map'], data['pass_stim'], data['gabor'] = self.get_passive_data()

        return data


    def get_spikesorting_path(self):
        """
        Determine spike sorting path based on input collection or auto-detection.
        """

        probe_path = self.session_path.joinpath('alf', self.probe_label)

        if self.spike_collection == '':
            return probe_path
        elif self.spike_collection:
            return probe_path.joinpath(self.spike_collection)

        # Find all spike sorting collections
        all_collections = self.one.list_collections(self.eid)
        # iblsorter is default, then pykilosort
        for sorter in ['iblsorter', 'pykilosort']:
            if f'alf/{self.probe_label}/{sorter}' in all_collections:
                return probe_path.joinpath(sorter)
        # If neither exist return ks2 path
        return probe_path

    @staticmethod
    def load_data(load_function, *args, raise_message=None, raise_exception=ALFObjectNotFound,
                  raise_error=False, **kwargs):
        """
        Wrapper to load ONE data with logging and error handling.
        """
        alf_object = args[1]
        try:
            data = load_function(*args, **kwargs)
            if isinstance(data, (dict, Bunch)):
                data['exists'] = True
            return data
        except raise_exception as e:
            raise_message = raise_message or f'{alf_object} data was not found, some plots will not display'
            logger.warning(raise_message)
            if raise_error:
                logger.error(raise_message)
                logger.error(traceback.format_exc())
                raise e
            return {'exists': False}


    def get_passive_data(self):
        """
        Load passive stimulus data (RF map, visual stim, gabor stimuli).
        """
        # TO DO need to add in collections also improve hadnling of this will all these expections
        # Load in RFMAP data
        try:
            rf_data = self.load_data(self.one.load_object, self.eid, 'passiveRFM')
            frame_path = self.load_data(self.one.load_dataset,self.eid, '_iblrig_RFMapStim.raw.bin',
                                        download_only=True)
            frames = np.fromfile(frame_path, dtype="uint8")
            rf_data['frames'] = np.transpose(np.reshape(frames, [15, 15, -1], order="F"), [2, 1, 0])
        except Exception:
            logger.warning('rfmap data was not found, some plots will not display')
            rf_data = {}
            rf_data['exists'] = False

        # Load in passive stim data
        stim_data = self.load_data(self.one.load_object, self.eid, 'passiveStims')

        # Load in passive gabor data
        try:
            gabor = self.load_data(self.one.load_object, self.eid, 'passiveGabor')
            vis_stim = {}
            vis_stim['leftGabor'] = gabor['start'][(gabor['position'] == 35) & (gabor['contrast'] > 0.1)]
            vis_stim['rightGabor'] = gabor['start'][(gabor['position'] == -35) & (gabor['contrast'] > 0.1)]
            vis_stim['exists'] = True
        except Exception:
            logger.warning('passive gabor data was not found, some plots will not display')
            vis_stim = {}
            vis_stim['exists'] = False

        return rf_data, stim_data, vis_stim

    def get_rms_data(self, band='AP'):
        """
        Load RMS data for AP or LF band.
        """

        rms_data = self.load_data(
            self.one.load_object, self.eid, f'ephysTimeRms{band}',
            collection=f'raw_ephys_data/{self.probe_label}')

        if rms_data['exists']:
            if 'amps' in rms_data.keys():
                rms_data['rms'] = rms_data.pop('amps')
            if 'timestamps' not in rms_data.keys():
                rms_data['timestamps'] = np.array([0, rms_data['rms'].shape[0]])
                rms_data['xaxis'] = 'Time samples'
            else:
                rms_data['xaxis'] = 'Time (s)'

        return rms_data

    def get_psd_data(self, band='LF'):
        """
        Load PSD data for AP or LF band.
        """

        psd_data = self.load_data(
            self.one.load_object, self.eid, f'ephysSpectralDensity{band}',
            collection=f'raw_ephys_data/{self.probe_label}')

        if psd_data['exists']:
            if 'amps' in psd_data.keys():
                psd_data['power'] = psd_data.pop('amps')

        return psd_data


    def get_spikesorting_data(self, filter=True):
        """
        Load spike sorting data (spikes, clusters, channels) and filter by min_fr
        """

        spikes = self.load_data(
            self.one.load_object, self.eid, 'spikes', raise_error=False,
            collection=self.probe_collection, attribute=['depths', 'amps', 'times', 'clusters'])

        clusters = self.load_data(
            self.one.load_object, self.eid, 'clusters', raise_error=False,
            collection=self.probe_collection, attribute=['metrics', 'peakToTrough', 'waveforms', 'channels'])

        channels = self.load_data(
            self.one.load_object, self.eid, 'channels', raise_error=False,
            collection=self.probe_collection, attribute=['rawInd', 'localCoordinates'])

        if filter:
            # Remove low firing rate clusters
            spikes, clusters = self.filter_spikes_and_clusters(spikes, clusters)

        return spikes, clusters, channels

    @staticmethod
    def filter_spikes_and_clusters(spikes, clusters, min_fr=50/3600):
        """
        Remove low-firing clusters and filter spikes accordingly.
        """

        clu_idx = clusters.metrics.firing_rate > min_fr
        exists = clusters.pop('exists')
        clusters = Bunch({k: v[clu_idx] for k, v in clusters.items()})
        clusters['exists'] = exists

        spike_idx, ib = ismember(spikes.clusters, clusters.metrics.index)
        clusters.metrics.reset_index(drop=True, inplace=True)
        exists = spikes.pop('exists')
        spikes = Bunch({k: v[spike_idx] for k, v in spikes.items()})
        spikes['exists'] = exists
        spikes.clusters = clusters.metrics.index[ib].astype(np.int32)

        return spikes, clusters
