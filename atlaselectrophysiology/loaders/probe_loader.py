from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import timedelta
import numpy as np
import pandas as pd
from pathlib import Path
import re

from atlaselectrophysiology.loaders.alignment_loader import AlignmentLoaderONE, AlignmentLoaderLocal
from atlaselectrophysiology.loaders.data_loader import (DataLoaderONE, DataLoaderLocal, CollectionData,
                                                        SpikeGLXLoaderLocal, SpikeGLXLoaderONE)
from atlaselectrophysiology.loaders.data_uploader import DataUploaderONE, DataUploaderLocal
from atlaselectrophysiology.loaders.histology_loader import NrrdSliceLoader, download_histology_data
from atlaselectrophysiology.loaders.plot_loader import PlotLoader
from iblatlas.atlas import AllenAtlas
from iblutil.util import Bunch
from one import params
from one.api import ONE

# TODO docstrings and typing and logging
# TODO local implementation dealing with multi shanks

class ProbeLoader(ABC):
    def __init__(self, brain_atlas):
        self.brain_atlas = brain_atlas or AllenAtlas()

        self.shanks = Bunch()

        self.configs = None
        self.selected_config = None
        self.current_config = None

        self.selected_shank = None
        self.current_shank = None

    def get_current_shank(self):
        return self.shanks[self.current_shank]

    def get_selected_shank(self):
        return self.shanks[self.selected_shank]

    # Access methods in loaders['align']
    def load_previous_alignments(self):
        return self.get_selected_shank().loaders['align'].load_previous_alignments()

    def get_starting_alignment(self, idx):
        self.get_selected_shank().loaders['align'].get_starting_alignment(idx)

    def get_previous_alignments(self):
        return self.get_selected_shank().loaders['align'].get_previous_alignments()

    def set_init_alignment(self):
        self.get_current_shank().loaders['align'].set_init_alignment()

    @property
    def feature_prev(self):
        return self.get_current_shank().loaders['align'].feature_prev

    # Access methods in loaders['align'].align
    def offset_hist_data(self, val):
        self.get_current_shank().loaders['align'].align.offset_hist_data(val)

    def scale_hist_data(self, line_track, line_feature, extend_feature=1, lin_fit=True):
        self.get_current_shank().loaders['align'].align.scale_hist_data(
            line_track, line_feature, extend_feature=extend_feature, lin_fit=lin_fit)

    def get_scaled_histology(self):
        return self.get_current_shank().loaders['align'].align.get_scaled_histology()

    def feature2track_lin(self, depths, feature, track):
        return self.get_current_shank().loaders['align'].align.ephysalign.feature2track_lin(depths, feature, track)

    def reset_features_and_tracks(self):
        self.get_current_shank().loaders['align'].align.reset_features_and_tracks()

    def next_idx(self):
        return self.get_selected_shank().loaders['align'].align.next_idx()

    def prev_idx(self):
        return self.get_selected_shank().loaders['align'].align.prev_idx()

    @property
    def current_idx(self):
        return self.get_selected_shank().loaders['align'].align.current_idx

    @property
    def total_idx(self):
        return self.get_selected_shank().loaders['align'].align.total_idx

    @property
    def track(self):
        return self.get_current_shank().loaders['align'].align.track

    @property
    def feature(self):
        return self.get_current_shank().loaders['align'].align.feature

    @property
    def xyz_channels(self):
        return self.get_current_shank().loaders['align'].align.xyz_channels

    @property
    def xyz_track(self):
        return self.get_current_shank().loaders['align'].align.xyz_track

    @property
    def track_lines(self):
        return self.get_current_shank().loaders['align'].align.track_lines

    # Access methods in plotdata
    @property
    def chn_min(self):
        return np.min([0, self.get_current_shank().loaders['plots'].chn_min])

    @property
    def chn_max(self):
        return self.get_current_shank().loaders['plots'].chn_max

    def get_plots(self, plot):
        keys = []
        for shank in self.shanks:
            keys += getattr(self.shanks[shank].loaders['plots'], plot).keys()

        return sorted(set(keys))

    @property
    def image_plots(self):
        return self.get_current_shank().loaders['plots'].image_plots

    @property
    def image_keys(self):
        return self.get_plots('image_plots')

    @property
    def scatter_plots(self):
        return self.get_current_shank().loaders['plots'].scatter_plots

    @property
    def scatter_keys(self):
        return self.get_plots('scatter_plots')

    @property
    def line_plots(self):
        return self.get_current_shank().loaders['plots'].line_plots

    @property
    def line_keys(self):
        return self.get_plots('line_plots')

    @property
    def probe_plots(self):
        return self.get_current_shank().loaders['plots'].probe_plots

    @property
    def probe_keys(self):
        return self.get_plots('probe_plots')

    @property
    def slice_plots(self):
        return self.get_current_shank().loaders['plots'].slice_plots

    @property
    def slice_keys(self):
        return self.get_plots('slice_plots')

    def upload_data(self):
        return self.get_selected_shank().upload_data()

    @staticmethod
    def normalize_shank_label(shank_label):
        match = re.match(r'(probe\d+)', shank_label)
        return match.group(1) if match else shank_label

    # Methods that must be implemented
    @abstractmethod
    def get_info(self, *args):
        pass

    @abstractmethod
    def download_histology(self):
        pass

    @abstractmethod
    def load_data(self):
        pass


class ProbeLoaderONE(ProbeLoader):
    def __init__(self, one=None, brain_atlas=None, spike_collection=None):
        self.one = one or ONE()
        self.spike_collection = spike_collection
        super().__init__(brain_atlas)

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

        return self.sessions

    def get_shanks(self, idx):
        sess = self.sessions[idx]

        sess_idx = [i for i, e in enumerate(self.sess) if self.get_session_probe_name(e) == sess]
        self.shank_labels = [self.sess[idx] for idx in sess_idx]
        shanks = [s['name'] for s in self.shank_labels]
        # TODO better way to do this pleasee!
        idx = np.argsort(shanks)
        self.shank_labels = np.array(self.shank_labels)[idx]
        shanks = np.array(shanks)[idx]

        self.initialise_shanks()

        return list(shanks)

    def download_histology(self):
        _, hist_path = download_histology_data(self.subj, self.lab)
        self.slice_loader = NrrdSliceLoader(hist_path, self.brain_atlas)

    def load_data(self):
        self.download_histology()
        for shank in self.shanks.keys():
            self.shanks[shank].loaders['hist'] = self.slice_loader
            self.shanks[shank].load_data()

    def initialise_shanks(self):

        self.shanks = Bunch()

        for ins in self.shank_labels:
            loaders = Bunch()
            loaders['data'] = DataLoaderONE(ins, self.one, spike_collection=self.spike_collection)
            loaders['align'] = AlignmentLoaderONE(ins, self.one, self.brain_atlas)
            loaders['upload'] = DataUploaderONE(ins, self.one, self.brain_atlas)
            loaders['ephys'] = SpikeGLXLoaderONE(ins, self.one)

            self.shanks[ins['name']] = ShankLoader(loaders)

    def get_session_probe_name(self, ins):
        return ins['session_info']['start_time'][:10] + ' ' + self.normalize_shank_label(ins['name'])

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

        self.selected_shank = self.shank_labels[idx]['name']
        self.current_shank = self.selected_shank
        self.shank_idx = idx
        self.subj = self.shank_labels[idx]['session_info']['subject']
        self.lab = self.shank_labels[idx]['session_info']['lab']


# This is pretty bespoke to IBL situation atm
class ProbeLoaderCSV(ProbeLoader):
    def __init__(self, csv_file='/Users/admin/int-brain-lab/quarter.csv', one=None, brain_atlas=None):
        super().__init__(brain_atlas)

        self.df = pd.read_csv(csv_file)
        self.df['session_strip'] = self.df['session'].str.rsplit('/', n=1).str[0]
        self.one = one or ONE()

        self.configs = ['quarter', 'dense']
        self.selected_config = 'both'
        self.current_config = self.selected_config

    def get_subjects(self):
        # Returns sessions
        self.subjects = self.df['session_strip'].unique()
        return self.subjects

    def get_sessions(self, idx):
        # Returns probes
        self.session_df = self.df.loc[self.df['session_strip'] == self.subjects[idx]]
        self.sessions = np.unique([self.normalize_shank_label(pr) for pr in self.session_df['probe'].values])
        return self.sessions

    def get_shanks(self, idx):
        # Returns shank
        shank = self.sessions[idx]
        self.shank_df = self.session_df.loc[self.session_df['probe'].str.contains(shank)].sort_values('probe')
        self.initialise_shanks()
        self.shank_labels = self.shank_df['probe'].unique()
        return self.shank_labels

    def get_config(self, idx):

        return ['quarter', 'dense', 'both']

    def get_info(self, idx):
        self.selected_shank = self.shank_labels[idx]
        self.current_shank = self.selected_shank
        self.shank_idx = idx
        # TODO don't harcode
        self.lab = 'steinmetzlab'
        self.subj = 'KM_027'

    def initialise_shanks(self):

        self.shanks = defaultdict(Bunch)
        user = params.get().ALYX_LOGIN

        for _, shank in self.shank_df.iterrows():
            loaders = Bunch()
            # Quarter is offline
            if shank.is_quarter:
                loaders['data'] = DataLoaderLocal(Path(shank.local_path), CollectionData())
                loaders['align'] = AlignmentLoaderLocal(Path(shank.local_path), 0, 1, self.brain_atlas, user=user)
                loaders['upload'] = DataUploaderLocal(Path(shank.local_path), 0, 1, self.brain_atlas, user=user)
                loaders['ephys'] = SpikeGLXLoaderLocal(Path(shank.local_path), '')
                self.shanks[shank.probe]['quarter'] = ShankLoader(loaders)
            else:
                # Dense is online
                ins = self.get_insertion(shank)
                collections = CollectionData(
                    spike_collection=f'alf/{shank.probe}/iblsorter',
                    ephys_collection=f'raw_ephys_data/{shank.probe}',
                    task_collection='alf/task_01',
                    raw_task_collection='raw_task_data_01',
                    meta_collection=f'raw_ephys_data/{shank.probe}')

                loaders['data'] = DataLoaderLocal(Path(shank.local_path), collections)
                loaders['align'] = AlignmentLoaderONE(ins, self.one, self.brain_atlas, user=user)
                loaders['upload'] = DataUploaderONE(ins, self.one, self.brain_atlas)
                loaders['ephys'] = SpikeGLXLoaderONE(ins, self.one)
                self.shanks[shank.probe]['dense'] = ShankLoader(loaders)

        self._sync_alignments()

    def _sync_alignments(self):
        """Synchronize alignments between dense and quarter loaders."""
        for probe, shank_group in self.shanks.items():
            dense_align = shank_group['dense'].loaders['align']
            quarter_align = shank_group['quarter'].loaders['align']

            if dense_align.alignment_keys != ['original']:
                # Alyx alignment exists: overwrite local
                quarter_align.alignments = dense_align.alignments
                quarter_align.get_previous_alignments()
                quarter_align.get_starting_alignment(0)

            elif quarter_align.alignment_keys != ['original']:
                # Local alignment exists: add to online
                dense_align.add_extra_alignments(quarter_align.alignments)
                dense_align.get_starting_alignment(0)

                # Ensure consistency by syncing quarter with updated dense
                quarter_align.alignments = dense_align.alignments
                quarter_align.get_previous_alignments()
                quarter_align.get_starting_alignment(0)

    def get_insertion(self, shank):

        ins = self.one.alyx.rest('insertions', 'list', session=self.one.path2eid(shank.session), name=shank.probe)
        return ins[0]

    def download_histology(self):
        _, hist_path = download_histology_data(self.subj, self.lab)
        self.slice_loader = NrrdSliceLoader(hist_path, self.brain_atlas)

    def load_data(self):
        self.download_histology()
        for probe in self.shanks.keys():
            for config in self.configs:
                self.shanks[probe][config].loaders['hist'] = self.slice_loader
                self.shanks[probe][config].load_data()

    def get_current_shank(self):
        return self.shanks[self.current_shank][self.current_config]

    def get_selected_shank(self):
        # TODO CHANGE THIS ONCE WE ARE HAPPY IT WORKS!
        return self.shanks[self.selected_shank]

    def get_starting_alignment(self, idx):
        for config in self.configs:
            self.get_selected_shank()[config].loaders['align'].get_starting_alignment(idx)

    def load_previous_alignments(self):
        self.get_selected_shank()['dense'].loaders['align'].load_previous_alignments()
        self.get_selected_shank()['quarter'].loaders['align'].alignments = self.get_selected_shank()['dense'].loaders[
            'align'].alignments
        self.get_selected_shank()['quarter'].loaders['align'].get_previous_alignments()

        return self.get_selected_shank()['dense'].loaders['align'].get_previous_alignments()

    def get_previous_alignments(self):
        # Always return the dense alignment
        return self.get_selected_shank()['dense'].loaders['align'].get_previous_alignments()

    def set_init_alignment(self):
        for config in self.configs:
            self.get_selected_shank()[config].loaders['align'].set_init_alignment()

    # TODO can we do this in the shank loop???
    def next_idx(self):
        for config in self.configs:
            la = self.get_selected_shank()[config].loaders['align'].align.next_idx()
        return la

    def prev_idx(self):
        for config in self.configs:
            self.config = config
            la = self.get_selected_shank()[config].loaders['align'].align.prev_idx()
        return la

    @property
    def current_idx(self):
        return self.get_selected_shank()[self.current_config].loaders['align'].align.current_idx

    @property
    def total_idx(self):
        return self.get_selected_shank()[self.current_config].loaders['align'].align.total_idx

    @property
    def chn_min(self):
        config = 'quarter' if self.selected_config == 'both' else self.selected_config
        return np.min([0, self.get_selected_shank()[config].loaders['plots'].chn_min])

    @property
    def chn_max(self):
        config = 'quarter' if self.selected_config == 'both' else self.selected_config
        return self.get_selected_shank()[config].loaders['plots'].chn_max

    def get_plots(self, plot):
        keys = []
        for shank in self.shanks:
            for config in self.configs:
                keys += getattr(self.shanks[shank][config].loaders['plots'], plot).keys()

        return sorted(set(keys))

    def upload_data(self):
        info = Bunch()
        for config in self.configs:
            info[config] = self.get_selected_shank()[config].upload_data()
        return info['dense']


class ProbeLoaderLocal(ProbeLoader):
    def __init__(self, brain_atlas=None):
        super().__init__(brain_atlas)

    def get_info(self, idx):
        self.probe_label = f'shank_{self.shank_labels[idx]}'
        self.shank_idx = idx

    def get_shanks(self, folder_path):
        """
        Find out the number of shanks on the probe, either 1 or 4
        """
        self.folder_path = Path(folder_path)

        self.chn_coords_all = np.load(self.folder_path.joinpath('channels.localCoordinates.npy'))
        chn_x = np.unique(self.chn_coords_all[:, 0])
        chn_x_diff = np.diff(chn_x)
        self.n_shanks = np.sum(chn_x_diff > 100) + 1

        if self.n_shanks == 1:
            shank_list = ['1/1']
        else:
            shank_list = [f'{iShank + 1}/{self.n_shanks}' for iShank in range(self.n_shanks)]

        self.shank_labels = shank_list

        self.initiate_shanks()

        # TODO add logic for finding the channels associated with shanks to get self.orig_idx (see original code)

        return shank_list

    def download_histology(self):
        # TODO make this so it can be another folder, some kind of path plugin to define
        #  where the data will be in relation to the selected path
        self.slice_loader = NrrdSliceLoader(self.folder_path, self.brain_atlas)

    def load_data(self):
        self.download_histology()
        for probe in self.shanks.keys():
            self.shanks[probe].loaders['hist'] = self.slice_loader
            self.shanks[probe].load_data()

    def initiate_shanks(self):

        self.shanks = Bunch()

        for ishank in self.shank_labels:
            loaders = Bunch()
            loaders['data'] = DataLoaderLocal(self.folder_path, CollectionData())
            loaders['align'] = AlignmentLoaderLocal(self.folder_path, ishank, self.n_shanks, self.brain_atlas)
            loaders['upload'] = DataUploaderLocal(self.folder_path, ishank, self.n_shanks, self.brain_atlas)

            self.shanks[f'shank_{ishank}'] = ShankLoader(loaders)


class ShankLoader:
    def __init__(self, loaders):
        self.loaders = loaders
        self.loaders['align'].load_previous_alignments()
        self.loaders['align'].get_starting_alignment(0)
        self.align_exists = True
        self.data_loaded = False

    def load_data(self):
        if self.data_loaded:
            return

        # Load data
        self.meta_data = self.loaders['ephys'].get_meta_data()
        self.raw_data = self.loaders['data'].get_data()
        self.raw_data['raw_snippets'] = self.loaders['ephys'].load_ap_snippets()

        if self.meta_data['exists']:
            self.chn_coords = np.c_[self.meta_data['x'], self.meta_data['y']]
            self.chn_depths = self.chn_coords[:, 1]
        elif self.raw_data['channels']['exists']:
            self.chn_coords = self.raw_data['channels']['localCoordinates']
            self.chn_depths = self.chn_coords[:, 1]
        else:
            self.chn_coords = None
            self.chn_depths = None

        if self.raw_data['clusters']['exists']:
            self.cluster_chns = self.raw_data['clusters']['channels']
        else:
            self.cluster_chns = np.arange(self.chn_depths.size)

        # Generate plots
        self.loaders['plots'] = PlotLoader(self.raw_data, self.meta_data, 0)
        self.loaders['plots'].get_plots()

        if self.chn_coords is not None:
            # Load the alignment handler
            # TODO Also need to set align_exists here based on presence of xyz_picks
            self.loaders['align'].load_align(self.chn_depths)
            # Load in the histology data
            self.loaders['plots'].slice_plots = self.loaders['hist'].get_slices(
                self.loaders['align'].align.xyz_samples)
        else:
            self.align_exists = False

        self.data_loaded = True

    def filter_plots(self, filter_type):

        self.loaders['plots'].filter_units(filter_type)
        self.loaders['plots'].get_data()
        self.loaders['plots'].get_plots()

    def upload_data(self):
        # TODO use a dataclass
        data = {'chn_coords': self.chn_coords,
                'xyz_channels': self.loaders['align'].xyz_channels,
                'feature': self.loaders['align'].feature.tolist(),
                'track': self.loaders['align'].track.tolist(),
                'alignments': self.loaders['align'].alignments,
                'cluster_chns': self.cluster_chns,
                'probe_collection': self.loaders['data'].probe_collection,
                'probe_path': self.loaders['data'].probe_path,
                'chn_depths': self.chn_depths,
                'xyz_picks': self.loaders['align'].xyz_picks,
                }
        return self.loaders['upload'].upload_data(data)
