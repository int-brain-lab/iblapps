from abc import ABC, abstractmethod
import numpy as np
from iblutil.util import Bunch
import re
from datetime import timedelta
from atlaselectrophysiology.loaders.data_loader import DataLoaderONE, DataLoaderLocal, CollectionData
from atlaselectrophysiology.loaders.alignment_loader import AlignmentLoaderONE, AlignmentLoaderLocal
from atlaselectrophysiology.loaders.histology_loader import NrrdSliceLoader, download_histology_data
from atlaselectrophysiology.loaders.data_uploader import DataUploaderONE, DataUploaderLocal
from atlaselectrophysiology.loaders.plot_loader import PlotLoader
# from atlaselectrophysiology.plot_data import PlotData
from iblatlas.atlas import AllenAtlas
from one.api import ONE
from pathlib import Path
import pandas as pd
from collections import defaultdict
from one import params

from atlaselectrophysiology.plot_data import PlotData


# Combinations
# Session   - online | offline | offline
# Raw data  - online | offline | offline
# Histology - online | offline | online
# Picks     - online | offline | online
# Upload    - online | offline | online

# online



class ProbeLoader(ABC):
    def __init__(self, brain_atlas):
        self.brain_atlas = brain_atlas or AllenAtlas()
        self.probes = dict()
        self.probe_label = None
        self.configs = None
        self.selected_config=None
        self.config = None

    @abstractmethod
    def get_info(self, *args):
        """
        :param idx:
        :return:
        """

    def get_starting_alignment(self, idx):
        self.get_selected_probe().get_starting_alignment(idx)

    def get_previous_alignments(self):
        return self.get_selected_probe().get_previous_alignments()

    def get_selected_probe(self):
        return self.probes[self.probe_label]

    def set_init_alignment(self):
        self.get_selected_probe().loaders['align'].set_init_alignment()

    def offset_hist_data(self, val):
        self.get_selected_probe().loaders['align'].offset_hist_data(val)

    def scale_hist_data(self, line_track, line_feature, extend_feature=1, lin_fit=True):
        self.get_selected_probe().loaders['align'].scale_hist_data(line_track, line_feature, extend_feature=extend_feature, lin_fit=lin_fit)

    def next_idx(self):
        return self.get_selected_probe().loaders['align'].next_idx()

    def prev_idx(self):
        return self.get_selected_probe().loaders['align'].prev_idx()

    @property
    def current_idx(self):
        return self.get_selected_probe().loaders['align'].current_idx
    @property
    def total_idx(self):
        return self.get_selected_probe().loaders['align'].total_idx

    def reset_features_and_tracks(self):
        self.get_selected_probe().loaders['align'].reset_features_and_tracks()

    @property
    def chn_min(self):
        return np.min([0, self.get_selected_probe().plotdata.chn_min])

    @property
    def chn_max(self):
        return self.get_selected_probe().plotdata.chn_max

    def get_plots(self, plot):
        keys = []
        for shank in self.probes:
            keys += getattr(self.probes[shank], plot).keys()

        return set(keys)

    @property
    def img_plots(self):
        return self.get_plots('img_plots')

    @property
    def scat_plots(self):
        return self.get_plots('scatter_plots')

    @property
    def line_plots(self):
        return self.get_plots('line_plots')

    @property
    def probe_plots(self):
        return self.get_plots('probe_plots')

    @property
    def slice_plots(self):
        return self.get_plots('slice_plots')




    @abstractmethod
    def download_histology(self):
        """
        :return:
        """

    @abstractmethod
    def load_data(self):
        # Should this be here?
        """

        :return:
        """

    @staticmethod
    def normalize_probe_name(probe_name):
        match = re.match(r'(probe\d+)', probe_name)
        return match.group(1) if match else probe_name

    def data(self, shank):
        return self.probes[shank]



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
        self.shanks = [self.sess[idx] for idx in sess_idx]
        shanks = [s['name'] for s in self.shanks]
        # TODO better way to do this pleasee!
        idx = np.argsort(shanks)
        self.shanks = np.array(self.shanks)[idx]
        shanks = np.array(shanks)[idx]

        self.initialise_shanks()

        return list(shanks)

    def download_histology(self):
        _, hist_path = download_histology_data(self.subj, self.lab)
        self.slice_loader = NrrdSliceLoader(hist_path, self.brain_atlas)

    def load_data(self):
        self.download_histology()
        for probe in self.probes.keys():
            self.probes[probe].loaders['hist'] = self.slice_loader
            self.probes[probe].load_data()

    def initialise_shanks(self):

        self.probes = Bunch()

        for ins in self.shanks:
            loaders = Bunch()
            loaders['data'] = DataLoaderONE(ins, self.one, spike_collection=self.spike_collection)
            loaders['align'] = AlignmentLoaderONE(ins, self.one, self.brain_atlas)
            loaders['upload'] = DataUploaderONE(ins, self.one, self.brain_atlas)

            self.probes[ins['name']] = ShankLoader(loaders)

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

        self.probe_label = self.shanks[idx]['name']
        self.shank_idx = idx
        self.subj = self.shanks[idx]['session_info']['subject']
        self.lab = self.shanks[idx]['session_info']['lab']

    def upload_data(self):
        return self.get_selected_probe().upload_data()


# This is pretty bespoke to IBL situation atm
class ProbeLoaderCSV(ProbeLoader):
    def __init__(self, csv_file='/Users/admin/int-brain-lab/quarter.csv', one=None, brain_atlas=None):
        self.df = pd.read_csv(csv_file)
        self.df['session_strip'] = self.df['session'].str.rsplit('/', n=1).str[0]
        self.one = one or ONE()
        super().__init__(brain_atlas)
        self.configs = ['quarter', 'dense']
        # TODO having both config and selected_config seems unnecesarily confusing
        self.selected_config='both'
        self.config = self.selected_config

    def data(self, shank):
        return self.probes[shank][self.config]

    def get_subjects(self):
        # Returns sessions
        self.subjects = self.df['session_strip'].unique()

        return self.subjects

    def get_sessions(self, idx):
        # Returns probes
        self.session_df = self.df.loc[self.df['session_strip'] == self.subjects[idx]]
        self.sessions = np.unique([self.normalize_probe_name(pr) for pr in self.session_df['probe'].values])
        return self.sessions

    def get_shanks(self, idx):
        # Returns shank
        shank = self.sessions[idx]
        self.shank_df = self.session_df.loc[self.session_df['probe'].str.contains(shank)].sort_values('probe')

        self.initialise_shanks()

        self.shanks = self.shank_df['probe'].unique()

        return self.shanks

    def get_config(self, idx):

        return ['quarter', 'dense', 'both']


    def get_info(self, idx):
        self.probe_label = self.shanks[idx]
        self.shank_idx = idx
        # TODO don't harcode
        self.lab = 'steinmetzlab'
        self.subj = 'KM_027'

    def initialise_shanks(self):

        self.probes = defaultdict(Bunch)
        user = params.get().ALYX_LOGIN

        for _, shank in self.shank_df.iterrows():
            loaders = Bunch()
            # Quarter is offline
            if shank.is_quarter:
                loaders['data'] = DataLoaderLocal(Path(shank.local_path), CollectionData())
                loaders['align'] = AlignmentLoaderLocal(Path(shank.local_path), 0, 1, self.brain_atlas, user=user)
                loaders['upload'] = DataUploaderLocal(Path(shank.local_path),0, 1, self.brain_atlas, user=user)
                self.probes[shank.probe]['quarter'] = ShankLoader(loaders)
            else:
                # Dense is online
                ins = self.get_insertion(shank)
                collections = CollectionData(
                    spike_collection=f'alf/{shank.probe}/iblsorter',
                    ephys_collection=f'raw_ephys_data/{shank.probe}',
                    task_collection='alf/task_01',
                    raw_task_collection='raw_task_data_01')

                loaders['data'] = DataLoaderLocal(Path(shank.local_path), collections)
                loaders['align'] = AlignmentLoaderONE(ins, self.one, self.brain_atlas, user=user)
                loaders['upload'] = DataUploaderONE(ins, self.one, self.brain_atlas)
                self.probes[shank.probe]['dense'] = ShankLoader(loaders)

            # Assign previous alignments
            # If Alyx exists this one takes over, if not and a local exists we load this one
        for shank in self.probes.keys():
            if self.probes[shank]['dense'].loaders['align'].alignment_keys != ['original']:
                # Set the quarter shank to whatever is on alyx regardless if there is an original local file
                self.probes[shank]['quarter'].loaders['align'].alignments = self.probes[shank]['dense'].loaders['align'].alignments
                self.probes[shank]['quarter'].get_previous_alignments()
                self.probes[shank]['quarter'].get_starting_alignment(0)

            elif self.probes[shank]['quarter'].loaders['align'].alignment_keys != ['original']:
                # If we have a local file, populate the dense alignment with these values
                self.probes[shank]['dense'].loaders['align'].add_extra_alignments(self.probes[shank]['quarter'].loaders['align'].alignments)
                self.probes[shank]['dense'].get_previous_alignments()
                self.probes[shank]['dense'].get_starting_alignment(0)
                # TODO should we then set the quarter to the dense to make sure names are consistent??
                #  e should otherwise they are different when doing get_starting alignment they will be different
                self.probes[shank]['quarter'].loaders['align'].alignments = self.probes[shank]['dense'].loaders['align'].alignments
                self.probes[shank]['quarter'].get_previous_alignments()
                self.probes[shank]['quarter'].get_starting_alignment(0)


    def get_insertion(self, shank):

        ins = self.one.alyx.rest('insertions', 'list', session=self.one.path2eid(shank.session), name=shank.probe)
        return ins[0]

    # def get_quarter_density_alignment(self, shank):
    #     sub_date = '/'.join(shank.session.split('/')[:-1])
    #     quarter = self.df.loc[self.df['session'].str.contains(sub_date) & (self.df['probe'] == shank)  &
    #                           (self.df['is_quarter'] == True)]
    #     if len(quarter) == 1:
    #         prev_align = Path(quarter.iloc[0].local_path).joinpath('prev_alignments.json')
    #         if prev_align.exists():
    #             print(prev_align)
    #             return prev_align

    def download_histology(self):
        _, hist_path = download_histology_data(self.subj, self.lab)
        self.slice_loader = NrrdSliceLoader(hist_path, self.brain_atlas)

    def load_data(self):
        self.download_histology()
        for probe in self.probes.keys():
            for config in self.configs:
                self.probes[probe][config].loaders['hist'] = self.slice_loader
                self.probes[probe][config].load_data()

    def get_starting_alignment(self, idx):
        for config in self.configs:
            self.get_selected_probe()[config].get_starting_alignment(idx)

    def get_previous_alignments(self):
        # Always return the dense alignment
        return self.get_selected_probe()['dense'].get_previous_alignments()

    def get_selected_probe(self):
        return self.probes[self.probe_label]

    def set_init_alignment(self):
        for config in self.configs:
            self.get_selected_probe()[config].loaders['align'].set_init_alignment()

    def offset_hist_data(self, val):
        for config in self.configs:
            self.get_selected_probe()[config].loaders['align'].offset_hist_data(val)

    def scale_hist_data(self, line_track, line_feature, extend_feature=1, lin_fit=True):
        for config in self.configs:
            self.get_selected_probe()[config].loaders['align'].scale_hist_data(line_track, line_feature,
                                                                   extend_feature=extend_feature, lin_fit=lin_fit)

    def next_idx(self):
        for config in self.configs:
            next_idx = self.get_selected_probe()[config].loaders['align'].next_idx()
        return next_idx

    def prev_idx(self):
        for config in self.configs:
            prev_idx = self.get_selected_probe()[config].loaders['align'].prev_idx()
        return prev_idx

    @property
    def current_idx(self):
        return self.get_selected_probe()[self.config].loaders['align'].current_idx
    @property
    def total_idx(self):
        return self.get_selected_probe()[self.config].loaders['align'].total_idx

    def reset_features_and_tracks(self):
        for config in self.configs:
            self.get_selected_probe()[config].loaders['align'].reset_features_and_tracks()

    @property
    def chn_min(self):
        return np.min([0, self.get_selected_probe()[self.config].plotdata.chn_min])

    @property
    def chn_max(self):
        # TODO this loop won't do!
        return self.get_selected_probe()[self.config].plotdata.chn_max

    def get_plots(self, plot):
        keys = []
        for shank in self.probes:
            for config in self.configs:
                keys += getattr(self.probes[shank][config], plot).keys()

        return set(keys)

    def upload_data(self):

        for config in self.configs:
            info = self.get_selected_probe()[config].upload_data()
        # TODO we assume info is from alyx just because of the order of self.configs but should handle better
        return info









class ProbeLoaderLocal(ProbeLoader):
    def __init__(self, brain_atlas=None):
        super().__init__(brain_atlas)

    def get_info(self, idx):
        self.probe_label = f'shank_{self.shanks[idx]}'
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

        self.shanks = shank_list

        self.initiate_shanks()

        return shank_list

    # TODO
    # def get_shank_info(self):
    #
    #     if self.n_shanks > 1:
    #
    #         chn_x = np.unique(self.chn_coords_all[:, 0])
    #         groups = np.split(chn_x, np.where(np.diff(chn_x) > 100)[0] + 1)
    #
    #         assert len(groups) == self.n_shanks
    #
    #         shanks = {}
    #         for iShank, grp in enumerate(groups):
    #             if len(grp) == 1:
    #                 grp = np.array([grp[0], grp[0]])
    #             shanks[iShank] = [grp[0], grp[1]]
    #
    #         shank_chns = np.bitwise_and(self.chn_coords_all[:, 0] >= shanks[self.shank_idx][0],
    #                                     self.chn_coords_all[:, 0] <= shanks[self.shank_idx][1])
    #         self.orig_idx = np.where(shank_chns)[0]
    #         self.chn_coords = self.chn_coords_all[shank_chns, :]
    #
    #     else:
    #         self.orig_idx = None
    #         self.chn_coords = self.chn_coords_all
    #
    #     chn_depths = self.chn_coords[:, 1]


    def download_histology(self):
        # TODO make this so it can be another folder, some kind of path plugin to define where the data will be in relation to the selected path
        self.slice_loader = NrrdSliceLoader(self.folder_path, self.brain_atlas)

    def load_data(self):
        self.download_histology()
        for probe in self.probes.keys():
            self.probes[probe].loaders['hist'] = self.slice_loader
            self.probes[probe].load_data()

    def initiate_shanks(self):

        self.probes = Bunch()

        for ins in self.shanks:
            loaders = Bunch()
            # TODO add ins as the argument
            loaders['data'] = DataLoaderLocal(self.folder_path, CollectionData())
            loaders['align'] = AlignmentLoaderLocal(self.folder_path, ins, self.n_shanks, self.brain_atlas)
            loaders['upload'] = DataUploaderLocal(self.folder_path, ins, self.n_shanks, self.brain_atlas)

            self.probes[f'shank_{ins}'] = ShankLoader(loaders)







class ShankLoader:
    def __init__(self, loaders):
        self.loaders = loaders
        # TODO need to account for cases when no histology
        self.histology = True
        self.get_previous_alignments()
        self.get_starting_alignment(0)

        self.data_loaded = False

    def get_previous_alignments(self):
        return self.loaders['align'].get_previous_alignments()

    def get_starting_alignment(self, idx):
        self.loaders['align'].get_starting_alignment(idx)

    def load_data(self):
        # TODO cleaner way to do this
        if self.data_loaded:
            return

        self.raw_data = self.loaders['data'].get_data()

        self.chn_coords = self.raw_data['channels']['localCoordinates']
        self.chn_depths = self.chn_coords[:, 1]
        self.cluster_chns = self.raw_data['clusters']['channels']

        # Generate plots
        self.plotdata = PlotLoader(self.raw_data, 0)
        #self.plotdata = PlotData(self.loaders['data'].probe_path, self.raw_data, 0)
        self.get_plots()

        self.loaders['align'].get_align(self.chn_depths)

        self.slice_plots = self.loaders['hist'].get_slices(self.loaders['align'].xyz_samples)

        self.data_loaded = True

        # TODO better handling
        self.probe_path = self.loaders['data'].probe_path
        self.probe_collection =self.loaders['data'].probe_collection

    def get_plots(self):
        self.plotdata.get_plots()

        #TODO we don't need to do this, directly get items wihtout reassigning
        self.img_plots = self.plotdata.img_plots
        self.scatter_plots = self.plotdata.scat_plots
        self.probe_plots = self.plotdata.probe_plots
        self.line_plots = self.plotdata.line_plots

    def filter_plots(self, filter_type):

        self.plotdata.filter_units(filter_type)
        self.plotdata.get_data()
        self.plotdata.get_plots()

        self.img_plots = self.plotdata.img_plots
        self.scatter_plots = self.plotdata.scat_plots
        self.probe_plots = self.plotdata.probe_plots
        self.line_plots = self.plotdata.line_plots


    def upload_data(self):
        # TODO this is nasty, can we make it nicer (at leaset add dataclass)
        data = {'chn_coords': self.chn_coords,
                'xyz_channels': self.loaders['align'].xyz_channels,
                'feature': self.loaders['align'].feature.tolist(),
                'track': self.loaders['align'].track.tolist(),
                'alignments': self.loaders['align'].alignments,
                'cluster_chns': self.cluster_chns,
                'probe_collection': self.probe_collection,
                'probe_path': self.probe_path,
                'chn_depths': self.chn_depths,
                'xyz_picks': self.loaders['align'].xyz_picks,
                }
        return self.loaders['upload'].upload_data(data)




