from abc import ABC, abstractmethod
import numpy as np
from iblutil.util import Bunch
import re
from datetime import timedelta
from atlaselectrophysiology.loaders.data_loader import DataLoaderONE, DataLoaderLocal, CollectionData
from atlaselectrophysiology.loaders.alignment_loader import AlignmentLoaderONE, AlignmentLoaderLocal
from atlaselectrophysiology.loaders.histology_loader import NrrdSliceLoader, download_histology_data
from atlaselectrophysiology.loaders.data_uploader import DataUploaderONE, DataUploaderLocal
from atlaselectrophysiology.plot_data import PlotData
from iblatlas.atlas import AllenAtlas
from one.api import ONE
from pathlib import Path
import pandas as pd


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

        return list(shanks) + ['all']

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


# This is pretty bespoke to IBL situation atm
class ProbeLoaderCSV(ProbeLoader):
    def __init__(self, csv_file='/Users/admin/int-brain-lab/quarter.csv', one=None, brain_atlas=None):
        self.df = pd.read_csv(csv_file)
        self.sessions = self.df['session'].unique()
        self.one = one or ONE()
        super().__init__(brain_atlas)

    def get_subjects(self):
        self.subjects = self.df['session'].unique()

        return self.subjects

    def get_sessions(self, idx):
        self.session_df = self.df.loc[self.df['session'] == self.subjects[idx]]
        self.sessions = np.unique([self.normalize_probe_name(pr) for pr in self.session_df['probe'].values])
        return self.sessions

    def get_shanks(self, idx):
        shank = self.sessions[idx]
        self.shank_df = self.session_df.loc[self.session_df['probe'].str.contains(shank)].sort_values('probe')

        self.initialise_shanks()

        self.shanks = self.shank_df['probe'].values

        return self.shanks

    def get_info(self, idx):
        self.probe_label = self.shanks[idx]
        self.shank_idx = idx
        self.lab = 'steinmetzlab'
        self.subj = 'KM_027'

    def initialise_shanks(self):

        self.probes = Bunch()

        for _, shank in self.shank_df.iterrows():
            loaders = Bunch()
            if shank.is_quarter:
                loaders['data'] = DataLoaderLocal(Path(shank.local_path), CollectionData())
                loaders['align'] = AlignmentLoaderLocal(Path(shank.local_path), 0, 1, self.brain_atlas)
                loaders['upload'] = DataUploaderLocal(Path(shank.local_path),0, 1, self.brain_atlas)
            else:
                ins = self.get_insertion(shank)
                collections = CollectionData(
                    spike_collection=f'alf/{shank.probe}/iblsorter',
                    ephys_collection=f'raw_ephys_data/{shank.probe}',
                    task_collection='alf/task_01',
                    raw_task_collection='raw_task_data_01')
                loaders['data'] = DataLoaderLocal(Path(shank.local_path), collections)
                loaders['align'] = AlignmentLoaderONE(ins, self.one, self.brain_atlas)
                quarter = self.get_quarter_density_alignment(shank)
                if quarter:
                    loaders['align'].add_extra_alignments(quarter)
                loaders['upload'] = DataUploaderONE(ins, self.one, self.brain_atlas)

            self.probes[shank.probe] = ShankLoader(loaders)

    def get_insertion(self, shank):

        ins = self.one.alyx.rest('insertions', 'list', session=self.one.path2eid(shank.session), name=shank.probe)
        return ins[0]

    def get_quarter_density_alignment(self, shank):
        sub_date = '/'.join(shank.session.split('/')[:-1])
        quarter = self.df.loc[self.df['session'].str.contains(sub_date) & (self.df['probe'] == shank.probe)  &
                              (self.df['is_quarter'] == True)]
        if len(quarter) == 1:
            prev_align = Path(quarter.iloc[0].local_path).joinpath('prev_alignments.json')
            if prev_align.exists():
                print(prev_align)
                return prev_align

    def download_histology(self):
        _, hist_path = download_histology_data(self.subj, self.lab)
        self.slice_loader = NrrdSliceLoader(hist_path, self.brain_atlas)

    def load_data(self):
        self.download_histology()
        for probe in self.probes.keys():
            self.probes[probe].loaders['hist'] = self.slice_loader
            self.probes[probe].load_data()







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
        self.plotdata = PlotData(self.loaders['data'].probe_path, self.raw_data, 0)
        self.get_plots()

        self.loaders['align'].get_align(self.chn_depths)

        self.slice_plots = self.loaders['hist'].get_slices(self.loaders['align'].xyz_samples)

        self.data_loaded = True

        # TODO better handling
        self.probe_path = self.loaders['data'].probe_path
        self.probe_collection =self.loaders['data'].probe_collection

    def get_plots(self):
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
        # TODO handle this in the data loader
        # self.img_plots.update(self.plotdata.get_raw_data_image(self.probe_id, one=self.one))
        self.img_plots.update(self.plotdata.get_passive_events())

        self.probe_plots.update(self.plotdata.get_rfmap_data())

        self.scatter_plots.update(self.plotdata.get_depth_data_scatter())
        self.scatter_plots.update(self.plotdata.get_fr_p2t_data_scatter())

        self.line_plots.update(self.plotdata.get_fr_amp_data_line())

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
        # self.img_plots.update(self.plotdata.get_raw_data_image(self.probe_id, one=self.one))
        self.img_plots.update(self.plotdata.get_passive_events())

        self.probe_plots.update(self.plotdata.get_rfmap_data())

        self.scatter_plots.update(self.plotdata.get_depth_data_scatter())
        self.scatter_plots.update(self.plotdata.get_fr_p2t_data_scatter())

        self.line_plots.update(self.plotdata.get_fr_amp_data_line())

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




