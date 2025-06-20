import logging
import traceback
import numpy as np
from datetime import datetime
import ibllib.pipes.histology as histology
from ibllib.pipes.ephys_alignment import EphysAlignment
from neuropixel import trace_header
import iblatlas.atlas as atlas
from ibllib.qc.alignment_qc import AlignmentQC
from iblutil.numerical import ismember
from iblutil.util import Bunch
from one.api import ONE
from one.remote import aws
from pathlib import Path
import one.alf as alf
from one.alf.exceptions import ALFObjectNotFound
from one import params
import glob
import json
from atlaselectrophysiology.load_histology import download_histology_data, tif2nrrd
from atlaselectrophysiology.plot_data import PlotData
import ibllib.qc.critical_reasons as usrpmt
from datetime import timedelta
import re

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
            # Download bwm aggregate tables for for ephys feature gui
            table_path = self.one.cache_dir.joinpath('bwm_features')
            s3, bucket_name = aws.get_s3_from_alyx(alyx=self.one.alyx)
            aws.s3_download_folder("aggregates/bwm/latest", table_path, s3=s3, bucket_name=bucket_name)

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


        # self.n_sess = self.sess[idx]['session_info']['number']
        # self.date = self.sess[idx]['session_info']['start_time'][:10]
        # self.probe_label = self.sess[idx]['name']
        # self.probe_id = self.sess[idx]['id']
        # self.lab = self.sess[idx]['session_info']['lab']
        # self.eid = self.sess[idx]['session']
        # self.subj = self.sess[idx]['session_info']['subject']

        #
        # self.probes = Bunch()
        # #insertions = [self.sess[idx]] + self.get_other_shanks()
        # for ins in self.shanks:
        #     self.probes[ins['name']] = ProbeLoader(ins, self.one, self.brain_atlas)

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

    # TODO BETTER NAMING
    def load_data(self):
        for probe in self.probes.keys():
            self.probes[probe].load_data()

        # # TODO deal with histology exists
        # histology_exists = False
        #
        # # Make sure there is a histology track
        # hist_traj = self.one.alyx.rest('trajectories', 'list', probe_insertion=self.probe_id,
        #                                provenance='Histology track')
        # if len(hist_traj) == 1:
        #     if hist_traj[0]['x'] is not None:
        #         histology_exists = True
        #         self.traj_id = hist_traj[0]['id']
        #
        # return self.get_previous_alignments(), histology_exists

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
                                                     'datasets__name__icontains,spikes.times',
                                                     expires=timedelta(days=1))
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


    def get_other_shanks(self):

        insertions = self.one.alyx.rest('insertions', 'list', session=self.eid)
        # TODO IMPROVE HANDLING OF THIS
        insertions = [ins for ins in insertions if self.probe_label[:7] in ins['name'] and
                      self.probe_label != ins['name']]

        return insertions



    def load_session_notes(self):
        sess = self.one.alyx.rest('sessions', 'read', id=self.eid)
        sess_notes = None
        if sess['notes']:
            sess_notes = sess['notes'][0]['text']
        if not sess_notes:
            sess_notes = sess['narrative']
        if not sess_notes:
            sess_notes = 'No notes for this session'

        return sess_notes


class CircularIndexTracker:
    """
    A class to manage circular buffer indexing with tracking for navigation
    and overwriting behavior.

    Attributes
    ----------
    max_idx : int
        Size of the circular buffer.
    current_idx : int
        The current index in logical (non-wrapped) space.
    total_idx : int
        The highest index filled so far in logical space.
    last_idx : int
        The last recorded total index.
    diff_idx : int
        Offset between current index and last index used in reset logic.
    idx : int
        The wrapped index used in the circular buffer.
    idx_prev : int
        The previous wrapped index.
    """
    def __init__(self, max_idx: int):

        self.max_idx = max_idx
        self.current_idx = 0
        self.total_idx = 0
        self.last_idx = 0
        self.diff_idx = 0
        self.idx = 0
        self.idx_prev = 0

    def _update_diff_idx(self) -> None:
        """Internal method to update the diff_idx value."""
        if self.current_idx < self.last_idx:
            self.total_idx = self.current_idx
            delta = np.mod(self.last_idx, self.max_idx) - np.mod(self.total_idx, self.max_idx)
            self.diff_idx = self.max_idx - delta if delta >= 0 else np.abs(delta)
        else:
            self.diff_idx = self.max_idx - 1


    def next_idx_to_fill(self) -> None:
        """
        Advance to the next index in fill mode (as if appending new data to a circular buffer).
        """
        self._update_diff_idx()
        self.total_idx += 1
        self.current_idx += 1
        self.idx_prev = self.idx
        self.idx = np.mod(self.current_idx, self.max_idx)

    def previous_idx(self) -> bool:
        """
        Move backward through previously visited indices.
        """
        if self.total_idx > self.last_idx:
            self.last_idx = self.total_idx

        if self.current_idx > np.max([0, self.total_idx - self.diff_idx]):
            self.current_idx -= 1
            self.idx = np.mod(self.current_idx, self.max_idx)

            return True

    def next_idx(self) -> None:
        """
        Move forward through the buffer if within bounds.
        """
        if (self.current_idx < self.total_idx) & (self.current_idx > self.total_idx - self.max_idx):
            self.current_idx += 1
            self.idx = np.mod(self.current_idx, self.max_idx)

            return True
            # idx_prev is the fit
            # idx is the one that is being displayed

    def reset_idx(self) -> None:
        """
        Reset the index as if beginning a new overwrite sequence, recalculating the circular position.
        """
        self._update_diff_idx()
        self.total_idx += 1
        self.current_idx += 1
        self.idx = np.mod(self.current_idx, self.max_idx)



class AlignData:
    def __init__(self, xyz_picks, chn_depths, brain_atlas):

        self.buffer = CircularIndexTracker(10)
        self.brain_atlas = brain_atlas
        self.ephysalign = EphysAlignment(xyz_picks, chn_depths, brain_atlas=self.brain_atlas)
        self.initiate_features_and_tracks()
        self.hist_mapping = 'Allen'

        #
        # self.shank.align.features[self.idx], self.shank.align.track[self.idx], _ \
        #     = self.shank.align.ephysalign.get_track_and_feature()

    @property
    def idx(self):
        return self.buffer.idx

    @property
    def idx_prev(self):
        return self.buffer.idx_prev

    @property
    def xyz_channels(self):
        return self.ephysalign.get_channel_locations(self.features[self.idx], self.tracks[self.idx])

    @property
    def track_lines(self):
        return self.ephysalign.get_perp_vector(self.features[self.idx], self.tracks[self.idx])

    @property
    def xyz_track(self):
        return self.ephysalign.xyz_track

    @property
    def xyz_samples(self):
        return self.ephysalign.xyz_samples

    @property
    def track(self):
        return self.tracks[self.idx]

    @property
    def feature(self):
        return self.features[self.idx]

    def set_init_feature_track(self, feature, track):
        self.ephysalign.feature_init = feature
        self.ephysalign.track_init = track

    def initiate_features_and_tracks(self):
        self.tracks = [0] * (self.buffer.max_idx + 1)
        self.features = [0] * (self.buffer.max_idx + 1)

    def reset_features_and_tracks(self):
        self.buffer.reset_idx()
        self.tracks[self.idx] = self.ephysalign.track_init
        self.features[self.idx] = self.ephysalign.feature_init

    def get_scaled_histology(self):
        hist_data = {}
        scale_data = {}
        hist_data_ref = {}
        if self.hist_mapping == 'FP':
            region_label = self.region_label_fp
            region = self.region_fp
            colour = self.region_colour_fp
        else:
            region_label = None
            region = None
            colour = self.ephysalign.region_colour


        hist_data['region'], hist_data['axis_label'] = self.ephysalign.scale_histology_regions(
            self.features[self.idx], self.tracks[self.idx], region=region, region_label=region_label)
        hist_data['colour'] = colour

        scale_data['region'], scale_data['scale'] = self.ephysalign.get_scale_factor(hist_data['region'], region_orig=region)
        hist_data_ref['region'], hist_data_ref['axis_label'] = self.ephysalign.scale_histology_regions(
            self.ephysalign.track_extent, self.ephysalign.track_extent, region=region, region_label=region_label)

        hist_data_ref['colour'] = colour

        return hist_data, hist_data_ref, scale_data

    def offset_hist_data(self, offset):

        self.buffer.next_idx_to_fill()
        self.tracks[self.idx] = (self.tracks[self.idx_prev] + offset)
        self.features[self.idx] = (self.features[self.idx_prev])


    def scale_hist_data(self, line_track, line_feature, extend_feature=1, lin_fit=True):
        """
        Scale brain regions along probe track
        """
        self.buffer.next_idx_to_fill()
        depths_track = np.sort(np.r_[self.tracks[self.idx_prev][[0, -1]], line_track])


        self.tracks[self.idx] = self.ephysalign.feature2track(depths_track,
                                                             self.features[self.idx_prev],
                                                             self.tracks[self.idx_prev])

        self.features[self.idx] = np.sort(np.r_[self.features[self.idx_prev]
                                                [[0, -1]], line_feature])



        if (self.features[self.idx].size >= 5) & lin_fit:
            self.features[self.idx], self.tracks[self.idx] = \
                self.ephysalign.adjust_extremes_linear(self.features[self.idx], self.tracks[self.idx], extend_feature)

        else:
            self.tracks[self.idx] = self.ephysalign.adjust_extremes_uniform(self.features[self.idx],
                                                                           self.tracks[self.idx])

    def compute_nearby_boundaries(self):

        nearby_bounds = self.ephysalign.get_nearest_boundary(self.ephysalign.xyz_samples,
                                                             self.allen, steps=6,
                                                             brain_atlas=self.brain_atlas)
        [self.hist_nearby_x, self.hist_nearby_y,
         self.hist_nearby_col] = self.ephysalign.arrange_into_regions(
            self.ephysalign.sampling_trk, nearby_bounds['id'], nearby_bounds['dist'],
            nearby_bounds['col'])

        [self.hist_nearby_parent_x,
         self.hist_nearby_parent_y,
         self.hist_nearby_parent_col] = self.ephysalign.arrange_into_regions(
            self.ephysalign.sampling_trk, nearby_bounds['parent_id'], nearby_bounds['parent_dist'],
            nearby_bounds['parent_col'])







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


    def load_data(self):
        # TODO cleaner way to do this
        if not self.data_loaded:
            self.raw_data = DataLoader(self.one, self.insertion['session'], self.insertion['name'],
                                   spike_collection=self.spike_collection)

            self.chn_coords = self.raw_data.data['channels']['localCoordinates']
            self.chn_depths = self.chn_coords[:, 1]

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

            self.slice_data, self.fp_slice_data = self.get_slice_images(self.align.xyz_samples)

        # Set the featuer to the current chosen alignment
        self.align.set_init_feature_track(self.feature_prev, self.track_prev)


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
        hist_path_cb = None
        # First see if the histology file exists before attempting to connect with FlatIron and
        # download
        if self.download_hist:
            hist_dir = Path(self.session_path.parent.parent, 'histology')
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

            files = download_histology_data('MB059', 'hausserlab')
            if files is not None:
                hist_path_cb = files[1]

        if hist_path_rd:
            hist_atlas_rd = atlas.AllenAtlas(hist_path=hist_path_rd)
            hist_slice_rd = hist_atlas_rd.image[index[:, 0], :, index[:, 2]]
            hist_slice_rd = np.swapaxes(hist_slice_rd, 0, 1)
            del hist_atlas_rd
        else:
            print('Could not find red histology image for this subject')
            hist_slice_rd = np.copy(ccf_slice)

        if hist_path_gr:
            hist_atlas_gr = atlas.AllenAtlas(hist_path=hist_path_gr)
            hist_slice_gr = hist_atlas_gr.image[index[:, 0], :, index[:, 2]]
            hist_slice_gr = np.swapaxes(hist_slice_gr, 0, 1)
            del hist_atlas_gr
        else:
            print('Could not find green histology image for this subject')
            hist_slice_gr = np.copy(ccf_slice)

        if hist_path_cb:
            hist_atlas_cb = atlas.AllenAtlas(hist_path=hist_path_cb)
            hist_slice_cb = hist_atlas_cb.image[index[:, 0], :, index[:, 2]]
            hist_slice_cb = np.swapaxes(hist_slice_cb, 0, 1)
            del hist_atlas_cb
        else:
            print('Could not find example cerebellar histology image')
            hist_slice_cb = np.copy(ccf_slice)

        slice_data = {
            'hist_rd': hist_slice_rd,
            'hist_gr': hist_slice_gr,
            'hist_cb': hist_slice_cb,
            'ccf': ccf_slice,
            'label': label_slice,
            'scale': np.array([(width[-1] - width[0]) / ccf_slice.shape[0],
                               (height[-1] - height[0]) / ccf_slice.shape[1]]),
            'offset': np.array([width[0], height[0]])
        }

        if self.franklin_atlas is not None:
            index = self.franklin_atlas.bc.xyz2i(xyz_channels)[:, self.franklin_atlas.xyz2dims]
            label_slice = self.franklin_atlas._label2rgb(self.franklin_atlas.label[index[:, 0], :, index[:, 2]])
            label_slice = np.swapaxes(label_slice, 0, 1)
            width = [self.franklin_atlas.bc.i2x(0), self.franklin_atlas.bc.i2x(456)]
            height = [self.franklin_atlas.bc.i2z(index[0, 2]), self.franklin_atlas.bc.i2z(index[-1, 2])]

            franklin_slice_data = {
                'label': label_slice,
                'scale': np.array([(width[-1] - width[0]) / ccf_slice.shape[0],
                                   (height[-1] - height[0]) / ccf_slice.shape[1]]),
                'offset': np.array([width[0], height[0]])
            }
        else:
            franklin_slice_data = None

        return slice_data, franklin_slice_data



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

    def upload_dj(self, align_qc, ephys_qc, ephys_desc):
        # Upload qc results to datajoint table

        if len(ephys_desc) == 0:
            ephys_desc_str = 'None'
        else:
            ephys_desc_str = ", ".join(ephys_desc)

        self.alyx_str = ephys_qc.upper() + ': ' + ephys_desc_str

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
