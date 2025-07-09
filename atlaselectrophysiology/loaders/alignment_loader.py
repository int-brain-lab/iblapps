from abc import ABC, abstractmethod
from datetime import timedelta
import json
import numpy as np

from atlaselectrophysiology.utils.align import AlignData
from iblutil.util import Bunch

# TODO docstrings and typing and logging


class AlignmentLoader(ABC):
    def __init__(self, brain_atlas, user, xyz_picks=None):
        self.brain_atlas = brain_atlas

        self.alignments = Bunch()
        self.alignment_keys = ['original']
        self.xyz_picks = self.load_xyz_picks() if xyz_picks is None else xyz_picks

        self.user = user

    @abstractmethod
    def load_alignments(self):
        pass

    @abstractmethod
    def load_xyz_picks(self):
        pass

    def load_align(self, chn_depths):

        self.align = AlignData(self.xyz_picks, chn_depths, self.brain_atlas)
        self.set_init_alignment()

    def set_init_alignment(self):
        self.align.set_init_feature_track(self.feature_prev, self.track_prev)

    def load_previous_alignments(self):

        data = self.load_alignments()
        if data:
            self.alignments = data

        return self.get_previous_alignments()

    def get_previous_alignments(self):

        self.alignment_keys = [*self.alignments.keys()]
        self.alignment_keys = sorted(self.alignment_keys, reverse=True)
        self.alignment_keys.append('original')

        return self.alignment_keys

    def get_starting_alignment(self, idx):
        start_lims = 6000 / 1e6
        if self.alignment_keys[idx] == 'original':
            self.feature_prev = np.array([-1 * start_lims, start_lims])
            self.track_prev = np.array([-1 * start_lims, start_lims])
        else:
            self.feature_prev = np.array(self.alignments[self.alignment_keys[idx]][0])
            self.track_prev = np.array(self.alignments[self.alignment_keys[idx]][1])

    def add_extra_alignments(self, extra_alignments):

        extra_align = Bunch()
        for key, val in extra_alignments.items():
            if len(key) == 19 and self.user:
                extra_align[key + '_' + self.user] = val
            else:
                extra_align[key] = val

        if self.alignments:
            self.alignments.update(extra_align)
        else:
            self.alignments = extra_align

        return self.get_previous_alignments()


class AlignmentLoaderONE(AlignmentLoader):
    def __init__(self, insertion, one, brain_atlas, user=None):
        self.insertion = insertion
        self.one = one
        self.traj_id = None

        super().__init__(brain_atlas, user=user)

    def load_xyz_picks(self):
        xyz_picks = self.insertion['json'].get('xyz_picks', None)
        return np.array(xyz_picks) / 1e6 if xyz_picks is not None else None

    def load_alignments(self):
        traj = self.one.alyx.rest('trajectories', 'list', probe_insertion=self.insertion['id'],
                                  provenance='Ephys aligned histology track', no_cache=True)
        if traj:
            return traj[0]['json']

    def load_trajectory(self):
        hist = self.one.alyx.rest('trajectories', 'list', probe_insertion=self.insertion['id'],
                                  provenance='Histology track')

        if hist and hist[0]['x'] is not None:
            self.traj_id = hist[0]['id']


class AlignmentLoaderLocal(AlignmentLoader):
    def __init__(self, data_path, shank_idx, n_shanks, brain_atlas, user=None, xyz_picks=None):
        self.data_path = data_path
        self.shank_idx = shank_idx
        self.n_shanks = n_shanks

        super().__init__(brain_atlas, user, xyz_picks=xyz_picks)

    def load_xyz_picks(self):
        xyz_file_name = '*xyz_picks.json' if self.n_shanks == 1 else \
            f'*xyz_picks_shank{self.shank_idx + 1}.json'
        xyz_file = sorted(self.data_path.glob(xyz_file_name))

        #assert (len(xyz_file) == 1)
        if len(xyz_file) == 0:
            return

        with open(xyz_file[0], "r") as f:
            user_picks = json.load(f)

        return np.array(user_picks['xyz_picks']) / 1e6

    def load_alignments(self):

        # If previous alignment json file exists, read in previous alignments
        prev_align_filename = 'prev_alignments.json' if self.n_shanks == 1 else \
            f'prev_alignments_shank{self.shank_idx + 1}.json'

        prev_align_file = self.data_path.joinpath(prev_align_filename)

        if prev_align_file.exists():
            with open(prev_align_file, "r") as f:
                data = json.load(f)

            return data
