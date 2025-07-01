import numpy as np
from abc import ABC, abstractmethod
import json
from atlaselectrophysiology.utils.align import AlignData
from functools import wraps
from typing import List, Type, Union
from one import params


# TODO make this cleaner
def delegate_attributes(
    target_obj_path: str,
    attr_names: List[str],
    to_cls: Type,
    is_property: bool = True
) -> None:
    """
    Dynamically delegate attributes from a nested attribute path on `self` to the class `to_cls`.

    Args:
        target_obj_path (str): Dot-separated path to the target object (e.g. "align.ephysalign").
        attr_names (list[str]): List of attribute names to delegate.
        to_cls (type): Class to which to attach delegated properties/methods.
        is_property (bool): If True, delegate as properties; if False, delegate as methods.

    Usage:
        delegate_attributes('align', ['idx', 'track'], AlignmentLoader, is_property=True)
        delegate_attributes('align.ephysalign', ['get_channel_locations'], AlignmentLoader, is_property=False)
    """
    # Cache the path parts once
    path_parts = target_obj_path.split('.')

    def get_target_obj(self):
        obj = self
        try:
            for part in path_parts:
                obj = getattr(obj, part)
        except AttributeError as e:
            raise AttributeError(
                f"Failed to delegate attribute; could not find '{part}' in path '{target_obj_path}'"
            ) from e
        return obj

    def make_property(attr):
        def getter(self):
            obj = get_target_obj(self)
            return getattr(obj, attr)
        return property(getter)

    def make_method(attr):
        @wraps(getattr(to_cls, attr, lambda *a, **k: None))  # Preserve docstring if available
        def method(self, *args, **kwargs):
            obj = get_target_obj(self)
            func = getattr(obj, attr)
            return func(*args, **kwargs)
        return method

    for attr_name in attr_names:
        if is_property:
            setattr(to_cls, attr_name, make_property(attr_name))
        else:
            setattr(to_cls, attr_name, make_method(attr_name))


class AlignmentLoader(ABC):
    def __init__(self, brain_atlas, user):
        self.brain_atlas = brain_atlas
        self.alignments = dict()
        self.alignment_keys = ['original']
        self.xyz_picks = self.load_xyz_picks()
        self._delegate_alignment_attributes()
        self.extra_alignments = dict()
        self.user = user

    def _delegate_alignment_attributes(self) -> None:
        """
        Delegate selected attributes and methods from nested objects to this class.
        Called once after initialization.
        """
        delegate_attributes(
            'align',
            [
                'idx', 'idx_prev', 'track', 'feature', 'xyz_channels',
                'track_lines', 'xyz_track', 'xyz_samples'
            ],
            AlignmentLoader,
            is_property=True
        )

        delegate_attributes(
            'align.ephysalign',
            ['xyz_track', 'xyz_samples'],
            AlignmentLoader,
            is_property=True
        )

        delegate_attributes(
            'align.buffer',
            ['current_idx', 'total_idx'],
            AlignmentLoader,
            is_property=True
        )

        delegate_attributes(
            'align.buffer',
            ['next_idx', 'prev_idx'],
            AlignmentLoader,
            is_property=False
        )

        delegate_attributes(
            'align',
            ['scale_hist_data', 'offset_hist_data', 'get_scaled_histology',
             'reset_features_and_tracks'],
            AlignmentLoader,
            is_property=False
        )



    def get_align(self, chn_depths):

        self.align = AlignData(self.xyz_picks, chn_depths, self.brain_atlas)
        self.set_init_alignment()


    def get_previous_alignments(self):

        data = self.load_alignments()

        if data:
            data.update(self.extra_alignments)
            self.alignments = data
        else:
            self.alignments = self.extra_alignments

        self.alignment_keys = [*self.alignments.keys()]
        self.alignment_keys = sorted(self.alignment_keys, reverse=True)
        self.alignment_keys.append('original')

        return self.alignment_keys

    @abstractmethod
    def load_alignments(self):
        """

        :return:
        """


    @abstractmethod
    def load_xyz_picks(self):
        """

        :return:
        """

    def get_starting_alignment(self, idx):
        start_lims = 6000 / 1e6
        if self.alignment_keys[idx] == 'original':
            self.feature_prev = np.array([-1 * start_lims, start_lims])
            self.track_prev = np.array([-1 * start_lims, start_lims])
        else:
            self.feature_prev = np.array(self.alignments[self.alignment_keys[idx]][0])
            self.track_prev = np.array(self.alignments[self.alignment_keys[idx]][1])


    def set_init_alignment(self):
        self.align.set_init_feature_track(self.feature_prev, self.track_prev)

    def add_extra_alignments(self, extra_alignments):

        for key, val in extra_alignments.items():
            if len(key) == 19 and self.user:
                self.extra_alignments[key + '_' + self.user] = val
            else:
                self.extra_alignments[key] = val



class AlignmentLoaderONE(AlignmentLoader):
    def __init__(self, insertion, one, brain_atlas, user=None):
        self.insertion = insertion
        self.one = one
        self.traj_id = None
        super().__init__(brain_atlas, user=user)

    def load_xyz_picks(self):
        return np.array(self.insertion['json']['xyz_picks']) / 1e6

    def load_alignments(self):
        ephys_traj_prev = self.one.alyx.rest('trajectories', 'list', probe_insertion=self.insertion['id'],
                                             provenance='Ephys aligned histology track', no_cache=True)
        if ephys_traj_prev:
            return ephys_traj_prev[0]['json'] or {}

    def load_trajectory(self):
        hist_traj = self.one.alyx.rest('trajectories', 'list', probe_insertion=self.insertion['id'],
                                       provenance='Histology track')
        # TODO clean this up
        if len(hist_traj) == 1:
            if hist_traj[0]['x'] is not None:
                self.traj_id = hist_traj[0]['id']



class AlignmentLoaderLocal(AlignmentLoader):
    def __init__(self, data_path, shank_idx, n_shanks, brain_atlas, user=None):
        self.data_path = data_path
        self.shank_idx = shank_idx
        self.n_shanks = n_shanks
        super().__init__(brain_atlas, user=user)

    def load_xyz_picks(self):
        xyz_file_name = '*xyz_picks.json' if self.n_shanks == 1 else \
            f'*xyz_picks_shank{self.shank_idx + 1}.json'
        xyz_file = sorted(self.data_path.glob(xyz_file_name))

        assert (len(xyz_file) == 1)
        with open(xyz_file[0], "r") as f:
            user_picks = json.load(f)

        return np.array(user_picks['xyz_picks']) / 1e6

    def load_alignments(self):

        # If previous alignment json file exists, read in previous alignments
        prev_align_filename = 'prev_alignments.json' if self.n_shanks == 1 else \
            f'prev_alignments_shank{self.shank_idx + 1}.json'

        if self.data_path.joinpath(prev_align_filename).exists():
            with open(self.data_path.joinpath(prev_align_filename), "r") as f:
                # TODO see if we can use json.loads
                data = json.load(f)
            return data
