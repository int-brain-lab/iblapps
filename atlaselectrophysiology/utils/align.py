import numpy as np
from ibllib.pipes.ephys_alignment import EphysAlignment

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

    def prev_idx(self) -> bool:
        """
        Move backward through previously visited indices.
        """
        if self.total_idx > self.last_idx:
            self.last_idx = self.total_idx

        if self.current_idx > np.max([0, self.total_idx - self.diff_idx]):
            self.current_idx -= 1
            self.idx = np.mod(self.current_idx, self.max_idx)

            return True

    def next_idx(self) -> bool:
        """
        Move forward through the buffer if within bounds.
        """
        if (self.current_idx < self.total_idx) & (self.current_idx > self.total_idx - self.max_idx):
            self.current_idx += 1
            self.idx = np.mod(self.current_idx, self.max_idx)

            return True

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

    @property
    def idx(self):
        return self.buffer.idx

    @property
    def idx_prev(self):
        return self.buffer.idx_prev

    @property
    def xyz_track(self):
        return self.ephysalign.xyz_track

    @property
    def xyz_samples(self):
        return self.ephysalign.xyz_samples

    @property
    def xyz_channels(self):
        return self.ephysalign.get_channel_locations(self.features[self.idx], self.tracks[self.idx])

    @property
    def track_lines(self):
        return self.ephysalign.get_perp_vector(self.features[self.idx], self.tracks[self.idx])

    @property
    def track(self):
        return self.tracks[self.idx]

    @property
    def feature(self):
        return self.features[self.idx]

    @property
    def current_idx(self):
        return self.buffer.current_idx

    @property
    def total_idx(self):
        return self.buffer.total_idx

    def next_idx(self):
        return self.buffer.next_idx()

    def prev_idx(self):
        return self.buffer.prev_idx()


    def set_init_feature_track(self, feature=None, track=None):
        if feature is not None:
            self.ephysalign.feature_init = feature
        if track is not None:
            self.ephysalign.track_init = track
        self.features[self.idx], self.tracks[self.idx], _ = self.ephysalign.get_track_and_feature()

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

        self.features[self.idx] = np.sort(np.r_[self.features[self.idx_prev][[0, -1]], line_feature])


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

