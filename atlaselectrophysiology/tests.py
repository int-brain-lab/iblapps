import unittest
import numpy as np
import ibllib.atlas as atlas
from atlaselectrophysiology.improve_load_data import (EphysAlignmentFromLocal, TIP_SIZE_UM, _cumulative_distance)
ba = atlas.AllenAtlas(25)

xyz_picks = np.array([[-0.002713, -0.004124, -0.000543],
                      [-0.002688, -0.004124, -0.000842],
                      [-0.002663, -0.004124, -0.001167],
                      [-0.002638, -0.004124, -0.001392],
                      [-0.002613, -0.0041  , -0.001643],
                      [-0.002563, -0.004124, -0.001968],
                      [-0.002513, -0.00415 , -0.002392],
                      [-0.002463, -0.00415 , -0.002768],
                      [-0.002113, -0.004575, -0.004817],
                      [-0.002038, -0.00465 , -0.005193],
                      [-0.002038, -0.004674, -0.005318]])


depths = np.arange(20, 3840 + 20, 20) / 1e6
feature_prev = np.load('feature_prev.npy')
track_prev = np.load('track_prev.npy')
xyz_channels_ref = np.load('xyz_extreme_prev.npy')
brain_regions_unique = np.load('brain_regions.npy', allow_pickle=True)


class TestsEphysAlignment(unittest.TestCase):

    def setUp(self) -> None:
        self.ephysalign = EphysAlignmentFromLocal(xyz_picks)
        self.feature = self.ephysalign.feature_init
        self.track = self.ephysalign.track_init

    def test_no_scaling(self):
        xyz_channels = self.ephysalign.get_channel_locations(self.feature, self.track, depths=depths)
        coords = np.r_[[xyz_picks[-1, :]], [xyz_channels[0, :]]]
        dist_to_fist_electrode = np.around(_cumulative_distance(coords)[-1], 5)
        assert np.isclose(dist_to_fist_electrode, (TIP_SIZE_UM + 20) / 1e6)

    def test_offset(self):
        feature_val = 500 / 1e6
        track_val = 1000 / 1e6

        tracks = np.sort(np.r_[self.track[[0, -1]], track_val])
        track_new = self.ephysalign.feature2track(tracks, self.feature, self.track)
        feature_new = np.sort(np.r_[self.feature[[0,-1]], feature_val])
        track_new = self.ephysalign.adjust_extremes_uniform(feature_new, track_new)

        xyz_channels = self.ephysalign.get_channel_locations(feature_new, track_new, depths=depths)
        coords = np.r_[[xyz_picks[-1, :]], [xyz_channels[0, :]]]
        dist_to_fist_electrode = np.around(_cumulative_distance(coords)[-1], 5)
        assert np.isclose(dist_to_fist_electrode, (TIP_SIZE_UM + 500 + 20) / 1e6)
        track_val = self.ephysalign.track2feature(track_val, feature_new, track_new)
        self.assertTrue(np.all(np.isclose(track_val, feature_val)))

        region_new, _ = self.ephysalign.scale_histology_regions(feature_new, track_new)
        _, scale_factor = self.ephysalign.get_scale_factor(region_new)
        self.assertTrue(np.all(np.isclose(scale_factor, 1)))

    def test_uniform_scaling(self):
        feature_val = np.array([500, 700, 2000]) / 1e6
        track_val = np.array([1000, 1300, 2700]) / 1e6

        tracks = np.sort(np.r_[self.track[[0, -1]], track_val])
        track_new = self.ephysalign.feature2track(tracks, self.feature, self.track)
        feature_new = np.sort(np.r_[self.feature[[0,-1]], feature_val])
        track_new = self.ephysalign.adjust_extremes_uniform(feature_new, track_new)

        region_new, _ = self.ephysalign.scale_histology_regions(feature_new, track_new)
        _, scale_factor = self.ephysalign.get_scale_factor(region_new)
        self.assertTrue(np.isclose(scale_factor[0], 1))
        self.assertTrue(np.isclose(scale_factor[-1], 1))

    def test_linear_scaling(self):
        feature_val = np.array([500, 700, 2000]) / 1e6
        track_val = np.array([1000, 1300, 2700]) / 1e6

        tracks = np.sort(np.r_[self.track[[0, -1]], track_val])
        track_new = self.ephysalign.feature2track(tracks, self.feature, self.track)
        feature_new = np.sort(np.r_[self.feature[[0,-1]], feature_val])

        fit = np.polyfit(feature_new[1:-1], track_new[1:-1], 1)
        linear_fit = np.around(1/fit[0],3)

        feature_new, track_new = self.ephysalign.adjust_extremes_linear(feature_new,
                                                                        track_new, extend_feature=1)#5000 / 1e6)

        region_new, _ = self.ephysalign.scale_histology_regions(feature_new, track_new)
        _, scale_factor = self.ephysalign.get_scale_factor(region_new)

        self.assertTrue(np.isclose(np.around(scale_factor[0], 3), linear_fit))
        self.assertTrue(np.isclose(np.around(scale_factor[-1], 3), linear_fit))


class TestsEphysReconstruction(unittest.TestCase):

        def setUp(self) -> None:
            self.ephysalign = EphysAlignmentFromLocal(xyz_picks, use_previous=True,
                                                    track_previous=track_prev, feature_previous=feature_prev)
            self.feature = self.ephysalign.feature_init
            self.track = self.ephysalign.track_init

        def test_channel_locations(self):
            xyz_channels = self.ephysalign.get_channel_locations(self.feature, self.track, depths=depths)
            self.assertTrue(np.all(np.isclose(xyz_channels[0,:], xyz_channels_ref[0])))
            self.assertTrue(np.all(np.isclose(xyz_channels[-1, :], xyz_channels_ref[-1])))

            brain_regions = self.ephysalign.get_brain_locations(xyz_channels)
            self.assertTrue(np.all(np.equal(np.unique(brain_regions.acronym), brain_regions_unique)))









if __name__ == "__main__":
    unittest.main(exit=False)


