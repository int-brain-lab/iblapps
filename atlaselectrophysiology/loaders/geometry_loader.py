from abc import ABC, abstractmethod
from iblutil.util import Bunch
import numpy as np


class GeometryLoader(ABC):
    def __init__(self, chn_x, chn_y, chn_ind):
        self.chn_x = chn_x
        self.chn_y = chn_y
        self.chn_ind = chn_ind

    @abstractmethod
    def _get_nshanks(self):
        pass

    @abstractmethod
    def _get_shank_groups(self):
        pass

    def split_chns_per_shank(self):
        self.n_shanks = self._get_nshanks()
        self.shank_groups = self._get_shank_groups()

        self.shanks = Bunch()

        for i in range(self.n_shanks):
            info = Bunch()
            # this is the orig idx
            orig_idx = self.shank_groups[i]
            chn_x = self.chn_x[orig_idx]
            chn_y = self.chn_y[orig_idx]
            chn_sort = np.argsort(chn_y)

            info['orig_ind'] = self.chn_ind[orig_idx[chn_sort]]
            info['chn_ind'] = np.arange(info['orig_chn_ind'].size)
            info['chn_coords'] = np.c_[chn_x[chn_sort], chn_y[chn_sort]]

            info['chn_min'] = np.nanmin(info['chn_y'])
            info['chn_max'] = np.nanmax(info['chn_y'])
            info['chn_diff'] = np.min(np.abs(np.diff(np.unique(info['chn_y']))))
            info['chn_full'] = np.arange(info['chn_min'], info['chn_max'] + info['chn_diff'], info['chn_diff'])
            info['idx_full'] = np.where(np.isin(info['chn_full'], info['chn_coords'][:, 1]))[0]

            self.shanks[i] = info

    def get_chns_for_shank(self, shank_idx):
        return self.shanks[shank_idx]


class ChannelGeometryLoader(GeometryLoader):
    def __init__(self, channels, shank_diff=100):
        chn_x = channels['localCoordinates'][:, 0]
        chn_y = channels['localCoordinates'][:, 1]
        chn_ind = channels['rawInd']

        self.shank_diff = shank_diff
        super().__init__(chn_x, chn_y, chn_ind)

    def _get_nshanks(self):

        x_coords = np.unique(self.chn_x)
        x_coords_diff = np.diff(x_coords)
        self.n_shanks = np.sum(x_coords_diff > self.shank_diff) + 1

    def _get_shank_groups(self):

        x_coords = np.unique(self.chn_x)
        shank_groups = np.split(x_coords, np.where(np.diff(x_coords) > self.shank_diff)[0] + 1)

        assert len(shank_groups) == self.n_shanks

        groups = Bunch()
        for i, grp in enumerate(shank_groups):
            if len(grp) == 1:
                grp = np.array([grp[0], grp[0]])
            grp_shank = [grp[0], grp[1]]

            shank_chns = np.bitwise_and(self.chn_x >= grp_shank[0], self.chn_x <= grp_shank[1])
            # this is the orig idex
            groups[i] = np.where(shank_chns)[0]

        return groups


class MetaGeometryLoader(GeometryLoader):
    def __init__(self, meta):
        self.meta = meta
        chn_x = self.meta['x']
        chn_y = self.meta['y']
        # TODO for openephys extraction we need to look into what the channel ind should be
        # chn_ind = self.meta['ind']
        chn_ind = np.arange(chn_x.size)

        super().__init__(chn_x, chn_y, chn_ind)

    def _get_nshanks(self):
        n_shanks = np.unique(self.meta['shank']).size

        return n_shanks

    def _get_shank_groups(self):
        groups = Bunch()
        for i in range(self.n_shanks):
            groups[i] = np.where(self.meta['shank'] == i)[0]

        return groups


