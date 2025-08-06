from abc import ABC, abstractmethod
from iblutil.util import Bunch
import numpy as np
from atlaselectrophysiology.loaders.data_loader import DataLoader, CollectionData
import one.alf.io as alfio
from one.alf.exceptions import ALFObjectNotFound

import spikeglx


class Geometry(ABC):
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

        self.raw_ind = np.arange(self.chn_x.size)

        for i in range(self.n_shanks):
            info = Bunch()
            # this is the orig idx


            # This is where the shank is in the sorted metadata
            # The lfp and rms plots are already ordered according to the sorted metadata
            # The spike sorting data channels correspond to the original channel maps

            # FYI when we have channels it is going to be different!!!!

            orig_idx = self.shank_groups[i]
            chn_x = self.chn_x[orig_idx]
            chn_y = self.chn_y[orig_idx]

            # Make sure it is sorted in depth
            chn_sort = np.argsort(chn_y)

            info['spikes_ind'] = self.chn_ind[orig_idx[chn_sort]]

            # This assumes that the ibl qc files have been sorted according to shank number and saved in this way
            info['raw_ind'] = orig_idx[chn_sort]


            # This is once the data is split and used in plot_loader - I think we can remove this
            info['chn_ind'] = np.arange(info['raw_ind'].size)

            info['chn_coords'] = np.c_[chn_x[chn_sort], chn_y[chn_sort]]
            info['chn_min'] = np.nanmin(chn_y)
            info['chn_max'] = np.nanmax(chn_y)
            info['chn_diff'] = np.min(np.abs(np.diff(np.unique(chn_y))))
            info['chn_full'] = np.arange(info['chn_min'], info['chn_max'] + info['chn_diff'], info['chn_diff'])
            info['idx_full'] = np.where(np.isin(info['chn_full'], info['chn_coords'][:, 1]))[0]

            self.shanks[i] = info

    def get_chns_for_shank(self, shank_idx):
        return self.shanks[shank_idx]


class ChannelGeometry(Geometry):
    def __init__(self, channels, shank_diff=100):
        chn_x = channels['localCoordinates'][:, 0]
        chn_y = channels['localCoordinates'][:, 1]
        chn_ind = np.arange(chn_x.size)

        self.shank_diff = shank_diff
        super().__init__(chn_x, chn_y, chn_ind)

    def _get_nshanks(self):

        x_coords = np.unique(self.chn_x)
        x_coords_diff = np.diff(x_coords)
        n_shanks = np.sum(x_coords_diff > self.shank_diff) + 1
        return n_shanks

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


class MetaGeometry(Geometry):
    def __init__(self, meta):
        self.meta = meta
        chn_x = self.meta['x']
        chn_y = self.meta['y']
        # TODO for openephys extraction we need to look into what the channel ind should be
        chn_ind = self.meta['ind']

        super().__init__(chn_x, chn_y, chn_ind)

    def _get_nshanks(self):
        n_shanks = np.unique(self.meta['shank']).size

        return n_shanks

    def _get_shank_groups(self):
        groups = Bunch()
        shanks = np.unique(self.meta['shank'])

        # TODO should this be like this or should we keep the individual shank numbering?
        for i, sh in enumerate(shanks):
            groups[i] = np.where(self.meta['shank'] == sh)[0]

        return groups

class GeometryLoader(ABC):
    def __init__(self):
        self.geometry = None


    def get_geometry(self):
        meta = self.load_metadata()
        if meta is not None:
            print('using meta')
            geometry = MetaGeometry(meta)
        else:
            chns = self.load_channels()
            # TODO if chns is also None we need to raise and error
            geometry = ChannelGeometry(chns)

        self.geometry = geometry
        self.geometry.split_chns_per_shank()

    @abstractmethod
    def load_metadata(self):
        pass

    @abstractmethod
    def load_channels(self, **kwargs):
        pass


class GeometryLoaderONE(GeometryLoader):
    def __init__(self, insertion, one, session_path=None):
        self.one = one
        self.eid = insertion['session']
        self.session_path = session_path or one.eid2path(self.eid)
        self.probe_label = insertion['name']

        super().__init__()

    def load_metadata(self):
        try:
            meta_file = self.one.load_dataset(
                self.eid, '*.ap.meta', collection=f'raw_ephys_data/{self.probe_label}', download_only=True)
            return spikeglx.read_geometry(meta_file)
        except ALFObjectNotFound:
            return None

    def load_channels(self, **kwargs):
        # For now see if we can always just use the metadata otherwise we will need to find the spikesorting
        # collection and then get the channels via one.load_object
        return None


class GeometryLoaderLocal(GeometryLoader):
    def __init__(self, probe_path, collections: CollectionData):
        self.probe_path = probe_path
        self.spike_path = probe_path.joinpath(collections.spike_collection)
        self.meta_path = probe_path.joinpath(collections.meta_collection)

        super().__init__()

    def load_metadata(self):
        meta_file = next(self.meta_path.glob('*.ap.*meta'), None)
        print(meta_file)
        if meta_file is None:
            return
        return spikeglx.read_geometry(meta_file)

    def load_channels(self, **kwargs):
        chns = DataLoader.load_data(alfio.load_object, self.spike_path, 'channels',
                              attribute=['localCoordinates'], **kwargs)
        return chns if chns['exists'] else None




