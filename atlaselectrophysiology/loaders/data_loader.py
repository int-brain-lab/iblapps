from typing import Optional
from dataclasses import dataclass
from iblutil.util import Bunch
from iblutil.numerical import ismember
import numpy as np
import logging
from one.alf.exceptions import ALFObjectNotFound
import one.alf.io as alfio
import traceback
from abc import ABC, abstractmethod
# TODO check which logger
logger = logging.getLogger('ibllib')

# TODO dataloader directly from spike interface?
# TODO raw data loader
# TODO class that can read in the metafiles and determine the number of shanks based on this read in the appropriate
# data for the shank in question (needs to either read from the meta file or detect from the chns.localCoordinates)
# TODO for offline, need to prepare the data per shank in the same way: Three options
# 1. spikesorting joined, raw data joined
# 2. spikesorting split, raw data joined
# 3. spikesorting split, raw data split


@dataclass
class CollectionData:
    spike_collection: Optional[str] = ''
    ephys_collection: Optional[str] = ''
    task_collection: Optional[str] = ''
    raw_task_collection: Optional[str] = ''


class DataLoader(ABC):
    def __init__(self):
        self.filter = False

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

    @abstractmethod
    def load_passive_data(self, alf_object, **kwargs): pass

    @abstractmethod
    def load_raw_passive_data(self, alf_object, **kwargs): pass

    def get_passive_data(self):
        """
        Load passive stimulus data (RF map, visual stim, gabor stimuli).
        """
        # TO DO need to add in collections also improve hadnling of this will all these expections
        # Load in RFMAP data
        try:
            rf_data = self.load_passive_data('passiveRFM')
            frame_path = self.load_raw_passive_data('_iblrig_RFMapStim.raw.bin')
            frames = np.fromfile(frame_path, dtype="uint8")
            rf_data['frames'] = np.transpose(np.reshape(frames, [15, 15, -1], order="F"), [2, 1, 0])
        except Exception:
            logger.warning('rfmap data was not found, some plots will not display')
            rf_data = {}
            rf_data['exists'] = False

        # Load in passive stim data
        stim_data = self.load_passive_data('passiveStims')

        # Load in passive gabor data
        try:
            gabor = self.load_passive_data('passiveGabor')
            vis_stim = {}
            vis_stim['leftGabor'] = gabor['start'][(gabor['position'] == 35) & (gabor['contrast'] > 0.1)]
            vis_stim['rightGabor'] = gabor['start'][(gabor['position'] == -35) & (gabor['contrast'] > 0.1)]
            vis_stim['exists'] = True
        except Exception:
            logger.warning('passive gabor data was not found, some plots will not display')
            vis_stim = {}
            vis_stim['exists'] = False

        return rf_data, stim_data, vis_stim

    @abstractmethod
    def load_ephys_data(self, alf_object, **kwargs): pass


    def get_rms_data(self, band='AP'):
        """
        Load RMS data for AP or LF band.
        """

        rms_data = self.load_ephys_data(f'ephysTimeRms{band}')

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

        psd_data = self.load_ephys_data(f'ephysSpectralDensity{band}')

        if psd_data['exists']:
            if 'amps' in psd_data.keys():
                psd_data['power'] = psd_data.pop('amps')

        return psd_data

    @abstractmethod
    def load_spikes_data(self, alf_object, attributes, **kwargs): pass


    def get_spikesorting_data(self):
        """
        Load spike sorting data (spikes, clusters, channels) and filter by min_fr
        """

        spikes = self.load_spikes_data('spikes', ['depths', 'amps', 'times', 'clusters'])

        clusters = self.load_spikes_data('clusters', ['metrics', 'peakToTrough', 'waveforms', 'channels'])

        channels = self.load_spikes_data('channels', ['rawInd', 'localCoordinates'])

        if self.filter:
            # Remove low firing rate clusters
            spikes, clusters = self.filter_spikes_and_clusters(spikes, clusters)

        return spikes, clusters, channels

    @staticmethod
    def filter_spikes_and_clusters(spikes, clusters, min_fr=50 / 3600):
        """
        Remove low-firing clusters and filter spikes accordingly.
        """

        clu_idx = clusters.metrics.firing_rate > min_fr
        exists = clusters.pop('exists')
        clusters = alfio.AlfBunch({k: v[clu_idx] for k, v in clusters.items()})
        clusters['exists'] = exists

        spike_idx, ib = ismember(spikes.clusters, clusters.metrics.index)
        clusters.metrics.reset_index(drop=True, inplace=True)
        exists = spikes.pop('exists')
        spikes = alfio.AlfBunch({k: v[spike_idx] for k, v in spikes.items()})
        spikes['exists'] = exists
        spikes.clusters = clusters.metrics.index[ib].astype(np.int32)

        return spikes, clusters


class DataLoaderONE(DataLoader):
    def __init__(self, insertion, one, spike_collection=None):
        self.one = one
        self.eid = insertion['session']
        self.session_path = one.eid2path(self.eid)
        self.probe_label = insertion['name']
        self.spike_collection = spike_collection
        self.probe_path = self.get_spikesorting_path()
        self.probe_collection = str(self.probe_path.relative_to(self.session_path))

        super().__init__()
        self.filter = True

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

    def load_passive_data(self, alf_object, **kwargs):
        return self.load_data(self.one.load_object, self.eid, alf_object)

    def load_raw_passive_data(self, alf_object, **kwargs):
        return self.load_data(self.one.load_object, self.eid, alf_object)

    def load_ephys_data(self, alf_object, **kwargs):
        return self.load_data(self.one.load_object, self.eid, alf_object, f'raw_ephys_data/{self.probe_label}', **kwargs)

    def load_spikes_data(self, alf_object, attributes, **kwargs):
        return self.load_data(self.one.load_object, self.eid, alf_object, collection=self.probe_collection, attribute=attributes, **kwargs)


class DataLoaderLocal(DataLoader):
    def __init__(self, probe_path, collections: CollectionData):
        """
        Load and process data for a specific session/probe.
        """

        self.probe_path = probe_path
        self.spike_path = probe_path.joinpath(collections.spike_collection)
        self.ephys_path = probe_path.joinpath(collections.ephys_collection)
        self.task_path = probe_path.joinpath(collections.task_collection)
        self.raw_task_path = probe_path.joinpath(collections.raw_task_collection)
        self.probe_collection = collections.spike_collection
        super().__init__()

    def load_passive_data(self, alf_object, **kwargs):
        return self.load_data(alfio.load_object, self.task_path, alf_object, **kwargs)

    def load_raw_passive_data(self, alf_object, **kwargs):
        return self.load_data(alfio.load_object, self.raw_task_path, alf_object)

    def load_ephys_data(self, alf_object, **kwargs):
        return self.load_data(alfio.load_object, self.ephys_path, alf_object, **kwargs)

    def load_spikes_data(self, alf_object, attributes, **kwargs):
        return self.load_data(alfio.load_object, self.spike_path, alf_object, attribute=attributes, **kwargs)





