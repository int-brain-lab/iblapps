from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
import logging
import numpy as np
from pathlib import Path
import traceback
from typing import Optional

from brainbox.io.spikeglx import Streamer
import ibldsp.voltage
from iblutil.numerical import ismember
from iblutil.util import Bunch
from one.alf.exceptions import ALFObjectNotFound
import one.alf.io as alfio
import spikeglx

logger = logging.getLogger(__name__)

# TODO docstrings and typing and logging
# TODO spikeinterface dataloader
# TODO for offline loader need to deal with 3 different types of data
# 1. spikesorting joined, raw data joined
# 2. spikesorting split, raw data joined
# 3. spikesorting split, raw data split


@dataclass
class CollectionData:
    spike_collection: Optional[str] = ''
    ephys_collection: Optional[str] = ''
    task_collection: Optional[str] = ''
    raw_task_collection: Optional[str] = ''
    meta_collection: Optional[str] = ''


class DataLoader(ABC):
    def __init__(self):
        self.filter = False

    def get_data(self):
        """
        Load/ Download all data associated with probe
        """

        data = Bunch()
        # Load in spike sorting data
        data['spikes'], data['clusters'], data['channels'] = self.get_spikes_data()
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
    def load_passive_data(self, alf_object, **kwargs):
        pass

    @abstractmethod
    def load_raw_passive_data(self, alf_object, **kwargs):
        pass

    def get_passive_data(self):
        """
        Load passive stimulus data (RF map, visual stim, gabor stimuli).
        """
        try:
            rf_data = self.load_passive_data('passiveRFM')
            frame_path = self.load_raw_passive_data('RFMapStim')
            frames = np.fromfile(frame_path['raw'], dtype="uint8")
            rf_data['frames'] = np.transpose(np.reshape(frames, [15, 15, -1], order="F"), [2, 1, 0])
        except Exception:
            logger.warning('passiveRFM data was not found, some plots will not display')
            rf_data = Bunch(exists=False)

        # Load in passive stim data
        stim_data = self.load_passive_data('passiveStims')

        # Load in passive gabor data
        try:
            gabor = self.load_passive_data('passiveGabor')
            if not gabor['exists']:
                vis_stim = Bunch(exists=False)
            else:
                vis_stim = Bunch()
                vis_stim['leftGabor'] = gabor['start'][(gabor['position'] == 35) & (gabor['contrast'] > 0.1)]
                vis_stim['rightGabor'] = gabor['start'][(gabor['position'] == -35) & (gabor['contrast'] > 0.1)]
                vis_stim['exists'] = True
        except Exception:
            logger.warning('passiveGabor data was not found, some plots will not display')
            vis_stim = Bunch(exists=False)

        return rf_data, stim_data, vis_stim

    @abstractmethod
    def load_ephys_data(self, alf_object, **kwargs):
        pass

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
    def load_spikes_data(self, alf_object, attributes, **kwargs):
        pass

    def get_spikes_data(self):
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
    def __init__(self, insertion, one, session_path=None, spike_collection=None):
        self.one = one
        self.eid = insertion['session']
        self.session_path = session_path or one.eid2path(self.eid)
        self.probe_label = insertion['name']
        self.spike_collection = spike_collection
        self.probe_path = self.get_spike_sorting_path()
        self.probe_collection = str(self.probe_path.relative_to(self.session_path))
        self.filter = True

        super().__init__()

    def get_spike_sorting_path(self):
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
        return self.load_data(self.one.load_object, self.eid, alf_object,
                              collection=f'raw_ephys_data/{self.probe_label}', **kwargs)

    def load_spikes_data(self, alf_object, attributes, **kwargs):
        return self.load_data(self.one.load_object, self.eid, alf_object,
                              collection=self.probe_collection, attribute=attributes, **kwargs)


class DataLoaderLocal(DataLoader):
    def __init__(self, probe_path, collections: CollectionData):

        self.probe_path = probe_path
        self.spike_path = probe_path.joinpath(collections.spike_collection)
        self.ephys_path = probe_path.joinpath(collections.ephys_collection)
        self.task_path = probe_path.joinpath(collections.task_collection)
        self.raw_task_path = probe_path.joinpath(collections.raw_task_collection)
        self.meta_path = probe_path.joinpath(collections.meta_collection)
        self.probe_collection = collections.spike_collection

        super().__init__()

    def load_passive_data(self, alf_object, **kwargs):
        return self.load_data(alfio.load_object, self.task_path, alf_object, **kwargs)

    def load_raw_passive_data(self, alf_object, **kwargs):
        return self.load_data(alfio.load_object, self.raw_task_path, alf_object)

    def load_ephys_data(self, alf_object, **kwargs):
        return self.load_data(alfio.load_object, self.ephys_path, alf_object, **kwargs)

    def load_spikes_data(self, alf_object, attributes, **kwargs):
        return self.load_data(alfio.load_object, self.spike_path, alf_object,
                              attribute=attributes, **kwargs)


class SpikeGLXLoader(ABC):
    def __init__(self, save_path: Path = None):
        self.meta = None
        self.save_path = save_path
        self.cached_path = save_path.joinpath('alignment_gui_raw_data_snippets.npy') if save_path else None

    def get_meta_data(self) -> Bunch:
        self.meta = self.load_meta_data()
        if not self.meta:
            return Bunch({'exists': False})

        geometry = spikeglx.geometry_from_meta(self.meta, sort=True)
        return Bunch(geometry, exists=True)

    @abstractmethod
    def load_meta_data(self):
        pass

    @abstractmethod
    def load_ap_data(self):
        pass

    def load_ap_snippets(self, twin: float = 1) -> Bunch:

        if self.cached_path and self.cached_path.exists():
            return np.load(self.cached_path, allow_pickle=True).item()

        sr = self.load_ap_data()
        if not sr:
            return Bunch(exists=False)

        data = defaultdict(Bunch)
        for t in self.get_time_snippets(sr):
            data['images'][t] = self._get_snippet(sr, t, twin=twin)

        data['exists'] = True
        data['fs'] = sr.fs

        if self.cached_path:
            np.save(self.cached_path, data)

        return data

    def _get_snippet(self, sr, t: float, twin: float = 1):
        start_sample = int(t * sr.fs)
        end_sample = start_sample + int(twin * sr.fs)
        raw = sr[start_sample:end_sample, :-sr.nsync].T

        # Detect bad channels and destripe
        channel_labels, _ = ibldsp.voltage.detect_bad_channels(raw, sr.fs)
        raw = ibldsp.voltage.destripe(raw, fs=sr.fs, h=sr.geometry, channel_labels=channel_labels)

        # Extract a window in time (450–500 ms)
        window = slice(int(0.450 * sr.fs), int(0.500 * sr.fs))
        return raw[:, window].T

    @staticmethod
    def get_time_snippets(sr, n: int = 3, pad: int = 200) -> np.ndarray:
        file_duration = sr.meta['fileTimeSecs']
        pad = pad if file_duration > 500 else 0
        usable_time = file_duration - 2 * pad
        intervals = usable_time // n
        return intervals * (np.arange(n) + 1)


class SpikeGLXLoaderONE(SpikeGLXLoader):
    def __init__(self, insertion, one, session_path=None, force=False):
        self.one = one
        self.eid = insertion['session']
        self.session_path = session_path or self.one.eid2path(self.eid)
        self.pid = insertion['id']
        self.probe_label = insertion['name']
        self.force = force
        save_path = self.session_path.joinpath(f'raw_ephys_data/{self.probe_label}')

        super().__init__(save_path)

    def load_meta_data(self):
        try:
            meta_file = self.one.load_dataset(
                self.eid, '*.ap.meta', collection=f'raw_ephys_data/{self.probe_label}', download_only=True)
            return spikeglx.read_meta_data(meta_file)
        except ALFObjectNotFound:
            return None

    def load_ap_data(self):
        return Streamer(pid=self.pid, one=self.one, remove_cached=self.force, typ='ap')


class SpikeGLXLoaderLocal(SpikeGLXLoader):
    def __init__(self, probe_path, meta_collection):
        self.meta_path = probe_path.joinpath(meta_collection)

        super().__init__(self.meta_path)

    def load_meta_data(self):
        meta_file = next(self.meta_path.glob('*.ap.*meta'), None)
        return spikeglx.read_meta_data(meta_file) if meta_file else None

    def load_ap_data(self):
        ap_file = next(self.meta_path.glob('*.ap.*bin'), None)
        return spikeglx.Reader(ap_file) if ap_file else None
