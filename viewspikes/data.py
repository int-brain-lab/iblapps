from pathlib import Path
import shutil

from ibllib.io import spikeglx
from one.webclient import dataset_record_to_url

import numpy as np
import scipy.signal

import one.alf.io


CHUNK_DURATION_SECS = 1
OUTPUT_TO_TEST = True


def get_ks2_batch(ks2memmap, ibatch):
    BATCH_SIZE = 65600
    NTR = 384
    offset = BATCH_SIZE * NTR * ibatch
    from_to = np.array([0, BATCH_SIZE * NTR])
    slic = slice(from_to[0] + offset, from_to[1] + offset)

    ks2 = np.reshape(ks2memmap[slice(from_to[0] + offset, from_to[1] + offset)], (NTR, BATCH_SIZE))
    return ks2


# ks2 proc
def get_ks2(raw, dsets, one):
    kwm = next(dset for dset in dsets if dset['dataset_type'] == 'kilosort.whitening_matrix')
    kwm = np.load(one._download_dataset(kwm))
    channels = [dset for dset in dsets if dset['dataset_type'].startswith('channels')]
    malf_path = next(iter(one._download_datasets(channels))).parent
    channels = one.alf.io.load_object(malf_path, 'channels')
    _car = raw[channels['rawInd'], :] - np.mean(raw[channels.rawInd, :], axis=0)
    sos = scipy.signal.butter(3, 300 / 30000 / 2, btype='highpass', output='sos')
    ks2 = np.zeros_like(raw)
    ks2[channels['rawInd'], :] = scipy.signal.sosfiltfilt(sos, _car)
    std_carbutt = np.std(ks2)
    ks2[channels['rawInd'], :] = np.matmul(kwm, ks2[channels['rawInd'], :])
    ks2 = ks2 * std_carbutt / np.std(ks2)
    return ks2


def get_spikes(dsets, one):
    dtypes_spikes = ['spikes.clusters', 'spikes.amps', 'spikes.times', 'clusters.channels',
                     'spikes.samples', 'spikes.depths']
    dsets_spikes = [dset for dset in dsets if dset['dataset_type'] in dtypes_spikes]
    malf_path = next(iter(one._download_datasets(dsets_spikes))).parent
    channels = one.alf.io.load_object(malf_path, 'channels')
    clusters = one.alf.io.load_object(malf_path, 'clusters')
    spikes = one.alf.io.load_object(malf_path, 'spikes')
    return spikes, clusters, channels


def stream(pid, t, one=None, cache=True, dsets=None, typ='ap', tlen=1):
    """
    NB: returned Reader object must be closed after use
    :param pid: Probe UUID
    :param t:
    :param one: An instance of ONE
    :param cache:
    :param dsets:
    :param typ: 'ap' or 'lf'
    :param tlen: no. of seconds to stream
    :return: sr, dsets, t0
    """

    assert one
    assert typ in ['lf', 'ap']
    t0 = np.floor(t / CHUNK_DURATION_SECS) * CHUNK_DURATION_SECS
    if cache:
        samples_folder = Path(one.alyx._par.CACHE_DIR).joinpath('cache', typ)
    sample_file_name = Path(f"{pid}_{str(int(t0)).zfill(5)}.meta")
    if dsets is None:
        dsets = one.alyx.rest('datasets', 'list', probe_insertion=pid)
    if cache and samples_folder.joinpath(sample_file_name).exists():
        print(f'loading {sample_file_name} from cache')
        sr = spikeglx.Reader(samples_folder.joinpath(sample_file_name).with_suffix('.bin'),
                             open=True)
        return sr, dsets, t0

    dset_ch = next(dset for dset in dsets if dset['dataset_type'] == "ephysData.raw.ch" and
                   f'.{typ}.' in dset['name'])
    dset_meta = next(dset for dset in dsets if dset['dataset_type'] == "ephysData.raw.meta" and
                     f'.{typ}.' in dset['name'])
    dset_cbin = next(dset for dset in dsets if dset['dataset_type'] == f"ephysData.raw.{typ}" and
                     f'.{typ}.' in dset['name'])

    file_ch, file_meta = one._download_datasets([dset_ch, dset_meta])

    first_chunk = int(t0 / CHUNK_DURATION_SECS)
    last_chunk = int((t0 + tlen) / CHUNK_DURATION_SECS) - 1

    sr = spikeglx.download_raw_partial(
        one=one,
        url_cbin=dataset_record_to_url(dset_cbin)[0],
        url_ch=file_ch,
        first_chunk=first_chunk,
        last_chunk=last_chunk)

    if cache:
        samples_folder.mkdir(exist_ok=True, parents=True)
        out_meta = samples_folder.joinpath(sample_file_name)
        shutil.copy(sr.file_meta_data, out_meta)
        with open(out_meta.with_suffix('.bin'), 'wb') as fp:
            sr.open()
            sr._raw[:].tofile(fp)

    return sr, dsets, t0
