from ibllib.io import spikeglx
import numpy as np
import ibllib.dsp as dsp
from scipy import signal
from ibllib.misc import print_progress
from pathlib import Path
import one.alf.io as alfio
import logging
import ibllib.ephys.ephysqc as ephysqc
from phylib.io import alf

_logger = logging.getLogger('ibllib')


RMS_WIN_LENGTH_SECS = 3
WELCH_WIN_LENGTH_SAMPLES = 1024


def rmsmap(fbin, spectra=True):
    """
    Computes RMS map in time domain and spectra for each channel of Neuropixel probe

    :param fbin: binary file in spike glx format (will look for attached metatdata)
    :type fbin: str or pathlib.Path
    :param spectra: whether to compute the power spectrum (only need for lfp data)
    :type: bool
    :return: a dictionary with amplitudes in channeltime space, channelfrequency space, time
     and frequency scales
    """
    if not isinstance(fbin, spikeglx.Reader):
        sglx = spikeglx.Reader(fbin)
        sglx.open()
    rms_win_length_samples = 2 ** np.ceil(np.log2(sglx.fs * RMS_WIN_LENGTH_SECS))
    # the window generator will generates window indices
    wingen = dsp.WindowGenerator(ns=sglx.ns, nswin=rms_win_length_samples, overlap=0)
    # pre-allocate output dictionary of numpy arrays
    win = {'TRMS': np.zeros((wingen.nwin, sglx.nc)),
           'nsamples': np.zeros((wingen.nwin,)),
           'fscale': dsp.fscale(WELCH_WIN_LENGTH_SAMPLES, 1 / sglx.fs, one_sided=True),
           'tscale': wingen.tscale(fs=sglx.fs)}
    win['spectral_density'] = np.zeros((len(win['fscale']), sglx.nc))
    # loop through the whole session
    for first, last in wingen.firstlast:
        D = sglx.read_samples(first_sample=first, last_sample=last)[0].transpose()
        # remove low frequency noise below 1 Hz
        D = dsp.hp(D, 1 / sglx.fs, [0, 1])
        iw = wingen.iw
        win['TRMS'][iw, :] = dsp.rms(D)
        win['nsamples'][iw] = D.shape[1]
        if spectra:
            # the last window may be smaller than what is needed for welch
            if last - first < WELCH_WIN_LENGTH_SAMPLES:
                continue
            # compute a smoothed spectrum using welch method
            _, w = signal.welch(D, fs=sglx.fs, window='hanning', nperseg=WELCH_WIN_LENGTH_SAMPLES,
                                detrend='constant', return_onesided=True, scaling='density',
                                axis=-1)
            win['spectral_density'] += w.T
        # print at least every 20 windows
        if (iw % min(20, max(int(np.floor(wingen.nwin / 75)), 1))) == 0:
            print_progress(iw, wingen.nwin)

    sglx.close()
    return win


def extract_rmsmap(fbin, out_folder=None, spectra=True):
    """
    Wrapper for rmsmap that outputs _ibl_ephysRmsMap and _ibl_ephysSpectra ALF files

    :param fbin: binary file in spike glx format (will look for attached metatdata)
    :param out_folder: folder in which to store output ALF files. Default uses the folder in which
     the `fbin` file lives.
    :param spectra: whether to compute the power spectrum (only need for lfp data)
    :type: bool
    :return: None
    """
    _logger.info(f"Computing QC for {fbin}")
    sglx = spikeglx.Reader(fbin)
    # check if output ALF files exist already:
    if out_folder is None:
        out_folder = Path(fbin).parent
    else:
        out_folder = Path(out_folder)
    alf_object_time = f'_iblqc_ephysTimeRms{sglx.type.upper()}'
    alf_object_freq = f'_iblqc_ephysSpectralDensity{sglx.type.upper()}'

    # crunch numbers
    rms = rmsmap(fbin, spectra=spectra)
    # output ALF files, single precision with the optional label as suffix before extension
    if not out_folder.exists():
        out_folder.mkdir()
    tdict = {'rms': rms['TRMS'].astype(np.single), 'timestamps': rms['tscale'].astype(np.single)}
    alfio.save_object_npy(out_folder, object=alf_object_time, dico=tdict)
    if spectra:
        fdict = {'power': rms['spectral_density'].astype(np.single),
                 'freqs': rms['fscale'].astype(np.single)}
        alfio.save_object_npy(out_folder, object=alf_object_freq, dico=fdict)


def _sample2v(ap_file):
    """
    Convert raw ephys data to Volts
    """
    md = spikeglx.read_meta_data(ap_file.with_suffix('.meta'))
    s2v = spikeglx._conversion_sample2v_from_meta(md)
    return s2v['ap'][0]


def ks2_to_alf(ks_path, bin_path, out_path, bin_file=None, ampfactor=1, label=None, force=True):
    """
    Convert Kilosort 2 output to ALF dataset for single probe data
    :param ks_path:
    :param bin_path: path of raw data
    :param out_path:
    :return:
    """
    m = ephysqc.phy_model_from_ks2_path(ks2_path=ks_path, bin_path=bin_path, bin_file=bin_file)
    ephysqc.spike_sorting_metrics_ks2(ks_path, m, save=True, save_path=out_path)
    ac = alf.EphysAlfCreator(m)
    ac.convert(out_path, label=label, force=force, ampfactor=ampfactor)


def extract_data(ks_path, ephys_path, out_path):
    efiles = spikeglx.glob_ephys_files(ephys_path)

    for efile in efiles:
        if efile.get('ap') and efile.ap.exists():
            ks2_to_alf(ks_path, ephys_path, out_path, bin_file=efile.ap,
                       ampfactor=_sample2v(efile.ap), label=None, force=True)

            extract_rmsmap(efile.ap, out_folder=out_path, spectra=False)
        if efile.get('lf') and efile.lf.exists():
            extract_rmsmap(efile.lf, out_folder=out_path)


# if __name__ == '__main__':
#
#    ephys_path = Path('C:/Users/Mayo/Downloads/raw_ephys_data')
#    ks_path = Path('C:/Users/Mayo/Downloads/KS2')
#    out_path = Path('C:/Users/Mayo/Downloads/alf')
#    extract_data(ks_path, ephys_path, out_path)
