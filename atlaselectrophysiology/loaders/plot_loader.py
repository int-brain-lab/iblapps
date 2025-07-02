from dataclasses import dataclass
import logging
from matplotlib import cm
import numpy as np
from typing import Optional, Union, List
from qtpy import QtGui

from brainbox.population.decode import xcorr
from brainbox.task import passive
from iblutil.numerical import bincount2D
from iblutil.util import Bunch

logger = logging.getLogger(__name__)

# TODO docstrings and typing and logging
# TODO make sure the cluster idx for waveforms and autocorrelogram makes sense
# TODO decorator for when data doesn't exist
# TODO arrange channels 2 banks when we have metadata and not channels data
# TODO ap rms do we need to apply all this median subtraction etc?


BNK_SIZE = 10
AUTOCORR_BIN_SIZE = 0.25 / 1000
AUTOCORR_WIN_SIZE = 10 / 1000

FS = 30000


@dataclass
class ScatterData:
    x: np.ndarray
    y: np.ndarray
    levels: Union[float, np.ndarray]
    colours: np.ndarray
    pen: Optional[any]
    size: np.ndarray
    symbol: Union[str, np.ndarray]
    xrange: np.ndarray
    xaxis: str
    title: str
    cmap: str
    cluster: bool
    yrange: Optional[np.ndarray] = None
    pad: Optional[float] = None


@dataclass
class ImageData:
    img: np.ndarray
    scale: np.ndarray
    levels: np.ndarray
    offset: np.ndarray
    xrange: np.ndarray
    xaxis: str
    cmap: str
    title: str
    yrange: Optional[np.ndarray] = None
    pad: Optional[float] = None


@dataclass
class LineData:
    x: np.ndarray
    y: np.ndarray
    xrange: np.ndarray
    xaxis: str
    yrange: Optional[np.ndarray] = None
    pad: Optional[float] = None


@dataclass
class ProbeData:
    img: List[np.ndarray]
    scale: List[np.ndarray]
    levels: np.ndarray
    offset: List[np.ndarray]
    xrange: np.ndarray
    cmap: str
    title: str
    boundaries: Optional[np.ndarray] = None
    yrange: Optional[np.ndarray] = None
    pad: Optional[float] = None


FILTER_MATCH = {
    'IBL good': ('label', 1),
    'KS good': ('ks2_label', 'good'),
    'KS mua': ('ks2_label', 'mua'),
}

TBIN = 0.05
DBIN = 5


def compute_spike_average(spikes):
    # Hack!

    exists = spikes.pop('exists')
    spike_df = spikes.to_df().groupby('clusters')
    avgs = spike_df.agg(['mean', 'count'])
    fr = avgs['depths']['count'].values / spikes['times'].max()
    spikes['exists'] = exists
    return avgs.index.values, avgs['depths']['mean'].values, avgs['amps']['mean'].values * 1e6, fr


def compute_bincount(spike_times, spike_depths, spike_amps, xbin=TBIN, ybin=DBIN, xlim=None, ylim=None, **kwargs):

    count, times, depths = bincount2D(spike_times, spike_depths, xbin=xbin, ybin=ybin,
                                      xlim=xlim, ylim=ylim, **kwargs)

    amp, times, depths = bincount2D(spike_times, spike_depths, xbin=xbin, ybin=ybin,
                                    xlim=xlim, ylim=ylim, weights=spike_amps)

    return count, amp, times, depths


def group_bincount(arr: np.ndarray, group_size: int, axis: int = 1) -> np.ndarray:
    """
    Average over chunks of `group_size` along the given axis.
    If leftover elements exist, sum them and append as the final group.

    Parameters
    ----------
    arr : np.ndarray
        2D array to process.
    group_size : int
        Number of elements per group to average.
    axis : int, optional
        Axis to operate on: 0 (rows) or 1 (columns). Default is 1.

    Returns
    -------
    np.ndarray
        Array with grouped means and a final summed group if leftovers exist.
    """
    if arr.ndim != 2:
        raise ValueError("Input array must be 2D.")
    if axis not in (0, 1):
        raise ValueError("Axis must be 0 or 1.")

    # Transpose if operating on axis 0 to reuse logic
    if axis == 0:
        arr = arr.T

    num_elements = arr.shape[1]
    num_full = num_elements // group_size
    full_cols = num_full * group_size

    arr_full = arr[:, :full_cols]
    arr_extra = arr[:, full_cols:]

    # Compute mean over full groups
    arr_grouped = arr_full.reshape(arr.shape[0], num_full, group_size)
    arr_avg = arr_grouped.sum(axis=2)

    # Sum the leftover group (if any)
    if arr_extra.shape[1] > 0:
        arr_sum = arr_extra.sum(axis=1, keepdims=True)
        result = np.concatenate([arr_avg, arr_sum], axis=1)
    else:
        result = arr_avg

    return result.T if axis == 0 else result


class PlotLoader:
    def __init__(self, data, metadata, shank_idx=0):
        self.data = data

        if metadata['exists']:
            self.chn_coords = np.c_[metadata['x'], metadata['y']]
            self.chn_ind_raw = metadata['ind']
            self.chn_ind = np.arange(self.chn_ind_raw.size)

            self.chn_min = np.min(self.chn_coords[:, 1])
            self.chn_max = np.max(self.chn_coords[:, 1])
            self.chn_diff = np.min(np.abs(np.diff(np.unique(self.chn_coords[:, 1]))))
            self.chn_full = np.arange(self.chn_min, self.chn_max + self.chn_diff, self.chn_diff)

            self.N_BNK = len(np.unique(self.chn_coords[:, 0]))
            self.idx_full = np.where(np.isin(self.chn_full, self.chn_coords[:, 1]))[0]

        else:
            # TODO handle this logic in the data loading part
            self.chn_coords_all = data['channels']['localCoordinates']
            self.chn_ind_all = data['channels']['rawInd'].astype(int)

            self.chn_min = np.min(self.chn_coords_all[:, 1])
            self.chn_max = np.max(self.chn_coords_all[:, 1])
            self.chn_diff = np.min(np.abs(np.diff(np.unique(self.chn_coords_all[:, 1]))))

            self.chn_full = np.arange(self.chn_min, self.chn_max + self.chn_diff, self.chn_diff)

            chn_x = np.unique(self.chn_coords_all[:, 0])
            chn_x_diff = np.diff(chn_x)
            n_shanks = np.sum(chn_x_diff > 100) + 1

            groups = np.split(chn_x, np.where(np.diff(chn_x) > 100)[0] + 1)

            if n_shanks > 1:
                shanks = {}
                for iShank, grp in enumerate(groups):
                    if len(grp) == 1:
                        grp = np.array([grp[0], grp[0]])
                    shanks[iShank] = [grp[0], grp[1]]

                shank_chns = np.bitwise_and(self.chn_coords_all[:, 0] >= shanks[shank_idx][0],
                                            self.chn_coords_all[:, 0] <= shanks[shank_idx][1])
                self.chn_coords = self.chn_coords_all[shank_chns, :]
                self.chn_ind = self.chn_ind_all[shank_chns]
            else:
                self.chn_coords = self.chn_coords_all
                self.chn_ind = self.chn_ind_all

            chn_sort = np.argsort(self.chn_coords[:, 1])
            self.chn_coords = self.chn_coords[chn_sort]
            self.chn_ind = self.chn_ind[chn_sort]

            self.N_BNK = len(np.unique(self.chn_coords[:, 0]))
            self.idx_full = np.where(np.isin(self.chn_full, self.chn_coords[:, 1]))[0]

        self.filter_units('All')
        self.get_data()

    def _get_plots(self, plot_prefix):
        """
        Find all methods that begin with plot_prefix.

        Returns
        -------
        Dict[str, function]
        """

        results = Bunch()
        for attr_name in dir(self):
            if attr_name.startswith(plot_prefix):
                method = getattr(self, attr_name)
                if callable(method):
                    results.update(method())

        return results

    def get_plots(self):

        self.image_plots = self._get_plots('image')
        self.scatter_plots = self._get_plots('scatter')
        self.line_plots = self._get_plots('line')
        self.probe_plots = self._get_plots('probe')

    @property
    def spike_amps(self):
        return self.data['spikes']['amps'][self.spike_idx][self.kp_idx]

    @property
    def spike_depths(self):
        return self.data['spikes']['depths'][self.spike_idx][self.kp_idx]

    @property
    def spike_clusters(self):
        return self.data['spikes']['clusters'][self.spike_idx][self.kp_idx]

    @property
    def spike_times(self):
        return self.data['spikes']['times'][self.spike_idx][self.kp_idx]

    def get_data(self):

        if not self.data['spikes']['exists']:
            return

        self.clust_id, self.avg_depth, self.avg_amp, self.avg_fr = compute_spike_average(self.data['spikes'])
        self.update_data()
        self.compute_timescales()

    def update_data(self):

        if not self.data['spikes']['exists']:
            return

        self.chn_min_bc = np.min(np.r_[self.chn_min, self.spike_depths])
        self.chn_max_bc = np.max(np.r_[self.chn_max, self.spike_depths])

        self.fr, self.amp, self.times, self.depths = compute_bincount(
            self.spike_times, self.spike_depths, self.spike_amps, ylim=[self.chn_min_bc, self.chn_max_bc])

    def filter_units(self, filter_type):

        if not self.data['spikes']['exists']:
            return
        try:
            if filter_type == "All":
                self.cluster_idx = np.arange(self.data['clusters'].channels.size)
                self.spike_idx = np.arange(self.data['spikes']['clusters'].size)
            else:
                column, condition = FILTER_MATCH[filter_type]
                self.cluster_idx = np.where(self.data['clusters'].metrics[column] == condition)[0]
                self.spike_idx = np.where(np.isin(self.data['spikes']['clusters'], self.cluster_idx))[0]

            self.kp_idx = np.where(~np.isnan(self.data['spikes']['depths'][self.spike_idx]) &
                                   ~np.isnan(self.data['spikes']['amps'][self.spike_idx]))[0]
        except Exception:
            logger.warning(f'{filter_type} metrics not found will return all units instead')
            self.filter_units('All')

    # -------------------------------------------------------------------------------------------------
    # Scatter plots
    # -------------------------------------------------------------------------------------------------

    def scatter_firing_rate(self):
        if not self.data['spikes']['exists']:
            return {}

        A_BIN = 10
        subsample = 100

        times = self.spike_times[0:-1:subsample]
        depths = self.spike_depths[0:-1:subsample]
        amps = self.spike_amps[0:-1:subsample]

        amp_range = np.quantile(amps, [0, 0.9])
        amp_bins = np.linspace(amp_range[0], amp_range[1], A_BIN)
        colour_bin = np.linspace(0.0, 1.0, A_BIN + 1)

        colormap = cm.get_cmap('BuPu')(colour_bin)[..., :3]  # RGB only
        colours_rgb = (colormap * 255).astype(np.int32)

        spikes_colours = np.full(amps.size, QtGui.QColor('#000000'))
        spikes_size = np.zeros(amps.size)

        for i in range(A_BIN):
            if i == A_BIN - 1:
                idx = amps > amp_bins[i]
                # TODO remove the QT part and put in the qt_plots
                spikes_colours[idx] = QtGui.QColor('#400080')  # Dark purple for saturated
            else:
                idx = (amps > amp_bins[i]) & (amps <= amp_bins[i + 1])
                spikes_colours[idx] = QtGui.QColor(*colours_rgb[i])

            spikes_size[idx] = i / (A_BIN / 4)

        xrange = np.array([np.min(times), np.max(times)])

        scatter = ScatterData(
            x=times,
            y=depths,
            levels=amp_range * 1e6,
            colours=spikes_colours,
            pen=None,
            size=spikes_size,
            symbol=np.array('o'),
            xrange=xrange,
            xaxis='Time (s)',
            title='Amplitude (uV)',
            cmap='BuPu',
            cluster=False
        )

        return {'Amplitude': scatter}

    def scatter_amp_depth_fr(self):
        if not self.data['spikes']['exists']:
            return {}

        scatter = ScatterData(
            x=self.avg_amp[self.cluster_idx],
            y=self.avg_depth[self.cluster_idx],
            levels=np.quantile(self.avg_fr[self.cluster_idx], [0, 1]),
            colours=self.avg_fr[self.cluster_idx],
            pen='k',
            size=np.array(8),
            symbol=np.array('o'),
            xrange=np.array([0.9 * self.avg_amp[self.cluster_idx].min(), 1.1 * self.avg_amp[self.cluster_idx].max()]),
            xaxis='Amplitude (uV)',
            title='Firing Rate (Sp/s)',
            cmap='hot',
            cluster=True
        )

        return {'Cluster Amp vs Depth vs FR': scatter}

    def scatter_amp_depth_duration(self):
        if not self.data['spikes']['exists']:
            return {}

        scatter = ScatterData(
            # TODO change this to be clust_idx
            x=self.avg_amp[self.cluster_idx],
            y=self.avg_depth[self.cluster_idx],
            levels=np.array([-1.5, 1.5]),
            colours=self.data['clusters']['peakToTrough'][self.cluster_idx],
            pen='k',
            size=np.array(8),
            symbol=np.array('o'),
            xrange=np.array([0.9 * self.avg_amp[self.cluster_idx].min(), 1.1 * self.avg_amp[self.cluster_idx].max()]),
            xaxis='Amplitude (uV)',
            title='Peak to Trough duration (ms)',
            cmap='RdYlGn',
            cluster=True
        )

        return {'Cluster Amp vs Depth vs Duration': scatter}

    def scatter_fr_depth_amp(self):
        if not self.data['spikes']['exists']:
            return {}

        scatter = ScatterData(
            x=self.avg_fr[self.cluster_idx],
            y=self.avg_depth[self.cluster_idx],
            levels=np.quantile(self.avg_amp[self.cluster_idx], [0, 1]),
            colours=self.avg_amp[self.cluster_idx],
            pen='k',
            size=np.array(8),
            symbol=np.array('o'),
            xrange=np.array([0.9 * self.avg_fr[self.cluster_idx].min(), 1.1 * self.avg_fr[self.cluster_idx].max()]),
            xaxis='Firing Rate (Sp/s)',
            title='Amplitude (uV)',
            cmap='magma',
            cluster=True
        )

        return {'Cluster FR vs Depth vs Amp': scatter}

    # -------------------------------------------------------------------------------------------------
    # Image plots
    # -------------------------------------------------------------------------------------------------
    def image_firing_rate(self):
        if not self.data['spikes']['exists']:
            return {}

        xscale = (self.times[-1] - self.times[0]) / self.fr.shape[1]
        yscale = (self.depths[-1] - self.depths[0]) / self.fr.shape[0]

        img = ImageData(
            img=self.fr.T,
            scale=np.array([xscale, yscale]),
            levels=np.quantile(np.mean(self.fr.T, axis=0), [0, 1]),
            offset=np.array([0, self.chn_min]),
            xrange=np.array([self.times[0], self.times[-1]]),
            xaxis='Time (s)',
            cmap='binary',
            title='Firing Rate'
        )

        return {'Firing Rate': img}

    def image_correlation(self):
        if not self.data['spikes']['exists']:
            return {}

        dbin = 40
        factor = int(dbin / DBIN)
        bincount = group_bincount(self.fr, factor, axis=0)
        depths = self.depths[::factor]

        corr = np.corrcoef(bincount)
        corr[np.isnan(corr)] = 0
        scale = (np.max(depths) - np.min(depths)) / corr.shape[0]

        img = ImageData(
            img=corr,
            scale=np.array([scale, scale]),
            levels=np.array([np.min(corr), np.max(corr)]),
            offset=np.array([self.chn_min, self.chn_min]),
            xrange=np.array([self.chn_min, self.chn_max]),
            cmap='viridis',
            title='Correlation',
            xaxis='Distance from probe tip (um)'
        )
        return {'Correlation': img}

    def image_rms_ap(self):
        return self._image_rms('AP')

    def image_rms_lf(self):
        return self._image_rms('LF')

    def _image_rms(self, format):
        # Finds channels that are at equivalent depth on probe and averages rms values for each
        # time point at same depth together

        if not self.data[f'rms_{format}']['exists']:
            return {}

        _rms = np.take(self.data[f'rms_{format}']['rms'], self.chn_ind, axis=1)
        _, self.chn_depth, chn_count = np.unique(self.chn_coords[:, 1], return_index=True, return_counts=True)
        self.chn_depth_eq = np.copy(self.chn_depth)
        self.chn_depth_eq[np.where(chn_count == 2)] += 1

        def avg_chn_depth(a):
            return (np.mean([a[self.chn_depth], a[self.chn_depth_eq]], axis=0))

        def get_median(a):
            return (np.median(a))

        def median_subtract(a):
            return (a - np.median(a))

        img = np.apply_along_axis(avg_chn_depth, 1, _rms * 1e6)
        median = np.mean(np.apply_along_axis(get_median, 1, img))
        # Medium subtract to remove bands, but add back average median so values make sense
        img = np.apply_along_axis(median_subtract, 1, img) + median

        # For when the data is not contiguous, i.e say you have 16 channels then a big gap and then 32 at the top
        img_full = np.full((img.shape[0], self.chn_full.shape[0]), np.nan)
        img_full[:, self.idx_full] = img

        levels = np.quantile(img, [0.1, 0.9])
        xscale = (self.data[f'rms_{format}']['timestamps'][-1] - self.data[f'rms_{format}']['timestamps'][0]) / img_full.shape[0]
        yscale = (self.chn_max - self.chn_min) / img_full.shape[1]

        if format == 'AP':
            cmap = 'plasma'
        else:
            cmap = 'inferno'

        img = ImageData(
            img=img_full,
            scale=np.array([xscale, yscale]),
            levels=levels,
            offset=np.array([0, self.chn_min]),
            cmap=cmap,
            xrange=np.array([self.data[f'rms_{format}']['timestamps'][0], self.data[f'rms_{format}']['timestamps'][-1]]),
            xaxis=self.data[f'rms_{format}']['xaxis'],
            title=format + ' RMS (uV)'
        )

        return {f'rms {format}': img}

    def image_lfp_spectrum(self):
        if not self.data['psd_lf']['exists']:
            return {}

        # Power spectrum image
        freq_range = [0, 300]
        freq_idx = np.where((self.data['psd_lf']['freqs'] >= freq_range[0]) &
                            (self.data['psd_lf']['freqs'] < freq_range[1]))[0]
        _lfp = np.take(self.data['psd_lf']['power'][freq_idx], self.chn_ind, axis=1)
        _lfp_dB = 10 * np.log10(_lfp)
        _, self.chn_depth, chn_count = np.unique(self.chn_coords[:, 1], return_index=True,
                                                 return_counts=True)
        self.chn_depth_eq = np.copy(self.chn_depth)
        self.chn_depth_eq[np.where(chn_count == 2)] += 1

        def avg_chn_depth(a):
            return (np.mean([a[self.chn_depth], a[self.chn_depth_eq]], axis=0))

        img = np.apply_along_axis(avg_chn_depth, 1, _lfp_dB)
        img_full = np.full((img.shape[0], self.chn_full.shape[0]), np.nan)
        img_full[:, self.idx_full] = img

        levels = np.quantile(img, [0.1, 0.9])
        xscale = (freq_range[-1] - freq_range[0]) / img_full.shape[0]
        yscale = (self.chn_max - self.chn_min) / img_full.shape[1]

        img = ImageData(
            img=img_full,
            scale=np.array([xscale, yscale]),
            levels=levels,
            offset=np.array([0, self.chn_min]),
            cmap='viridis',
            xrange=np.array([freq_range[0], freq_range[-1]]),
            xaxis='Frequency (Hz)',
            title='PSD (dB)'
        )

        return {'LF spectrum': img}

    def image_passive_events(self):
        stim_keys = ['valveOn', 'toneOn', 'noiseOn', 'leftGabor', 'rightGabor']
        data_img = {}
        if not self.data['pass_stim']['exists'] and not self.data['gabor']['exists']:
            return data_img
        elif not self.data['pass_stim']['exists'] and self.data['gabor']['exists']:
            stim_types = ['leftGabor', 'rightGabor']
            stims = {stim_type: self.data['gabor'][stim_type] for stim_type in stim_types}
        elif self.data['pass_stim']['exists'] and not self.data['gabor']['exists']:
            stim_types = ['valveOn', 'toneOn', 'noiseOn']
            stims = {stim_type: self.data['pass_stim'][stim_type] for stim_type in stim_types}
        else:
            stim_types = stim_keys
            stims = {stim_type: self.data['pass_stim'][stim_type] for stim_type in stim_types[0:3]}
            stims.update({stim_type: self.data['gabor'][stim_type] for stim_type in stim_types[3:]})

        passive_imgs = {}

        base_stim = 1
        pre_stim = 0.4
        post_stim = 1
        stim_events = passive.get_stim_aligned_activity(stims, self.spike_times, self.spike_depths,
                                                        pre_stim=pre_stim, post_stim=post_stim,
                                                        base_stim=base_stim, y_lim=[self.chn_min, self.chn_max])

        for stim_type, aligned_img in stim_events.items():
            xscale = (post_stim + pre_stim) / aligned_img.shape[1]
            yscale = ((self.chn_max - self.chn_min) / aligned_img.shape[0])

            levels = [-10, 10]

            img = ImageData(
                img=aligned_img.T,
                scale=np.array([xscale, yscale]),
                levels=levels,
                offset=np.array([-1 * pre_stim, self.chn_min]),
                cmap='bwr',
                xrange=[-1 * pre_stim, post_stim],
                xaxis='Time from Stim Onset (s)',
                title='Firing rate (z score)'
            )

            passive_imgs.update({stim_type: img})

        return passive_imgs

    def image_raw_data(self):
        if not self.data['raw_snippets']['exists']:
            return {}

        def gain2level(gain):
            return 10 ** (gain / 20) * 4 * np.array([-1, 1])

        levels = gain2level(-90)

        raw_imgs = {}

        for t, raw_img in self.data['raw_snippets']['images'].items():
            x_range = np.array([0, raw_img.shape[0] - 1]) / self.data['raw_snippets']['fs'] * 1e3
            xscale = (x_range[1] - x_range[0]) / raw_img.shape[0]
            yscale = (self.chn_max - self.chn_min) / raw_img.shape[1]

            img = ImageData(
                img=raw_img,
                scale=np.array([xscale, yscale]),
                levels=levels,
                offset=np.array([0, self.chn_min]),
                cmap='bone',
                xrange=x_range,
                xaxis='Time (ms)',
                title='Power (uV)'
            )
            raw_imgs[f'Raw ap t={t}'] = img

        return raw_imgs

    # -------------------------------------------------------------------------------------------------
    # Line plots
    # -------------------------------------------------------------------------------------------------

    def line_firing_rate(self):
        if not self.data['spikes']['exists']:
            return {}

        dbin = 10
        factor = int(dbin / DBIN)
        bincount = group_bincount(self.fr, factor, axis=0)
        depths = self.depths[::factor]

        mean_fr = np.mean(bincount, axis=1)

        line = LineData(
            x=mean_fr,
            y=depths,
            xrange=np.array([0, np.max(mean_fr)]),
            xaxis='Firing Rate (Sp/s)'
        )

        return {'Firing Rate': line}

    def line_amplitude(self):
        if not self.data['spikes']['exists']:
            return {}

        dbin = 10
        factor = int(dbin / DBIN)
        bincount = group_bincount(self.amp, factor, axis=0)
        depths = self.depths[::factor]

        mean_amp = np.mean(bincount, axis=1) * 1e6

        line = LineData(
            x=mean_amp,
            y=depths,
            xrange=np.array([0, np.max(mean_amp)]),
            xaxis='Amplitude (uV)'
        )

        return {'Amplitude': line}

    # -------------------------------------------------------------------------------------------------
    # Probe plots
    # -------------------------------------------------------------------------------------------------
    def probe_rms_ap(self):
        return self._probe_rms('AP')

    def probe_rms_lf(self):
        return self._probe_rms('LF')

    def _probe_rms(self, format):

        if not self.data[f'rms_{format}']['exists']:
            return {}

        # Probe data
        rms_avg = (np.mean(self.data[f'rms_{format}']['rms'], axis=0)[self.chn_ind]) * 1e6
        probe_levels = np.quantile(rms_avg, [0.1, 0.9])
        probe_img, probe_scale, probe_offset = self.arrange_channels2banks(rms_avg)

        if format == 'AP':
            cmap = 'plasma'
        else:
            cmap = 'inferno'

        probe = ProbeData(
            img=probe_img,
            scale=probe_scale,
            offset=probe_offset,
            levels=probe_levels,
            cmap=cmap,
            xrange=np.array([0 * BNK_SIZE, (self.N_BNK) * BNK_SIZE]),
            title=format + ' RMS (uV)'
        )

        return {f'rms {format}': probe}

    def probe_lfp_spectrum(self):
        if not self.data['psd_lf']['exists']:
            return {}

        freq_bands = np.vstack(([0, 4], [4, 10], [10, 30], [30, 80], [80, 200]))
        data_probe = {}

        # Power spectrum in bands on probe
        for freq in freq_bands:
            freq_idx = np.where((self.data['psd_lf']['freqs'] >= freq[0]) & (self.data['psd_lf']['freqs'] < freq[1]))[0]
            lfp_avg = np.mean(self.data['psd_lf']['power'][freq_idx], axis=0)[self.chn_ind]
            lfp_avg_dB = 10 * np.log10(lfp_avg)
            probe_img, probe_scale, probe_offset = self.arrange_channels2banks(lfp_avg_dB)
            probe_levels = np.quantile(lfp_avg_dB, [0.1, 0.9])

            probe = ProbeData(
                img=probe_img,
                scale=probe_scale,
                offset=probe_offset,
                levels=probe_levels,
                cmap='viridis',
                xrange=np.array([0 * BNK_SIZE, (self.N_BNK) * BNK_SIZE]),
                title=f"{freq[0]} - {freq[1]} Hz (dB)"
            )
            data_probe.update({f"{freq[0]} - {freq[1]} Hz": probe})

        return data_probe

    def probe_rfmap(self):
        if not self.data['rf_map']['exists']:
            return {}

        (rf_map_times, rf_map_pos,
         rf_stim_frames) = passive.get_on_off_times_and_positions(self.data['rf_map'])

        rf_map, _ = \
            passive.get_rf_map_over_depth(rf_map_times, rf_map_pos, rf_stim_frames,
                                          self.spike_times,
                                          self.spike_depths,
                                          d_bin=160, y_lim=[self.chn_min_bc, self.chn_max_bc])
        rfs_svd = passive.get_svd_map(rf_map)
        img = {}
        img['on'] = np.vstack(rfs_svd['on'])
        img['off'] = np.vstack(rfs_svd['off'])

        yscale = ((self.chn_max - self.chn_min) / img['on'].shape[0])
        xscale = 1
        levels = np.quantile(np.c_[img['on'], img['off']], [0, 1])

        depths = np.linspace(self.chn_min, self.chn_max, len(rfs_svd['on']) + 1)
        data_img = {}
        sub_type = ['on', 'off']
        for sub in sub_type:
            sub_data = {
                f'RF Map - {sub}':
                    ProbeData(
                        img=[img[sub].T],
                        scale=[np.array([xscale, yscale])],
                        levels=levels,
                        offset=[np.array([0, self.chn_min])],
                        cmap='viridis',
                        xrange=np.array([0, 15]),
                        title='rfmap (dB)',
                        boundaries=depths
                    )
            }
            data_img.update(sub_data)

        return data_img

    def arrange_channels2banks(self, data):
        bnk_data = []
        bnk_scale = np.empty((self.N_BNK, 2))
        bnk_offset = np.empty((self.N_BNK, 2))
        for iX, x in enumerate(np.unique(self.chn_coords[:, 0])):
            bnk_idx = np.where(self.chn_coords[:, 0] == x)[0]

            bnk_ycoords = self.chn_coords[bnk_idx, 1]
            bnk_diff = np.min(np.abs(np.diff(bnk_ycoords)))

            # NP1.0 checkerboard
            if bnk_diff != self.chn_diff:
                bnk_full = np.arange(np.min(bnk_ycoords), np.max(bnk_ycoords) + bnk_diff, bnk_diff)
                _bnk_vals = np.full((bnk_full.shape[0]), np.nan)
                idx_full = np.where(np.isin(bnk_full, bnk_ycoords))
                _bnk_vals[idx_full] = data[bnk_idx]

                # Detect where the nans are, whether it is odd or even
                _bnk_data = _bnk_vals[np.newaxis, :]

                _bnk_yscale = ((self.chn_max -
                                self.chn_min) / _bnk_data.shape[1])
                _bnk_xscale = BNK_SIZE / _bnk_data.shape[0]

                _bnk_yoffset = np.min(bnk_ycoords)
                _bnk_xoffset = BNK_SIZE * iX

            else:  # NP2.0
                _bnk_vals = np.full((self.chn_full.shape[0]), np.nan)
                idx_full = np.where(np.isin(self.chn_full, bnk_ycoords))
                _bnk_vals[idx_full] = data[bnk_idx]

                _bnk_data = _bnk_vals[np.newaxis, :]

                _bnk_yscale = ((self.chn_max -
                                self.chn_min) / _bnk_data.shape[1])
                _bnk_xscale = BNK_SIZE / _bnk_data.shape[0]
                _bnk_yoffset = self.chn_min
                _bnk_xoffset = BNK_SIZE * iX

            bnk_data.append(_bnk_data)
            bnk_scale[iX, :] = np.array([_bnk_xscale, _bnk_yscale])
            bnk_offset[iX, :] = np.array([_bnk_xoffset, _bnk_yoffset])

        return bnk_data, bnk_scale, bnk_offset

    def compute_timescales(self):
        if not self.data['spikes']['exists']:
            return

        self.t_autocorr = 1e3 * np.arange((AUTOCORR_WIN_SIZE / 2) - AUTOCORR_WIN_SIZE,
                                          (AUTOCORR_WIN_SIZE / 2) + AUTOCORR_BIN_SIZE,
                                          AUTOCORR_BIN_SIZE)
        n_template = self.data['clusters']['waveforms'][0, :, 0].size
        self.t_template = 1e3 * (np.arange(n_template)) / FS

    def get_autocorr(self, clust_idx):
        idx = np.where(self.data['spikes']['clusters'] == self.clust_id[clust_idx])[0]
        autocorr = xcorr(self.data['spikes']['times'][idx], self.data['spikes']['clusters'][idx],
                         AUTOCORR_BIN_SIZE, AUTOCORR_WIN_SIZE)

        if self.data['clusters'].get('metrics', {}).get('cluster_id', None) is None:
            clust_id = self.clust_id[clust_idx]
        else:
            clust_id = self.data['clusters'].metrics.cluster_id[self.clust_id[clust_idx]]

        return autocorr[0, 0, :], clust_id

    def get_template_wf(self, clust_idx):
        template_wf = (self.data['clusters']['waveforms'][self.clust_id[clust_idx], :, 0])
        return template_wf * 1e6
