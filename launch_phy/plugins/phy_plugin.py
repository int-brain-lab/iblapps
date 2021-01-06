# import from plugins/cluster_metrics.py
"""Show how to add a custom cluster metrics."""

import logging
import numpy as np
from phy import IPlugin
#import brainbox as bb
from defined_metrics import *


class IBLMetricsPlugin(IPlugin):
    def attach_to_controller(self, controller):
        """Note that this function is called at initialization time, *before* the supervisor is
        created. The `controller.cluster_metrics` items are then passed to the supervisor when
        constructing it."""

        def cv_amp(cluster_id):
            amps = controller.get_amplitudes(cluster_id).data
            return np.std(amps) / np.mean(amps) if len(amps) >= 1 else 0

        def cum_amp_drift(cluster_id):
            amps = controller.get_amplitudes(cluster_id).data
            # return cum_feat(amps) if len(amps) >= 2 else 0
            return np.sum(np.abs(np.diff(amps))) / len(amps) if len(amps) >= 2 else 0

        def cum_depth_drift(cluster_id):
            depths = controller.model.spike_depths[controller.get_spike_ids(cluster_id)].data
            # return cum_feat(depths) if len(depths) >= 2 else 0
            return np.sum(np.abs(np.diff(depths))) / len(depths) if len(depths) >= 2 else 0

        def cv_fr(cluster_id):
            ts = controller.get_spike_times(cluster_id).data
            fr = bb.singlecell.firing_rate(ts, hist_win=0.01, fr_win=0.25)
            return np.std(fr) / np.mean(fr) if len(fr) >= 1 else 0

        def frac_isi_viol(cluster_id):
            ts = controller.get_spike_times(cluster_id).data
            frac_isi_viol, _, _ = isi_viol(ts, rp=0.002)
            return frac_isi_viol

        def frac_missing_spikes(cluster_id):
            amps = controller.get_amplitudes(cluster_id).data
            try:
                frac_missing_spks, _, _ = feat_cutoff(
                    amps, spks_per_bin=10, sigma=4, min_num_bins=50)
            except:
                frac_missing_spks = np.NAN
            return frac_missing_spks

        def fp_estimate(cluster_id):
            ts = controller.get_spike_times(cluster_id).data
            return fp_est(ts, rp=0.002)

        def presence_ratio(cluster_id):
            ts = controller.get_spike_times(cluster_id).data
            pr, _ = pres_ratio(ts, hist_win=10)
            return pr

        def presence_ratio_std(cluster_id):
            ts = controller.get_spike_times(cluster_id).data
            pr, pr_bins = pres_ratio(ts, hist_win=10)
            return pr / np.std(pr_bins)

        def refp_viol(cluster_id):
            ts = controller.get_spike_times(cluster_id).data
            return FP_RP(ts)

        def n_cutoff(cluster_id):
            amps = controller.get_amplitudes(cluster_id).data
            return noise_cutoff(amps, quartile_length=.25)

        def mean_amp_true(cluster_id):
            #    Three cases: raw data

            # THIS APPROACH WORKS FOR BOTH SAMPLE WAVEFORMS AND RAW DATA
            # Specify 100 waveforms to average over, but may be less in the case where in the subset
            # of sample waveforms it doesn't contain 100 samples for specific cluster

            # N.B. These waveforms are median subtracted and have a high pass applied
            waveforms = controller._get_waveforms_with_n_spikes(
                cluster_id, 100)['data'].data

            # This way finds (max-min) on each channel, and then for each waveform finds the
            # maximum (max-min) on any channel and averages these
            # For each sample waveform find value on channel that has largest (max-min)
            p2p = np.max((np.max(waveforms, axis=1) - np.min(waveforms, axis=1)), axis=1)
            # Find mean p2p across all available sample waveforms
            p2p_mean = np.mean(p2p)

            # Alternative way
            # This way finds (max-min) on each channel, on each channel averages across all sample
            # waveforms and then finds the channel with the maximum average value
            p2p_mean_alt = np.max(np.mean(np.max(waveforms, axis=1) - np.min(waveforms, axis=1),
                                          axis=0))

            # Approach 2
            # Only works for sample waveforms
            # For non median subtracted and no high pass can access sample waveforms like this
            # spike_ids = controller.get_spike_ids(cluster_id).data
            # # Find which of the spike ids are in the subset of sample waveforms
            # _, _, idx = np.intersect1d(spike_ids,
            #                            controller.model.spike_waveforms['spike_ids'].data,
            #                            return_indices=True)
            # # Sample waveforms that are available for this cluster
            # # Shape of waveforms (n_waveforms, n_samples, n_channels)
            # # where len(idx) == n_waveforms
            # waveforms = controller.model.spike_waveforms['waveforms'][idx].data

            # RAW DATA CASE

            return p2p_mean

        #    no waveforms

        # def ptp_sigma(cluster_id):
            # this also requires checking 3 cases above.

        def m_label(cluster_id):
            ts = controller.get_spike_times(cluster_id).data
            amps = controller.get_amplitudes(cluster_id).data
            metrics_label = int(FP_RP(ts) and noise_cutoff(amps, quartile_length=.25) < 20)
            return metrics_label
            # if amplitudes are correct (i.e. raw data or sample wfs exist):
            #metrics_label = (FP_RP(ts) and noise_cutoff(amps,quartile_length=.25)<20 and np.mean(amps)>50)

        # Use this dictionary to define custom cluster metrics.
        # We memcache the function so that cluster metrics are only computed once and saved
        # within the session, and also between sessions (the memcached values are also saved
        # on disk).
        controller.cluster_metrics['cv_amp'] = controller.context.memcache(cv_amp)
        controller.cluster_metrics['cum_amp_drift'] = controller.context.memcache(cum_amp_drift)

        # NOTE: the branch of phylib which defines model.spike_depths is not working yet
        # controller.cluster_metrics['cum_depth_drift'] = controller.context.memcache(
        #     cum_depth_drift)

        controller.cluster_metrics['cv_fr'] = controller.context.memcache(cv_fr)
        controller.cluster_metrics['frac_isi_viol'] = controller.context.memcache(frac_isi_viol)
        controller.cluster_metrics['frac_missing_spikes'] = controller.context.memcache(
            frac_missing_spikes)
        controller.cluster_metrics['fp_estimate'] = controller.context.memcache(fp_estimate)
        controller.cluster_metrics['presence_ratio'] = controller.context.memcache(presence_ratio)
        controller.cluster_metrics['presence_ratio_std'] = controller.context.memcache(
            presence_ratio_std)
        controller.cluster_metrics['refp_viol'] = controller.context.memcache(refp_viol)
        controller.cluster_metrics['noise_cutoff'] = controller.context.memcache(n_cutoff)
        controller.cluster_metrics['mean_amp'] = controller.context.memcache(mean_amp_true)
        controller.cluster_metrics['metrics_label'] = controller.context.memcache(m_label)
