# import from plugins/cluster_metrics.py
"""Show how to add a custom cluster metrics."""

import numpy as np
from phy import IPlugin
#import brainbox as bb
from launch_phy.defined_metrics import *


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
                frac_missing_spks, _, _ = feat_cutoff(amps, spks_per_bin=10, sigma=4, min_num_bins=50)
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
            pr, pr_bins  = pres_ratio(ts, hist_win=10)
            return pr / np.std(pr_bins)

        def refp_viol(cluster_id):
            ts = controller.get_spike_times(cluster_id).data
            return FP_RP(ts)

        def n_cutoff(cluster_id):
            amps = controller.get_amplitudes(cluster_id).data
            return noise_cutoff(amps, quartile_length=.25)

        #def mean_amp_true(cluster_id):
        #    Three cases: raw data
        #    Sample waveforms
        #    no waveforms

        #def ptp_sigma(cluster_id):
            #this also requires checking 3 cases above.

        # Use this dictionary to define custom cluster metrics.
        # We memcache the function so that cluster metrics are only computed once and saved
        # within the session, and also between sessions (the memcached values are also saved
        # on disk).
        controller.cluster_metrics['cv_amp'] = controller.context.memcache(cv_amp)
        controller.cluster_metrics['cum_amp_drift'] = controller.context.memcache(cum_amp_drift)
        controller.cluster_metrics['cum_depth_drift'] = controller.context.memcache(cum_depth_drift)
        controller.cluster_metrics['cv_fr'] = controller.context.memcache(cv_fr)
        controller.cluster_metrics['frac_isi_viol'] = controller.context.memcache(frac_isi_viol)
        controller.cluster_metrics['frac_missing_spikes'] = controller.context.memcache(frac_missing_spikes)
        controller.cluster_metrics['fp_estimate'] = controller.context.memcache(fp_estimate)
        controller.cluster_metrics['presence_ratio'] = controller.context.memcache(presence_ratio)
        controller.cluster_metrics['presence_ratio_std'] = controller.context.memcache(presence_ratio_std)
        controller.cluster_metrics['refp_viol'] = controller.context.memcache(refp_viol)
        controller.cluster_metrics['noise_cutoff'] = controller.context.memcache(n_cutoff)
        controller.cluster_metrics['mean_amp'] = controller.context.memcache(mean_amp_true)
        controller.cluster_metrics['ptp_sigma'] = controller.context.memcache(ptp_sigma)
