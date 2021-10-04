# import from plugins/cluster_metrics.py
"""Show how to add a custom cluster metrics."""

import logging
import numpy as np
from phy import IPlugin
import pandas as pd
from pathlib import Path
from brainbox.metrics.single_units import quick_unit_metrics


class IBLMetricsPlugin(IPlugin):
    def attach_to_controller(self, controller):
        """Note that this function is called at initialization time, *before* the supervisor is
        created. The `controller.cluster_metrics` items are then passed to the supervisor when
        constructing it."""

        clusters_file = Path(controller.dir_path.joinpath('clusters.metrics.pqt'))
        if clusters_file.exists():
            self.metrics = pd.read_parquet(clusters_file)
        else:
            self.metrics = None
            return

        def amplitudes(cluster_id):
            amps = controller.get_cluster_amplitude(cluster_id)
            return amps * 1e6

        def amp_max(cluster_id):
            return self.metrics['amp_max'].iloc[cluster_id] * 1e6

        def amp_min(cluster_id):
            return self.metrics['amp_min'].iloc[cluster_id] * 1e6

        def amp_median(cluster_id):
            return self.metrics['amp_median'].iloc[cluster_id] * 1e6

        def amp_std_dB(cluster_id):
            return self.metrics['amp_std_dB'].iloc[cluster_id]

        def contamination(cluster_id):
            return self.metrics['contamination'].iloc[cluster_id]

        def contamination_alt(cluster_id):
            return self.metrics['contamination_alt'].iloc[cluster_id]

        def drift(cluster_id):
            return self.metrics['drift'].iloc[cluster_id]

        def missed_spikes_est(cluster_id):
            return self.metrics['missed_spikes_est'].iloc[cluster_id]

        def noise_cutoff(cluster_id):
            return self.metrics['noise_cutoff'].iloc[cluster_id]

        def presence_ratio(cluster_id):
            return self.metrics['presence_ratio'].iloc[cluster_id]

        def presence_ratio_std(cluster_id):
            return self.metrics['presence_ratio_std'].iloc[cluster_id]

        def slidingRP_viol(cluster_id):
            return self.metrics['slidingRP_viol'].iloc[cluster_id]

        def spike_count(cluster_id):
            return self.metrics['spike_count'].iloc[cluster_id]

        def firing_rate(cluster_id):
            return self.metrics['firing_rate'].iloc[cluster_id]

        def label(cluster_id):
            return self.metrics['label'].iloc[cluster_id]

        def ks2_label(cluster_id):
            if 'ks2_label' in self.metrics.columns:
                return self.metrics['ks2_label'].iloc[cluster_id]
            else:
                return 'nan'

        controller.cluster_metrics['amplitudes'] = controller.context.memcache(amplitudes)
        controller.cluster_metrics['amp_max'] = controller.context.memcache(amp_max)
        controller.cluster_metrics['amp_min'] = controller.context.memcache(amp_min)
        controller.cluster_metrics['amp_median'] = controller.context.memcache(amp_median)
        controller.cluster_metrics['amp_std_dB'] = controller.context.memcache(amp_std_dB)
        controller.cluster_metrics['contamination'] = controller.context.memcache(contamination)
        controller.cluster_metrics['contamination_alt'] = controller.context.memcache(contamination_alt)
        controller.cluster_metrics['drift'] = controller.context.memcache(drift)
        controller.cluster_metrics['missed_spikes_est'] = controller.context.memcache(missed_spikes_est)
        controller.cluster_metrics['noise_cutoff'] = controller.context.memcache(noise_cutoff)
        controller.cluster_metrics['presence_ratio'] = controller.context.memcache(presence_ratio)
        controller.cluster_metrics['presence_ratio_std'] = controller.context.memcache(presence_ratio_std)
        controller.cluster_metrics['slidingRP_viol'] = controller.context.memcache(slidingRP_viol)
        controller.cluster_metrics['spike_count'] = controller.context.memcache(spike_count)
        controller.cluster_metrics['firing_rate'] = controller.context.memcache(firing_rate)
        controller.cluster_metrics['label'] = controller.context.memcache(label)
        controller.cluster_metrics['ks2_label'] = controller.context.memcache(ks2_label)
