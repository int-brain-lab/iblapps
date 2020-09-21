import logging
import argparse
from itertools import cycle
import random
import collections

import numpy as np
import pandas as pd
import qt as qt
from matplotlib.colors import TABLEAU_COLORS

from oneibl.one import ONE
import ibllib.plots as plots
from ibllib.qc.task_metrics import TaskQC
from task_qc_viewer import ViewEphysQC

one = ONE()

_logger = logging.getLogger('ibllib')


class QcFrame(TaskQC):
    def __init__(self, session_path, bpod_only=False):
        """
        Loads and extracts the QC data for a given session path
        :param session_path: A str or Path to a Bpod session
        :param bpod_only: When True all data is extracted from Bpod instead of FPGA for ephys
        """
        super().__init__(session_path, one=one, log=_logger)
        self.load_data(bpod_only=bpod_only)
        self.compute()
        self.n_trials = self.extractor.data['intervals'].shape[0]
        self.wheel_data = {'re_pos': self.extractor.data.pop('wheel_position'),
                           're_ts': self.extractor.data.pop('wheel_timestamps')}

        # Print failed
        outcome, results, outcomes = self.compute_session_status()
        map = {k: [] for k in set(outcomes.values())}
        for k, v in outcomes.items():
            map[v].append(k[6:])
        for k, v in map.items():
            if k == 'PASS':
                continue
            print(f'The following checks were labelled {k}:')
            print('\n'.join(v), '\n')

        # Make DataFrame from the trail level metrics
        def get_trial_level_failed(d):
            new_dict = {k[6:]: v for k, v in d.items() if
                        isinstance(v, collections.Sized) and len(v) == self.n_trials}
            return pd.DataFrame.from_dict(new_dict)

        metrics = get_trial_level_failed(self.metrics)
        passed = get_trial_level_failed(self.passed)
        passed = passed.add_suffix('_passed')
        new_col = np.empty((metrics.columns.size + passed.columns.size,), dtype=object)
        new_col[0::2], new_col[1::2] = metrics.columns, passed.columns
        self.frame = pd.concat([metrics, passed], axis=1)[new_col]
        self.frame['intervals_0'] = self.extractor.data['intervals'][:, 0]
        self.frame['intervals_1'] = self.extractor.data['intervals'][:, 1]
        self.frame.insert(loc=0, column='trial_no', value=self.frame.index)

    def create_plots(self, axes, wheel_axes=None, trial_events=None, color_map=None,
                     line_style=None):
        """
        Plots the data for bnc1 (sound) and bnc2 (frame2ttl)
        :param axes: An axes handle on which to plot the TTL events
        :param wheel_axes: An axes handle on which to plot the wheel trace
        :param trial_events: A list of Bpod trial events to plot, e.g. ['stimFreeze_times'],
        if None, valve, sound and stimulus events are plotted
        :param color_map: A color map to use for the events, default is the tableau color map
        line_style: A line style map to use for the events, default is random.
        :return: None
        """
        color_map = color_map or TABLEAU_COLORS.keys()
        if trial_events is None:
            # Default trial events to plot as vertical lines
            trial_events = [
                'goCue_times',
                'goCueTrigger_times',
                'feedback_times',
                'stimFreeze_times',
                'stimOff_times',
                'stimOn_times'
            ]

        plot_args = {
            'ymin': 0,
            'ymax': 3,
            'linewidth': 2,
            'ax': axes
        }

        bnc1 = self.extractor.frame_ttls
        bnc2 = self.extractor.audio_ttls
        trial_data = self.extractor.data

        plots.squares(bnc1['times'], bnc1['polarities'] * 0.4 + 1, ax=axes, color='k')
        plots.squares(bnc2['times'], bnc2['polarities'] * 0.4 + 2, ax=axes, color='k')
        if line_style is None:
            linestyle = random.choices(('-', '--', '-.', ':'), k=len(trial_events))
        else:
            linestyle = line_style
        for event, c, l in zip(trial_events, cycle(color_map), linestyle):
            plots.vertical_lines(trial_data[event], label=event, color=c, linestyle=l, **plot_args)
        axes.legend()
        axes.set_yticklabels(['', 'frame2ttl', 'sound', ''])
        axes.set_yticks([0, 1, 2, 3])
        axes.set_ylim([0, 3])

        if wheel_axes:
            wheel_plot_args = {
                'ax': wheel_axes,
                'ymin': self.wheel_data['re_pos'].min(),
                'ymax': self.wheel_data['re_pos'].max()}
            plot_args = {**plot_args, **wheel_plot_args}

            wheel_axes.plot(self.wheel_data['re_ts'], self.wheel_data['re_pos'], 'k-x')
            for event, c, ln in zip(trial_events, cycle(color_map), linestyle):
                plots.vertical_lines(trial_data[event],
                                     label=event, color=c, linestyle=ln, **plot_args)


if __name__ == "__main__":
    """Run TaskQC viewer with wheel data
    For information on the QC checks see the QC Flags & failures document:
    https://docs.google.com/document/d/1X-ypFEIxqwX6lU9pig4V_zrcR5lITpd8UJQWzW9I9zI/edit#
    """
    # Parse parameters
    parser = argparse.ArgumentParser(description='Quick viewer to see the behaviour data from'
                                                 'choice world sessions.')
    parser.add_argument('session', help='session uuid')
    parser.add_argument('--bpod', action='store_true', help='run QC on Bpod data only (no FPGA)')
    args = parser.parse_args()  # returns data from the options specified (echo)
    event_map = {'goCue_times': ['#2ca02c', '-'],  # green
                 'goCueTrigger_times': ['#2ca02c', '--'],  # green
                 'errorCue_times': ['#d62728', '-'],  # red
                 'errorCueTrigger_times': ['#d62728', '--'],  # red
                 'valveOpen_times': ['#17becf', '-'],  # cyan
                 'stimFreeze_times': ['#0000ff', ':'],  # blue
                 'stimOff_times': ['#9400d3', '-'],  # dark violet
                 'stimOffTrigger_times': ['#9400d3', '--'],  # dark violet
                 'stimOn_times': ['#e377c2', '-'],  # pink
                 'stimOnTrigger_times': ['#e377c2', '--'],  # pink
                 'response_times': ['#8c564b', '-'],  # brown
                 }
    color_map_ev = []
    line_style_ev = []
    for v in event_map.values():
        color_map_ev.append(v[0])
        line_style_ev.append(v[1])

    # Run QC and plot
    qc = QcFrame(args.session, bpod_only=args.bpod)
    w = ViewEphysQC.viewqc(wheel=qc.wheel_data)
    qc.create_plots(w.wplot.canvas.ax,
                    wheel_axes=w.wplot.canvas.ax2,
                    trial_events=event_map.keys(),
                    color_map=color_map_ev,
                    line_style=line_style_ev)
    # Update table and callbacks
    w.update_df(qc.frame)
    qt.run_app()