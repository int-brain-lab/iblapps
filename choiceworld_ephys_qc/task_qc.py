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
import ViewEphysQC as ViewEphysQC

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
        self.n_trials = self.extractor.data['intervals_0'].size

        # Print failed
        outcome, results, outcomes = self.compute_session_status()
        map = {k: [] for k in set(outcomes.values())}
        for k, v in outcomes.items():
            map[v].append(k[6:])
        for k, v in map.items():
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
        self.frame['intervals_0'] = self.extractor.data['intervals_0']
        self.frame['intervals_1'] = self.extractor.data['intervals_1']
        self.frame.insert(loc=0, column='trial_no', value=self.frame.index)

    def create_plots(self, display, wheel_display=None, trial_events=None):
        """
        Plots the data for bnc1 (sound) and bnc2 (frame2ttl)
        :param display: An axes handle on which to plot the TTL events
        :param wheel_display: An axes handle on which to plot the wheel trace
        :param trial_events: A list of Bpod trial events to plot, e.g. ['stimFreeze_times'],
        if None, valve, sound and stimulus events are plotted
        :return: None
        """
        if trial_events is None:
            # Default trial events to plot as vertical lines
            trial_events = [
                'goCue_times',
                'goCueTrigger_times',
                'errorCue_times',
                'valveOpen_times',
                'stimFreeze_times',
                'stimOff_times',
                'stimOn_times'
            ]

        plot_args = {
            'ymin': 0,
            'ymax': 3,
            'linewidth': 2,
            'ax': display
        }

        bnc1 = self.extractor.BNC1
        bnc2 = self.extractor.BNC2
        trial_data = self.extractor.data

        plots.squares(bnc1['times'], bnc1['polarities'] * 0.4 + 1, ax=display, color='k')
        plots.squares(bnc2['times'], bnc2['polarities'] * 0.4 + 2, ax=display, color='k')
        for event, c in zip(trial_events, cycle(TABLEAU_COLORS.keys())):
            plot_args['linestyle'] = random.choice(('-', '--', '-.', ':'))
            plots.vertical_lines(trial_data[event], label=event, color=c, **plot_args)
        display.legend()
        display.set_yticklabels(['', 'frame2ttl', 'sound', ''])
        display.set_yticks([0, 1, 2, 3])
        display.set_ylim([0, 3])

        if wheel_display:
            wheel_plot_args = {
                'ax': wheel_display,
                'ymin': trial_data['wheel_position'].min(),
                'ymax': trial_data['wheel_position'].max()}
            plot_args = {**plot_args, **wheel_plot_args}

            wheel_display.plot(trial_data['wheel_timestamps'], trial_data['wheel_position'], 'k-x')
            for event, c in zip(trial_events, cycle(TABLEAU_COLORS.keys())):
                plots.vertical_lines(trial_data[event], label=event, color=c, **plot_args)


if __name__ == "__main__":
    # https://docs.google.com/document/d/1X-ypFEIxqwX6lU9pig4V_zrcR5lITpd8UJQWzW9I9zI/edit#
    parser = argparse.ArgumentParser(description='Quick viewer to see the behaviour data from'
                                                 'choice world sessions.')
    parser.add_argument('session', help='session uuid')
    parser.add_argument('--bpod', action='store_true', help='run QC on Bpod data only (no FPGA)')
    args = parser.parse_args()  # returns data from the options specified (echo)

    WHEEL = True
    qc = QcFrame(args.session, bpod_only=args.bpod)
    if WHEEL:
        wheel_data = {'re_pos': qc.extractor.data['wheel_position'],
                      're_ts': qc.extractor.data['wheel_timestamps']}
        w = ViewEphysQC.viewqc(wheel=wheel_data)
        qc.create_plots(w.wplot.canvas.ax, wheel_display=w.wplot.canvas.ax2)
    else:
        w = ViewEphysQC.viewqc()
        qc.create_plots(w.wplot.canvas.ax)

    w.update_df(qc.frame)
    qt.run_app()
