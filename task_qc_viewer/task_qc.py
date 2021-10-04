import logging
import argparse
from itertools import cycle
import random
from collections.abc import Sized

import pandas as pd
import qt as qt
from matplotlib.colors import TABLEAU_COLORS

from one.api import ONE
import ibllib.plots as plots
from ibllib.io.extractors import ephys_fpga
from ibllib.qc.task_metrics import TaskQC
from ibllib.qc.task_extractors import TaskQCExtractor

from task_qc_viewer import ViewEphysQC

EVENT_MAP = {'goCue_times': ['#2ca02c', 'solid'],  # green
             'goCueTrigger_times': ['#2ca02c', 'dotted'],  # green
             'errorCue_times': ['#d62728', 'solid'],  # red
             'errorCueTrigger_times': ['#d62728', 'dotted'],  # red
             'valveOpen_times': ['#17becf', 'solid'],  # cyan
             'stimFreeze_times': ['#0000ff', 'solid'],  # blue
             'stimFreezeTrigger_times': ['#0000ff', 'dotted'],  # blue
             'stimOff_times': ['#9400d3', 'solid'],  # dark violet
             'stimOffTrigger_times': ['#9400d3', 'dotted'],  # dark violet
             'stimOn_times': ['#e377c2', 'solid'],  # pink
             'stimOnTrigger_times': ['#e377c2', 'dotted'],  # pink
             'response_times': ['#8c564b', 'solid'],  # brown
             }
cm = [EVENT_MAP[k][0] for k in EVENT_MAP]
ls = [EVENT_MAP[k][1] for k in EVENT_MAP]
CRITICAL_CHECKS = (
    'check_audio_pre_trial',
    'check_correct_trial_event_sequence',
    'check_error_trial_event_sequence',
    'check_n_trial_events',
    'check_response_feedback_delays',
    'check_response_stimFreeze_delays',
    'check_reward_volume_set',
    'check_reward_volumes',
    'check_stimOn_goCue_delays',
    'check_stimulus_move_before_goCue',
    'check_wheel_move_before_feedback',
    'check_wheel_freeze_during_quiescence'
)


_logger = logging.getLogger('ibllib')


class QcFrame(TaskQC):

    def __init__(self, session, bpod_only=False, local=False, one=None):
        """
        Loads and extracts the QC data for a given session path
        :param session: A str or Path to a session, or a session eid
        :param bpod_only: When True all data is extracted from Bpod instead of FPGA for ephys
        """
        if not isinstance(session, TaskQC):
            one = one or ONE()
            super().__init__(session, one=one, log=_logger)

            if local:
                dsets, out_files = ephys_fpga.extract_all(session, save=True)
                self.extractor = TaskQCExtractor(session, lazy=True, one=one)
                # Extract extra datasets required for QC
                self.extractor.data = dsets
                self.extractor.extract_data()
                # Aggregate and update Alyx QC fields
                self.run(update=False)
            else:
                self.load_data(bpod_only=bpod_only)
                self.compute()
        else:
            assert session.extractor and session.metrics, 'Please run QC before passing to QcFrame'
            super().__init__(session.eid or session.session_path, one=session.one, log=session.log)
            for attr in ('criteria', 'criteria', '_outcome',
                         'extractor', 'metrics', 'passed', 'fcns_value2status'):
                setattr(self, attr, getattr(session, attr))
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

        print('The following *critical* checks did not pass:')
        critical_checks = [f'_{x.replace("check", "task")}' for x in CRITICAL_CHECKS]
        for k, v in outcomes.items():
            if v != 'PASS' and k in critical_checks:
                print(k[6:])

        # Make DataFrame from the trail level metrics
        def get_trial_level_failed(d):
            new_dict = {k[6:]: v for k, v in d.items() if
                        isinstance(v, Sized) and len(v) == self.n_trials}
            return pd.DataFrame.from_dict(new_dict)

        self.frame = get_trial_level_failed(self.metrics)
        self.frame['intervals_0'] = self.extractor.data['intervals'][:, 0]
        self.frame['intervals_1'] = self.extractor.data['intervals'][:, 1]
        self.frame.insert(loc=0, column='trial_no', value=self.frame.index)

    def create_plots(self, axes,
                     wheel_axes=None, trial_events=None, color_map=None, linestyle=None):
        """
        Plots the data for bnc1 (sound) and bnc2 (frame2ttl)
        :param axes: An axes handle on which to plot the TTL events
        :param wheel_axes: An axes handle on which to plot the wheel trace
        :param trial_events: A list of Bpod trial events to plot, e.g. ['stimFreeze_times'],
        if None, valve, sound and stimulus events are plotted
        :param color_map: A color map to use for the events, default is the tableau color map
        linestyle: A line style map to use for the events, default is random.
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
            'ymax': 4,
            'linewidth': 2,
            'ax': axes
        }

        bnc1 = self.extractor.frame_ttls
        bnc2 = self.extractor.audio_ttls
        trial_data = self.extractor.data

        plots.squares(bnc1['times'], bnc1['polarities'] * 0.4 + 1, ax=axes, color='k')
        plots.squares(bnc2['times'], bnc2['polarities'] * 0.4 + 2, ax=axes, color='k')
        linestyle = linestyle or random.choices(('-', '--', '-.', ':'), k=len(trial_events))

        if self.extractor.bpod_ttls is not None:
            bpttls = self.extractor.bpod_ttls
            plots.squares(bpttls['times'], bpttls['polarities'] * 0.4 + 3, ax=axes, color='k')
            plot_args['ymax'] = 4
            ylabels = ['', 'frame2ttl', 'sound', 'bpod', '']
        else:
            plot_args['ymax'] = 3
            ylabels = ['', 'frame2ttl', 'sound', '']

        for event, c, l in zip(trial_events, cycle(color_map), linestyle):
            plots.vertical_lines(trial_data[event], label=event, color=c, linestyle=l, **plot_args)

        axes.legend(loc='upper left', fontsize='xx-small', bbox_to_anchor=(1, 0.5))
        axes.set_yticklabels(ylabels)
        axes.set_yticks(list(range(plot_args['ymax'] + 1)))
        axes.set_ylim([0, plot_args['ymax']])

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


def show_session_task_qc(qc_or_session=None, bpod_only=False, local=False, one=None):
    """
    Displays the task QC for a given session
    :param qc_or_session: session_path or TaskQC object
    :param bpod_only: (no FPGA)
    :param local: set True for local extraction
    :return: The QC object
    """
    if isinstance(qc_or_session, QcFrame):
        qc = qc_or_session
    elif isinstance(qc_or_session, TaskQC):
        qc = QcFrame(qc_or_session, one=one)
    else:
        qc = QcFrame(qc_or_session, bpod_only=bpod_only, local=local, one=one)
    # Run QC and plot
    w = ViewEphysQC.viewqc(wheel=qc.wheel_data)
    qc.create_plots(w.wplot.canvas.ax,
                    wheel_axes=w.wplot.canvas.ax2,
                    trial_events=EVENT_MAP.keys(),
                    color_map=cm,
                    linestyle=ls)
    # Update table and callbacks
    w.update_df(qc.frame)
    qt.run_app()
    return qc


if __name__ == "__main__":
    """Run TaskQC viewer with wheel data
    For information on the QC checks see the QC Flags & failures document:
    https://docs.google.com/document/d/1X-ypFEIxqwX6lU9pig4V_zrcR5lITpd8UJQWzW9I9zI/edit#
    ipython task_qc.py c9fec76e-7a20-4da4-93ad-04510a89473b
    ipython task_qc.py ./KS022/2019-12-10/001 --local
    """
    # Parse parameters
    parser = argparse.ArgumentParser(description='Quick viewer to see the behaviour data from'
                                                 'choice world sessions.')
    parser.add_argument('session', help='session uuid')
    parser.add_argument('--bpod', action='store_true', help='run QC on Bpod data only (no FPGA)')
    parser.add_argument('--local', action='store_true', help='run from disk location (lab server')
    args = parser.parse_args()  # returns data from the options specified (echo)

    show_session_task_qc(qc_or_session=args.session, bpod_only=args.bpod, local=args.local)
