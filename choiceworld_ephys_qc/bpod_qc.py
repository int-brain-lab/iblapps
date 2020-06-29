import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from oneibl.one import ONE
import ibllib.plots as plots
import alf.io
import logging
import argparse
import qt as qt
from ibllib.qc.bpodqc_metrics import BpodQC
import ViewEphysQC as ViewEphysQC
one = ONE()

_logger = logging.getLogger('ibllib')

dtypes = [
    "_iblrig_taskData.raw",
    "_iblrig_taskSettings.raw",
    "_iblrig_encoderPositions.raw",
    "_iblrig_encoderEvents.raw",
    "_iblrig_stimPositionScreen.raw",
    "_iblrig_syncSquareUpdate.raw",
    "_iblrig_encoderTrialInfo.raw",
    "_iblrig_ambientSensorData.raw",
]

class QcFromPath(object):
    def __init__(self, session_path):

        data = BpodQC(session_path, one=one)
        self.bnc1 = data.extractor.BNC1
        self.bnc2 = data.extractor.BNC2
        self.wheel = data.extractor.wheel_data
        self.trial_data = data.extractor.trial_data

        metrics = pd.DataFrame(data.metrics)
        passed = pd.DataFrame(data.passed)
        passed = passed.add_suffix('_passed')
        new_col = np.empty((metrics.columns.size + passed.columns.size,), dtype=object)
        new_col[0::2], new_col[1::2] = metrics.columns, passed.columns
        self.qc_frame = pd.concat([metrics, passed], axis=1)[new_col]
        self.qc_frame['intervals_0'] = self.trial_data['intervals_0']
        self.qc_frame['intervals_1'] = self.trial_data['intervals_1']

    def create_plots(self, display, wheel_display=None):
        # Plot data for bnc1 and bnc 2
        width = 0.5
        ymax = 3
        plots.squares(self.bnc1['times'], self.bnc1['polarities'] * 0.4 + 1,
                      ax=display, color='k')
        plots.squares(self.bnc2['times'], self.bnc2['polarities'] * 0.4 + 2,
                      ax=display, color='k')
        plots.vertical_lines(self.trial_data['goCue_times'], ymin=0, ymax=ymax,
                             ax=display, label='goCue_times', color='b', linewidth=width)
        plots.vertical_lines(self.trial_data['goCueTrigger_times'], ymin=0, ymax=ymax,
                             ax=display, label='goCueTrigger_times', color='m', linewidth=width)
        plots.vertical_lines(self.trial_data['errorCue_times'], ymin=0, ymax=ymax,
                             ax=display, label='errorCue_times', color='r', linewidth=width)
        plots.vertical_lines(self.trial_data['valveOpen_times'], ymin=0, ymax=ymax,
                             ax=display, label='valveOpen_times', color='g', linewidth=width)
        plots.vertical_lines(self.trial_data['stimFreeze_times'], ymin=0, ymax=ymax,
                             ax=display, label='stimFreeze_times', color='y', linewidth=width)
        plots.vertical_lines(self.trial_data['stimOff_times'], ymin=0, ymax=ymax,
                             ax=display, label='stimOff_times', color='c', linewidth=width)
        plots.vertical_lines(self.trial_data['stimOn_times'], ymin=0, ymax=ymax,
                             ax=display, label='stimOn_times', color='tab:orange', linewidth=width)
        display.legend()
        display.set_yticklabels(['', 'frame2ttl', 'sound', ''])
        display.set_yticks([0, 1, 2, 3])
        display.set_ylim([0, 3])

        if wheel_display:
            ymax = np.max(self.wheel['re_pos'])

            wheel_display.plot(self.wheel['re_ts'], self.wheel['re_pos'], color='k')
            plots.vertical_lines(self.trial_data['goCue_times'], ymin=0, ymax=ymax,
                                 ax=wheel_display, label='goCue_times', color='b', linewidth=width)
            plots.vertical_lines(self.trial_data['goCueTrigger_times'], ymin=0, ymax=ymax,
                                 ax=wheel_display, label='goCueTrigger_times', color='m', linewidth=width)
            plots.vertical_lines(self.trial_data['errorCue_times'], ymin=0, ymax=ymax,
                                 ax=wheel_display, label='errorCue_times', color='r', linewidth=width)
            plots.vertical_lines(self.trial_data['valveOpen_times'], ymin=0, ymax=ymax,
                                 ax=wheel_display, label='valveOpen_times', color='g',
                                 linewidth=width)
            plots.vertical_lines(self.trial_data['stimFreeze_times'], ymin=0, ymax=ymax,
                                 ax=wheel_display, label='stimFreeze_times', color='y',
                                 linewidth=width)
            plots.vertical_lines(self.trial_data['stimOff_times'], ymin=0, ymax=ymax,
                                 ax=wheel_display, label='stimOff_times', color='c', linewidth=width)
            plots.vertical_lines(self.trial_data['stimOn_times'], ymin=0, ymax=ymax,
                                 ax=wheel_display, label='stimOn_times', color='tab:orange',
                                 linewidth=width)


if __name__ == "__main__":
    # https://docs.google.com/document/d/1X-ypFEIxqwX6lU9pig4V_zrcR5lITpd8UJQWzW9I9zI/edit#
    parser = argparse.ArgumentParser(description='Quick viewer to see the behaviour data from'
                                                 'choice world sessions.')
    parser.add_argument('session', help='session uuid')
    args = parser.parse_args()  # returns data from the options specified (echo)
    if alf.io.is_uuid_string(args.session):
        eid = args.session
        files = one.load(eid, dataset_types=dtypes, download_only=True)
        files = [x for x in files if x]  # Remove None / empty entries
        if not any(files):
            raise ValueError("Session doesn't seem to have any data")
        sess_path = alf.io.get_session_path(files[0])
        _logger.info(f"{eid} {sess_path}")
    elif alf.io.is_session_path(args.session):
        sess_path = Path(args.session)
        _logger.info(f"{sess_path}")

    WHEEL = True
    qc = QcFromPath(sess_path)
    if WHEEL:
        w = ViewEphysQC.viewqc(wheel=qc.wheel)
        qc.create_plots(w.wplot.canvas.ax, wheel_display=w.wplot.canvas.ax2)
    else:
        w = ViewEphysQC.viewqc()
        qc.create_plots(w.wplot.canvas.ax)

    w.update_df(qc.qc_frame)
    qt.run_app()

