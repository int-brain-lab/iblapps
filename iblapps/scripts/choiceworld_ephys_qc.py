from pathlib import Path
import shutil
import logging
import argparse

import numpy as np
import matplotlib.pyplot as plt

from oneibl.one import ONE
import alf.io
from ibllib.ephys import ephysqc
from ibllib.io.extractors import ephys_fpga, training_wheel, ephys_trials
import ibllib.io.raw_data_loaders as rawio

import iblapps.ViewEphysQC as ViewEphysQC
import iblapps.qt as qt

_logger = logging.getLogger('ibllib')

one = ONE()

dtypes_search = [
    '_spikeglx_sync.channels',
    '_spikeglx_sync.polarities',
    '_spikeglx_sync.times',
    '_iblrig_taskSettings.raw',
    '_iblrig_taskData.raw',
]

dtypes = [
    '_spikeglx_sync.channels',
    '_spikeglx_sync.polarities',
    '_spikeglx_sync.times',
    '_iblrig_taskSettings.raw',
    '_iblrig_taskData.raw',
    '_iblrig_encoderEvents.raw',
    '_iblrig_encoderPositions.raw',
    '_iblrig_encoderTrialInfo.raw',
    '_iblrig_Camera.timestamps',
    'ephysData.raw.meta',
    'ephysData.raw.wiring',
]


def _qc_from_path(sess_path, display=True):
    WHEEL = False
    sess_path = Path(sess_path)
    temp_alf_folder = sess_path.joinpath('fpga_test', 'alf')
    temp_alf_folder.mkdir(parents=True, exist_ok=True)

    raw_trials = rawio.load_data(sess_path)
    tmax = raw_trials[-1]['behavior_data']['States timestamps']['exit_state'][0][-1] + 60

    sync, chmap = ephys_fpga._get_main_probe_sync(sess_path, bin_exists=False)
    _ = ephys_trials.extract_all(sess_path, output_path=temp_alf_folder, save=True)
    # check that the output is complete
    fpga_trials = ephys_fpga.extract_behaviour_sync(sync, output_path=temp_alf_folder, tmax=tmax,
                                                    chmap=chmap, save=True, display=display)
    # align with the bpod
    bpod2fpga = ephys_fpga.align_with_bpod(temp_alf_folder.parent)
    alf_trials = alf.io.load_object(temp_alf_folder, '_ibl_trials')
    shutil.rmtree(temp_alf_folder)
    # do the QC
    qcs, qct = ephysqc.qc_fpga_task(fpga_trials, alf_trials)

    # do the wheel part
    if WHEEL:
        bpod_wheel = training_wheel.get_wheel_data(sess_path, save=False)
        fpga_wheel = ephys_fpga.extract_wheel_sync(sync, chmap=chmap, save=False)

        if display:
            t0 = max(np.min(bpod2fpga(bpod_wheel['re_ts'])), np.min(fpga_wheel['re_ts']))
            dy = np.interp(t0, fpga_wheel['re_ts'], fpga_wheel['re_pos']) - np.interp(
                t0, bpod2fpga(bpod_wheel['re_ts']), bpod_wheel['re_pos'])

            fix, axes = plt.subplots(nrows=2, sharex='all', sharey='all')
            # axes[0].plot(t, pos), axes[0].title.set_text('Extracted')
            axes[0].plot(bpod2fpga(bpod_wheel['re_ts']), bpod_wheel['re_pos'] + dy)
            axes[0].plot(fpga_wheel['re_ts'], fpga_wheel['re_pos'])
            axes[0].title.set_text('FPGA')
            axes[1].plot(bpod2fpga(bpod_wheel['re_ts']), bpod_wheel['re_pos'] + dy)
            axes[1].title.set_text('Bpod')

    return alf.io.dataframe({**fpga_trials, **alf_trials, **qct})


if __name__ == "__main__":
    # https://docs.google.com/document/d/1X-ypFEIxqwX6lU9pig4V_zrcR5lITpd8UJQWzW9I9zI/edit#
    parser = argparse.ArgumentParser(description='Quick viewer to see the behaviour data from'
                                                 'choice world ephys sessions.')
    parser.add_argument('session', help='session uuid')
    args = parser.parse_args()  # returns data from the options specified (echo)
    if alf.io.is_uuid_string(args.session):
        eid = args.session
        files = one.load(eid, dataset_types=dtypes, download_only=True)
        if not any(files):
            raise ValueError("Session doesn't seem to have any data")
        sess_path = alf.io.get_session_path(files[0])
        _logger.info(f"{eid} {sess_path}")
    elif alf.io.is_session_path(args.session):
        sess_path = Path(args.session)
        _logger.info(f"{sess_path}")

    w = ViewEphysQC.viewqc()
    qc_frame = _qc_from_path(sess_path, display=w.wplot.canvas.ax)
    w.update_df(qc_frame)
    # w = ViewEphysQC.viewqc(qc_frame)

    qt.run_app()
