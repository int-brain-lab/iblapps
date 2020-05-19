# iblapps
pyqt5 dependent applications for IBL sessions

## Ephys QC Viewer
This will download the sync pulses and behaviour raw data and plot the results alongside
an interactive table.
The UUID is the session id. 

### Setup
Needs ibllib and ONE installed properly. Follow this guide for setup: https://readthedocs.org/projects/ibllib/downloads/pdf/latest/ 

If on the server PC, activate the environment by typing:
```
iblscripts
```
Otherwise, activate the iblenv as described in the guide above.

Go into the iblapps directory that you cloned:
```
cd /home/olivier/Documents/PYTHON/iblapps
```
Launch the Viewer by typing `ipython choiceworld_ephys_qc.py session_UUID` , example:
```
ipython choiceworld_ephys_qc.py c9fec76e-7a20-4da4-93ad-04510a89473b
```

If you encouter the error `ModuleNotFoundError: No module named 'PyQt5'`, write in (conda) terminal (with iblenv activated):
```
pip install pyqt5
```

### Plots
1) Synch pulse display :
- TTL synch pulses (as recorded on the ephys recording system, FPGA or PXI) for some key apparatus (e.g. frame2TTL, audio signal). TTL pulse trains are displayed in black (time on x-axis, voltage on y-axis), offset by an increment of 1 each time (e.g. audio signal is on line 3, cf legend).
- event types, vertical lines (marked in different colours)

2) Interactive table:
Each row is a trial entry.
Column-wise, the table has two parts: 
- the left-hand rows indicate the values of timestamps at which TTL pulses are detected.
- the right-hand rows indicate the result of tests (indicated by either integer values, or TRUE/FALSE output).

When double-clicking on any field of that table, the Synch pulse display time (x-) axis is adjusted so as to visualise the corresponding trial selected.

### What to look for
Tests are defined here : https://github.com/int-brain-lab/ibllib/blob/90163a40eb970cf0282b651667dd8ba341ff2044/ibllib/ephys/ephysqc.py#L419

You can search in this code for a specific test name (e.g. `stimOff_delay_valve` - which is a column in the GUI table) to find the corresponding explanation and implementation (e.g. `# stimOff 1 sec after valve, with 0.1 as acceptable jitter`).

Generally speaking, look for FALSE output. More precisely:

If test containin the wording `_nan` is FALSE (e.g. `goCue_times_nan`), it means the corresponding TTL pulse is not detected within the trial.

If test containing the wording `_before_` or `_delay_` (e.g. `stim_freeze_before_feedback`) is FALSE, it means order / delay between events is not respected.

Some tests check for the number of output detected, in which case the output is not TRUE/FALSE but an integer. E.g. `n_feedback` should always be = 1 given our implementation. 

### Exit
Close the GUI window containing the interactive table to exit.

## How to make DLC-labeled video
With the script DLC_labeled_video.py one can make DLC-labeled videos. The script downloads a specific IBL video ('body', 'left' or 'right') for some session (eid). It also downloads the wheel info and will print the wheel angle onto each frame.
To use it, start an ipython session, then type `run /path/to/DLC_labeled_video.py` to load the script. Next type 

`Viewer(eid, video_type, trial_range, save_video=True, eye_zoom=False)`

This will display and save a labeled video for a particular trial range. E.g. `Viewer('3663d82b-f197-4e8b-b299-7b803a155b84', 'left', [5,7])` will display and save a DLC labeled video for the left cam video of session with eid '3663d82b-f197-4e8b-b299-7b803a155b84' and range from trial 5 to trail 7. There's further the option to show a zoom of the pupil only. 






