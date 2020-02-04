# iblapps
pyqt5 dependent applications for IBL sessions

## Ephys QC Viewer
This will download the sync pulses and behaviour raw data and plot the results alongside
an interactive table.
The UUID is the session id. 
Needs ibllib and ONE installed properly.

If on the server PC, activate the environment by:
```
iblscripts
```
Go into the iblapps directory that you cloned:
```
cd /home/olivier/Documents/PYTHON/iblapps
```
Launch the Viewer by typing `ipython choiceworld_ephys_qc.py session_UUID` , example:
```
ipython choiceworld_ephys_qc.py c9fec76e-7a20-4da4-93ad-04510a89473b
```

Close the GUI window containing the sheet to exit.

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
You can search in this code `ephysqc.py` for a specific test name (e.g. `stimOff_delay_valve` - which is a column in the GUI table) to find the corresponding explanation as comment (e.g. `# stimOff 1 sec after valve, with 0.1 as acceptable jitter`) and test implementation.

Generally speaking, look for FALSE output. More precisely:

If test containin the wording `_nan` is FALSE (e.g. `goCue_times_nan`), it means the corresponding TTL pulse is not detected within the trial.

If test containing the wording `_before_` or `_delay_` (e.g. `stim_freeze_before_feedback`) is FALSE, it means order / delay between events is not respected.

