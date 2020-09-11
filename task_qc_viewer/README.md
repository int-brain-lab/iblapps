# iblapps
pyqt5 dependent applications for IBL sessions

## Task QC Viewer
This will download the TTL pulses and data collected on Bpod and/or FPGA and plot the results
alongside an interactive table.
The UUID is the session id. 

### Setup
Needs ibllib and ONE installed properly. Follow this guide for setup: https://github.com/int-brain-lab/iblenv#iblenv

NB: Only ibllib is required

If on the server PC, activate the environment by typing:
```
iblscripts
```
Otherwise, activate the iblenv as described in the guide above.

Go into the iblapps directory that you cloned:
```
cd /home/olivier/Documents/PYTHON/iblapps/task_qc_viewer
```
Launch the Viewer by typing `ipython task_qc.py session_UUID` , example:
```
ipython task_qc.py c9fec76e-7a20-4da4-93ad-04510a89473b
```
If you encouter the error `ModuleNotFoundError: No module named 'PyQt5'`, write in (conda) terminal (with iblenv activated):
```
pip install pyqt5
```

### Plots
1) Sync pulse display:
- TTL sync pulses (as recorded on the Bpod or FPGA for ephys sessions) for some key apparatus (i
.e. frame2TTL, audio signal). TTL pulse trains are displayed in black (time on x-axis, voltage on y-axis), offset by an increment of 1 each time (e.g. audio signal is on line 3, cf legend).
- trial event types, vertical lines (marked in different colours)

2) Wheel display:
- the wheel position in radians
- trial event types, vertical lines (marked in different colours)

3) Interactive table:
Each row is a trial entry.  Each column is a trial event

When double-clicking on any field of that table, the Sync pulse display time (x-) axis is adjusted so as to visualise the corresponding trial selected.

### What to look for
Tests are defined in the SINGLE METRICS section of ibllib/qc/task_metrics.py: https://github.com/int-brain-lab/ibllib/blob/master/ibllib/qc/task_metrics.py#L148-L149

### Exit
Close the GUI window containing the interactive table to exit.
