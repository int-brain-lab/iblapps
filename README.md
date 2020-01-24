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

What to look for:

[image + def of columns]


Close one of the GUI window to exit.