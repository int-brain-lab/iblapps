import numpy as np
import scipy
import matplotlib.pyplot as plt
from easyqc.gui import viewseis


from oneibl.one import ONE
from ibllib.ephys import neuropixel
from ibllib.dsp import voltage
from brainbox.plot import driftmap

from needles2 import run_needles2
from viewspikes.data import stream, get_ks2, get_spikes
from viewspikes.plots import plot_insertion, show_psd, overlay_spikes

one = ONE()


"da8dfec1-d265-44e8-84ce-6ae9c109b8bd",  # SWC_043_2020-09-21_probe00 ok
"b749446c-18e3-4987-820a-50649ab0f826",  # KS023_2019-12-10_probe01  ok
"f86e9571-63ff-4116-9c40-aa44d57d2da9",  # CSHL049_2020-01-08_probe00 a bit stripy but fine
"675952a4-e8b3-4e82-a179-cc970d5a8b01",  # CSH_ZAD_029_2020-09-19_probe01 a bit stripy as well

## Example 1: Stream one second of ephys data
pid, t0 = ("e864fca7-40e3-4a80-b736-51d4662405e4", 2155)
sr, dsets = stream(pid, t0=t0, one=one, cache=True)

## Example 2: Plot Insertion for a given PID (todo: use Needles 2 for interactive)
av = run_needles2.view(lazy=True)
av.add_insertion_by_id(pid)

## Example 3: Show the PSD
raw = sr[:, :-1].T
fig, axes = plt.subplots(1, 2)
fig.set_size_inches(18, 7)
show_psd(raw, sr.fs, ax=axes[0])

## Example 4: Display the raw / pre-proc and KS2 parts -
h = neuropixel.trace_header()
sos = scipy.signal.butter(3, 300 / sr.fs / 2, btype='highpass', output='sos')
butt = scipy.signal.sosfiltfilt(sos, raw)
fk_kwargs ={'dx': 1, 'vbounds': [0, 1e6], 'ntr_pad': 160, 'ntr_tap': 0, 'lagc': .01, 'btype': 'lowpass'}
destripe = voltage.destripe(raw, fs=sr.fs, fk_kwargs=fk_kwargs, tr_sel=np.arange(raw.shape[0]))
eqc_butt = viewseis(butt.T, si=1 / sr.fs, h=h, t0=t0, title='butt', taxis=0)
eqc_dest = viewseis(destripe.T, si=1 / sr.fs, h=h, t0=t0, title='destr', taxis=0)

# Example 5: overlay the spikes on the existing easyqc instances
spikes, clusters, channels = get_spikes(dsets, one)
_, tspi, xspi = overlay_spikes(eqc_butt, spikes, clusters, channels)
overlay_spikes(eqc_dest, spikes, clusters, channels)
overlay_spikes(eqc_ks2, spikes, clusters, channels)

# Do the driftmap
driftmap(spikes['times'], spikes['depths'], t_bin=0.1, d_bin=5, ax=axes[1])


##
import alf.io
eid = dsets[0]['session'][-36:]
tdsets = one.alyx.rest('datasets', 'list', session=eid, django='name__icontains,trials.')
one.download_datasets(tdsets)
trials = alf.io.load_object(one.path_from_eid(eid).joinpath('alf'), 'trials')

rewards = trials['feedback_times'][trials['feedbackType'] == 1]


##

rewards = trials['feedback_times'][trials['feedbackType'] == 1]


## do drift map
fig, ax = plt.subplots()
driftmap(spikes['times'], spikes['depths'], t_bin=0.1, d_bin=5, ax=ax)
from ibllib.plots import vertical_lines
vertical_lines(rewards, ymin=0, ymax=3800, ax=ax)

