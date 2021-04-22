import numpy as np
import scipy
import matplotlib.pyplot as plt
from easyqc.gui import viewseis


from oneibl.one import ONE
from ibllib.ephys import neuropixel
from ibllib.dsp import voltage
from ibllib.plots import color_cycle
from brainbox.plot import driftmap

from iblapps.needles2 import run_needles2
from iblapps.viewspikes.data import stream, get_ks2, get_spikes
from iblapps.viewspikes.plots import plot_insertion, show_psd, overlay_spikes

one = ONE()

## Example 1: Stream one second of ephys data
pid, t0 = ('8413c5c6-b42b-4ec6-b751-881a54413628', 610)
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
ks2 = get_ks2(raw, dsets, one)
eqc_butt = viewseis(butt.T, si=1 / sr.fs, h=h, t0=t0, title='butt', taxis=0)
eqc_dest = viewseis(destripe.T, si=1 / sr.fs, h=h, t0=t0, title='destr', taxis=0)
eqc_ks2 = viewseis(ks2.T, si=1 / sr.fs, h=h, t0=t0, title='ks2', taxis=0)

# Example 5: overlay the spikes on the existing easyqc instances
spikes, clusters, channels = get_spikes(dsets, one)
overlay_spikes(eqc_butt, spikes, clusters, channels)
overlay_spikes(eqc_dest, spikes, clusters, channels)
overlay_spikes(eqc_ks2, spikes, clusters, channels)

# Do the driftmap
driftmap(spikes['times'], spikes['depths'], t_bin=0.1, d_bin=5, ax=axes[1])


# eqc_concat = viewseis(np.r_[butt, destripe, ks2], si=1 / sr.fs, h=hhh, t0=t0, title='concat')
# overlay_spikes(eqc_concat, spikes, clusters, channels)
from easyqc import qt
import datetime
qtapp = qt.create_app()
screenshot = qtapp.primaryScreen().grabWindow(eqc_butt.winId())

fn = datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
screenshot.save(f'/home/olivier/Pictures/{fn}.png', 'png')
