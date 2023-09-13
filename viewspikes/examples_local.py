from pathlib import Path
import numpy as np
import scipy
import matplotlib.pyplot as plt
from easyqc.gui import viewseis

import one.alf.io as alfio
from one.api import ONE
import spikeglx
import neuropixel
from neurodsp import voltage
from brainbox.plot import driftmap

from needles2 import run_needles2
from viewspikes.data import stream, get_ks2, get_spikes
from viewspikes.plots import plot_insertion, show_psd, overlay_spikes

RAW_PATH = Path("/datadisk/Data/spike_sorting/benchmark/raw")
SORT_PATH = Path("/datadisk/team_drives/WG-Neural-Analysis/Spike-Sorting-Analysis/benchmarks")
SORTERS = ['ks2','ks3', 'pyks2.5']

"8413c5c6-b42b-4ec6-b751-881a54413628",
"8ca1a850-26ef-42be-8b28-c2e2d12f06d6",
"ce24bbe9-ae70-4659-9e9c-564d1a865de8",
"ce397420-3cd2-4a55-8fd1-5e28321981f4",

# Example 1
pid, t0 = ("ce24bbe9-ae70-4659-9e9c-564d1a865de8", 810)
bin_file = next(RAW_PATH.joinpath(pid).rglob("*.ap.bin"))
sr = spikeglx.Reader(bin_file)
sel = slice(int(t0 * sr.fs), int((t0 + 4) * sr.fs))
raw = sr[sel, :-1].T


# Example 2: Plot Insertion for a given PID
av = run_needles2.view(lazy=True)
av.add_insertion_by_id(pid)

# Example 3: Show the PSD
fig, ax = plt.subplots()
fig.set_size_inches(14, 7)
show_psd(raw, sr.fs, ax=ax)

# Example 4: Display the raw / pre-proc
h = neuropixel.trace_header()
sos = scipy.signal.butter(3, 300 / sr.fs / 2, btype='highpass', output='sos')
butt = scipy.signal.sosfiltfilt(sos, raw)
fk_kwargs ={'dx': 1, 'vbounds': [0, 1e6], 'ntr_pad': 160, 'ntr_tap': 0, 'lagc': .01, 'btype': 'lowpass'}
destripe = voltage.destripe(raw, fs=sr.fs, fk_kwargs=fk_kwargs, tr_sel=np.arange(raw.shape[0]))
eqc_butt = viewseis(butt.T, si=1 / sr.fs, h=h, t0=t0, title='butt', taxis=0)
eqc_dest = viewseis(destripe.T, si=1 / sr.fs, h=h, t0=t0, title='destr', taxis=0)
eqc_dest_ = viewseis(destripe.T, si=1 / sr.fs, h=h, t0=t0, title='destr_', taxis=0)
# Example 5: overlay the spikes on the existing easyqc instances
from ibllib.plots import color_cycle
ss = {}
symbols = 'x+o'
eqcsort = {}
for i, sorter in enumerate(SORTERS):
    alf_path = SORT_PATH.joinpath(sorter, pid,'alf')
    ss[sorter] = {}
    for k in ['spikes', 'clusters', 'channels']:
        ss[sorter][k] = alfio.load_object(alf_path, k)
    col = (np.array(color_cycle(i)) * 255).astype(np.uint8)
    eqcsort[sorter] = viewseis(destripe.T, si=1 / sr.fs, h=h, t0=t0, title=sorter, taxis=0)
    _, _, _ = overlay_spikes(
        eqcsort[sorter], ss[sorter]['spikes'], ss[sorter]['clusters'], ss[sorter]['channels'],
        label=sorter, symbol=symbols[i])
    _, _, _ = overlay_spikes(
        eqc_dest_, ss[sorter]['spikes'], ss[sorter]['clusters'], ss[sorter]['channels'],
        rgb=tuple(col), label=sorter, symbol=symbols[i])
    # overlay_spikes(eqc_dest, ss[sorter]['spikes'], ss[sorter]['clusters'], ss[sorter]['channels'])
    # sc.setPen(pg.mkPen((0, 255, 0, 155), width=1))

##
from ibllib.dsp.fourier import fshift
from ibllib.dsp.voltage import destripe
eqc_butt = viewseis(butt.T, si=1 / sr.fs, h=h, t0=t0, title='butt', taxis=0)
bshift = fshift(butt, h['sample_shift'], axis=1)
eqc_buts = viewseis(bshift.T, si=1 / sr.fs, h=h, t0=t0, title='shift', taxis=0)



##
from one.api import ONE
pid = "8413c5c6-b42b-4ec6-b751-881a54413628"
one = ONE()

dtypes = ['spikes.amps', 'spikes.clusters', 'spikes.times',
 'clusters.channels',
 'clusters.mlapdv']

from iblatlas.atlas import atlas
from ibllib.pipes import histology
from ibllib.ephys import neuropixel

import numpy as np
neuropixel.TIP_SIZE_UM
neuropixel.SITES_COORDINATES

len(one.alyx.rest('datasets', 'list', insertion=pid))

# if we don't have the data on the flatiron
pi = one.alyx.rest('insertions', 'read', id=pid)
traj = one.alyx.rest('trajectories', 'list', probe_insertion=pid)[-1]
ins = atlas.Insertion.from_dict(traj)
xyz_channels = histology.interpolate_along_track(
    ins.xyz, (neuropixel.SITES_COORDINATES[:, 1] + neuropixel.TIP_SIZE_UM) / 1e6)




