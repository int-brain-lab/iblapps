import numpy as np
import scipy
import matplotlib.pyplot as plt
from easyqc.gui import viewseis


from oneibl.one import ONE
from ibllib.ephys import neuropixel
from ibllib.dsp import voltage
from brainbox.plot import driftmap
import alf.io

from needles2 import run_needles2
from viewspikes.data import stream, get_ks2, get_spikes
from viewspikes.plots import plot_insertion, show_psd, overlay_spikes

one = ONE()

eids = ['56b57c38-2699-4091-90a8-aba35103155e',
       '746d1902-fa59-4cab-b0aa-013be36060d5',
       '7b26ce84-07f9-43d1-957f-bc72aeb730a3',
       'dac3a4c1-b666-4de0-87e8-8c514483cacf',
       '6f09ba7e-e3ce-44b0-932b-c003fb44fb89',
       '73918ae1-e4fd-4c18-b132-00cb555b1ad2',
       'f312aaec-3b6f-44b3-86b4-3a0c119c0438',
       'dda5fc59-f09a-4256-9fb5-66c67667a466',
       'ee40aece-cffd-4edb-a4b6-155f158c666a',
       'ecb5520d-1358-434c-95ec-93687ecd1396',
       '54238fd6-d2d0-4408-b1a9-d19d24fd29ce',
       'e535fb62-e245-4a48-b119-88ce62a6fe67',
       'b03fbc44-3d8e-4a6c-8a50-5ea3498568e0',
       'db4df448-e449-4a6f-a0e7-288711e7a75a',
       '064a7252-8e10-4ad6-b3fd-7a88a2db5463',
       '41872d7f-75cb-4445-bb1a-132b354c44f0',
       'dfd8e7df-dc51-4589-b6ca-7baccfeb94b4',
       '4a45c8ba-db6f-4f11-9403-56e06a33dfa4',
       '4b00df29-3769-43be-bb40-128b1cba6d35',
       '862ade13-53cd-4221-a3fa-dda8643641f2',
       '3638d102-e8b6-4230-8742-e548cd87a949',
       'c7248e09-8c0d-40f2-9eb4-700a8973d8c8',
       'aad23144-0e52-4eac-80c5-c4ee2decb198',
       'd0ea3148-948d-4817-94f8-dcaf2342bbbe',
       '7f6b86f9-879a-4ea2-8531-294a221af5d0',
       'd23a44ef-1402-4ed7-97f5-47e9a7a504d9']

insertions = one.alyx.rest('insertions', 'list', django=f'session__in,{eids}')

## Example 1: Stream one second of ephys data
# pid, t0 = ("e864fca7-40e3-4a80-b736-51d4662405e4", 2155)
# pid, t0 = ('ce24bbe9-ae70-4659-9e9c-564d1a865de8', 610)

pid, t0 = (insertions[10]['id'], 2500)


sr, dsets = stream(pid, t0=t0, one=one, cache=True)
raw = sr[:, :-1].T

## Example: Plot Insertion for a given PID (todo: use Needles 2 for interactive)
av = run_needles2.view(lazy=True)
av.add_insertion_by_id(pid)

## Example: Display the raw / pre-proc and KS2 parts -
h = neuropixel.trace_header()
sos = scipy.signal.butter(3, 300 / sr.fs / 2, btype='highpass', output='sos')
butt = scipy.signal.sosfiltfilt(sos, raw)
fk_kwargs ={'dx': 1, 'vbounds': [0, 1e6], 'ntr_pad': 160, 'ntr_tap': 0, 'lagc': .01, 'btype': 'lowpass'}
destripe = voltage.destripe(raw, fs=sr.fs, fk_kwargs=fk_kwargs, tr_sel=np.arange(raw.shape[0]))
ks2 = get_ks2(raw, dsets, one)
eqc_butt = viewseis(butt.T, si=1 / sr.fs, h=h, t0=t0, title='butt', taxis=0)
eqc_dest = viewseis(destripe.T, si=1 / sr.fs, h=h, t0=t0, title='destr', taxis=0)
eqc_ks2 = viewseis(ks2.T, si=1 / sr.fs, h=h, t0=t0, title='ks2', taxis=0)


## Example: overlay the spikes on the existing easyqc instances
spikes, clusters, channels = get_spikes(dsets, one)
_, tspi, xspi = overlay_spikes(eqc_butt, spikes, clusters, channels)
overlay_spikes(eqc_dest, spikes, clusters, channels)
overlay_spikes(eqc_ks2, spikes, clusters, channels)

## Get the behaviour information
eid = dsets[0]['session'][-36:]
tdsets = one.alyx.rest('datasets', 'list', session=eid, django='name__icontains,trials.')
one.download_datasets(tdsets)
trials = alf.io.load_object(one.path_from_eid(eid).joinpath('alf'), 'trials')
rewards = trials['feedback_times'][trials['feedbackType'] == 1]

## Do the drift map with some task information overlaid
fig, ax = plt.subplots()
driftmap(spikes['times'], spikes['depths'], t_bin=0.1, d_bin=5, ax=ax)
from ibllib.plots import vertical_lines
vertical_lines(rewards, ymin=0, ymax=3800, ax=ax)

