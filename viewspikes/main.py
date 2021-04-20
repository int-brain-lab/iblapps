from pathlib import Path

import scipy.signal
import numpy as np

from ibllib.io import spikeglx
from ibllib.dsp import voltage
from ibllib.ephys import neuropixel
from oneibl.one import ONE
from easyqc.gui import viewseis

from viewspikes.plots import plot_insertion, show_psd, overlay_spikes
from viewspikes.data import stream, get_spikes, get_ks2

folder_samples = Path('/datadisk/Data/spike_sorting/short_samples')
files_samples = list(folder_samples.rglob('*.bin'))

one = ONE()
SIDE_BY_SIDE = False
#
# pins = one.alyx.rest('insertions', 'list', django=('json__extended_qc__alignment_count__gt,0'))
# pid, t0 = ('3e7618b8-34ca-4e48-ba3a-0e0f88a43131', 1002)  # SWC_054_2020-10-10_probe01__  - sync w/ spikes !!!
# pid, t0 = ('04c9890f-2276-4c20-854f-305ff5c9b6cf', 1002.)  # SWC_054_2020-10-10_probe00__04c9890f-2276-4c20-854f-305ff5c9b6cf  - sync w/ spikes !!!
# pid, t0 = ('0925fb1b-cf83-4f55-bfb7-aa52f993a404', 500.)  # DY_013_2020-03-06_probe00__0925fb1b-cf83-4f55-bfb7-aa52f993a404
# pid, t0 = ('0ece5c6a-7d1e-4365-893d-ac1cc04f1d7b', 750.)  # CSHL045_2020-02-27_probe01__0ece5c6a-7d1e-4365-893d-ac1cc04f1d7b
# pid, t0 = ('0ece5c6a-7d1e-4365-893d-ac1cc04f1d7b', 3000.)  # CSHL045_2020-02-27_probe01__0ece5c6a-7d1e-4365-893d-ac1cc04f1d7b
pid, t0 = ('10ef1dcd-093c-4839-8f38-90a25edefb49', 2400.)
# pid, t0 = ('1a6a17cc-ba8c-4d79-bf20-cc897c9500dc', 5000)
# pid, t0 = ('2dd99c91-292f-44e3-bbf2-8cfa56015106', 2500)  # NYU-23_2020-10-14_probe01__2dd99c91-292f-44e3-bbf2-8cfa56015106
# pid, t0 = ('2dd99c91-292f-44e3-bbf2-8cfa56015106', 6000)  # NYU-23_2020-10-14_probe01__2dd99c91-292f-44e3-bbf2-8cfa56015106
# pid, t0 = ('30dfb8c6-9202-43fd-a92d-19fe68602b6f', 2400.)  # ibl_witten_27_2021-01-16_probe00__30dfb8c6-9202-43fd-a92d-19fe68602b6f
# pid, t0 = ('31dd223c-0c7c-48b5-a513-41feb4000133', 3000.)  # really good one : striping on not all channels
# pid, t0 = ('39b433d0-ec60-460f-8002-a393d81620a4', 2700.)  # ZFM-01577_2020-10-27_probe01 needs FDNAT
# pid, t0 = ('47da98a8-f282-4830-92c2-af0e1d4f00e2', 2700.)

# 67 frequency spike
# 458 /datadisk/Data/spike_sorting/short_samples/b45c8f3f-6361-41df-9bc1-9df98b3d30e6_01210.bin ERROR dans le chargement de la whitening matrix
# 433 /datadisk/Data/spike_sorting/short_samples/8d59da25-3a9c-44be-8b1a-e27cdd39ca34_04210.bin Cortex complètement silencieux.
# 531 /datadisk/Data/spike_sorting/short_samples/47be9ae4-290f-46ab-b047-952bc3a1a509_00010.bin Sympa pour le spike sorting, un bon example de trace pourrie à enlever avec FDNAT / Cadzow. Il y a du striping à la fin mais pas de souci pour KS2 ou pour le FK.
# 618 5b9ce60c-dcc9-4789-b2ff-29d873829fa5_03610.bin: gros cabossage plat laissé par le FK !! Tester un filtre K tout bête # spikes tous petits en comparaison. Le spike sorting a l'air décalé
# 681 /datadisk/Data/spike_sorting/short_samples/eab93ab0-26e3-4bd9-9c53-9f81c35172f4_02410.bin !! Spikes décalés. Superbe example de layering dans le cerveau avec 3 niveaux très clairement définis
# 739 /datadisk/Data/spike_sorting/short_samples/f03b61b4-6b13-479d-940f-d1608eb275cc_04210.bin: Autre example de layering ou les charactéristiques spectrales / spatiales sont très différentes. Spikes alignés
# 830 /datadisk/Data/spike_sorting/short_samples/b02c0ce6-2436-4fc0-9ea0-e7083a387d7e_03010.bin, très mauvaise qualité - spikes sont décalés ?!?



file_ind = np.random.randint(len(files_samples))
file_ind = 739 # very good quality spike sorting
print(file_ind, files_samples[file_ind])

pid, t0 = ('47da98a8-f282-4830-92c2-af0e1d4f00e2', 1425.)

pid = files_samples[file_ind]
# pid, t0 = ("01c6065e-eb3c-49ba-9c25-c1f17b18d529", 500)
if isinstance(pid, Path):
    file_sample = pid
    pid, t0 = file_sample.stem.split('_')
    t0 = float(t0)
    sr = spikeglx.Reader(file_sample)
    dsets = one.alyx.rest('datasets', 'list', probe_insertion=pid)
else:
    sr, dsets = stream(pid, t0, one=one, samples_folder=folder_samples)

#
plot_insertion(pid, one)


h = neuropixel.trace_header()
raw = sr[:, :-1].T

sos = scipy.signal.butter(3, 300 / sr.fs / 2, btype='highpass', output='sos')
butt = scipy.signal.sosfiltfilt(sos, raw)
# show_psd(butt, sr.fs)

fk_kwargs ={'dx': 1, 'vbounds': [0, 1e6], 'ntr_pad': 160, 'ntr_tap': 0, 'lagc': .01, 'btype': 'lowpass'}
destripe = voltage.destripe(raw, fs=sr.fs, fk_kwargs=fk_kwargs, tr_sel=np.arange(raw.shape[0]))
ks2 = get_ks2(raw, dsets, one)

# get the spikes corresponding to current chunk, here needs to go through samples for sync reasons
spikes, clusters, channels = get_spikes(dsets, one)

if SIDE_BY_SIDE:
    hhh = {k: np.tile(h[k], 3) for k in h}
    eqc_concat = viewseis(np.r_[butt, destripe, ks2], si=1 / sr.fs, h=hhh, t0=t0, title='concat')
    overlay_spikes(eqc_concat, spikes, clusters, channels)
else:
    eqc_butt = viewseis(butt.T, si=1 / sr.fs, h=h, t0=t0, title='butt', taxis=0)
    eqc_dest = viewseis(destripe.T, si=1 / sr.fs, h=h, t0=t0, title='destr', taxis=0)
    eqc_ks2 = viewseis(ks2.T, si=1 / sr.fs, h=h, t0=t0, title='ks2', taxis=0)
    overlay_spikes(eqc_butt, spikes, clusters, channels)
    overlay_spikes(eqc_dest, spikes, clusters, channels)
    overlay_spikes(eqc_ks2, spikes, clusters, channels)
