from pathlib import Path
import numpy as np
import one.alf.io as alfio
import pandas as pd
from one.api import ONE
from brainbox.processing import bincount2D
one = ONE()

root_path = Path(r'C:\Users\Mayo\Downloads\FlatIron')
files = list(root_path.rglob('electrodeSites.mlapdv.npy'))

all_df_chns = []

for file in files:
    try:
        session_path = Path(*file.parts[:10])
        ref = one.path2ref(session_path)
        eid = one.path2eid(session_path)
        probe = file.parts[11]

        lfp = alfio.load_object(session_path.joinpath('raw_ephys_data', probe), 'ephysSpectralDensityLF', namespace='iblqc')
        mean_lfp = 10 * np.log10(np.mean(lfp.power[:,:-1], axis=0))

        try:
            ap = alfio.load_object(session_path.joinpath('raw_ephys_data', probe), 'ephysTimeRmsAP', namespace='iblqc')
            mean_ap = np.mean(ap.rms[:, :384], axis=0)
        except Exception as err:
            mean_ap = 50 * np.ones((384))

        try:
            spikes = alfio.load_object(session_path.joinpath(f'alf/{probe}/pykilosort'), 'spikes')
            kp_idx = ~np.isnan(spikes['depths'])
            T_BIN = np.max(spikes['times'])
            D_BIN = 10
            chn_min = 10
            chn_max = 3840
            nspikes, times, depths = bincount2D(spikes['times'][kp_idx],
                                                spikes['depths'][kp_idx],
                                                T_BIN, D_BIN,
                                                ylim=[chn_min, chn_max])

            amp, times, depths = bincount2D(spikes['amps'][kp_idx],
                                            spikes['depths'][kp_idx],
                                            T_BIN, D_BIN, ylim=[chn_min, chn_max],
                                            weights=spikes['amps'][kp_idx])
            mean_fr = nspikes[:, 0] / T_BIN
            mean_amp = np.divide(amp[:, 0], nspikes[:, 0]) * 1e6
            mean_amp[np.isnan(mean_amp)] = 0
            remove_bins = np.where(nspikes[:, 0] < 50)[0]
            mean_amp[remove_bins] = 0
        except Exception as err:
            mean_fr = np.ones((384))
            mean_amp = np.ones((384))
            depths = np.ones((384)) * 20



        channels = alfio.load_object(file.parent, 'electrodeSites')
        data_chns = {}
        data_chns['x'] = channels['mlapdv'][:, 0]
        data_chns['y'] = channels['mlapdv'][:, 1]
        data_chns['z'] = channels['mlapdv'][:, 2]
        data_chns['axial_um'] = channels['localCoordinates'][:, 0]
        data_chns['lateral_um'] = channels['localCoordinates'][:, 1]
        data_chns['lfp'] = mean_lfp
        data_chns['ap'] = mean_ap
        data_chns['depth_line'] = depths
        data_chns['fr'] = mean_fr
        data_chns['amp'] = mean_amp
        data_chns['region_id'] = channels['brainLocationIds_ccf_2017']

        df_chns = pd.DataFrame.from_dict(data_chns)
        df_chns['subject'] = ref['subject']
        df_chns['date'] = str(ref['date'])
        df_chns['probe'] = probe
        df_chns['pid'] = eid + probe


        all_df_chns.append(df_chns)
    except Exception as err:
        print(err)





