
from oneibl.one import ONE

one = ONE()
session_number = 1
eid = one.search(subject='SWC_014', date='2019-12-12', number=session_number)[0]
data_path = one.load(eid, clobber=False, download_only=True)

# Uncomment to also load ephys.bin files for waveform plots
#ephys_types = ['ephysData.raw.ch', 'ephysData.raw.meta', 'ephysData.raw.ap']
#ephys_path = one.load(eid, dataset_types=ephys_types, clobber=False, download_only=True)
