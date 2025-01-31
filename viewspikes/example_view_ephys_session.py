from pathlib import Path
from one.api import ONE
from brainbox.io.one import EphysSessionLoader, SpikeSortingLoader
from iblapps.viewspikes.gui import view_raster

PATH_CACHE = Path("/datadisk/Data/NAWG/01_lick_artefacts/openalyx")

one = ONE(base_url="https://openalyx.internationalbrainlab.org", cache_dir=PATH_CACHE)

pid = '5135e93f-2f1f-4301-9532-b5ad62548c49'
eid, pname = one.pid2eid(pid)

# set stream to False if you have the file locally or want to download the full session
self = view_raster(pid=pid, one=one, stream=True)


