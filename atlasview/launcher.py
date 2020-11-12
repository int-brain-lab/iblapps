from pathlib import Path
import numpy as np
from atlasview import atlasview  # mouais il va falloir changer ça
av = atlasview.view()  # need to have an output argument here or the garbage collector will clean
# it up and boom

""" Roadmap
    - swap volumes combox (label RGB option / density)
    - overlay brain regions with transparency
    - overlay volumes (for example coverage with transparency)
    - overlay plots: probes channels
    - tilted slices
    - coordinate swaps: add Allen / Needles / Voxel options
    - should we add horizontal slices ?
"""

# add brain regions feature:
reg_values = np.load(Path(atlasview.__file__).parent.joinpath('region_values.npy'))
av.add_regions_feature(reg_values, 'Blues', opacity=0.7)

# add scatter feature:
chans = np.load(
    Path(atlasview.__file__).parent.joinpath('channels_test.npy'), allow_pickle=True)
av.add_scatter_feature(chans)
