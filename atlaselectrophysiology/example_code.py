from one.api import ONE
from atlaselectrophysiology.alignment_with_easyqc import viewer

one = ONE()
probe_id = 'ce397420-3cd2-4a55-8fd1-5e28321981f4'


av = viewer(probe_id, one=one)


# To add trials to the window