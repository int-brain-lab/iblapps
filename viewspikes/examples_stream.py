from oneibl.one import ONE
from atlaselectrophysiology.alignment_with_easyqc import viewer

one = ONE()


"da8dfec1-d265-44e8-84ce-6ae9c109b8bd",  # SWC_043_2020-09-21_probe00 ok
"b749446c-18e3-4987-820a-50649ab0f826",  # KS023_2019-12-10_probe01  ok
"f86e9571-63ff-4116-9c40-aa44d57d2da9",  # CSHL049_2020-01-08_probe00 a bit stripy but fine
"675952a4-e8b3-4e82-a179-cc970d5a8b01",  # CSH_ZAD_029_2020-09-19_probe01 a bit stripy as well

pid = "af2a0072-e17e-4368-b80b-1359bf6d4647"
av = viewer(pid, one=one)
