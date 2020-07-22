## How to make DLC-labeled video
With the script DLC_labeled_video.py one can make DLC-labeled videos. The script downloads a specific IBL video ('body', 'left' or 'right') for some session (eid). It also downloads the wheel info and will print the wheel angle onto each frame.
To use it, start an ipython session, then type `run /path/to/DLC_labeled_video.py` to load the script. Next type 

`Viewer(eid, video_type, trial_range, save_video=True, eye_zoom=False)`

This will display and save a labeled video for a particular trial range. E.g. `Viewer('3663d82b-f197-4e8b-b299-7b803a155b84', 'left', [5,7])` will display and save a DLC labeled video for the left cam video of session with eid '3663d82b-f197-4e8b-b299-7b803a155b84' and range from trial 5 to trial 7. There's further the option to show a zoom of the pupil only. 

See `Example_DLC_access.ipynb` for more intructions and an example how to load DLC results for other potential applications.

## Wheel and DLC live viewer
Plotting the wheel data along side the video and DLC can be done withe `wheel_dlc_viewer`.  The viewer loops over the 
video frames for a given trial and shows the wheel position plotted against time, indicating the
detected wheel movements.  __NB__: Besides the IBL enviroment dependencies, this module also
requires cv2.

### Running from command line
You can run the viewer from the terminal.  In Windows, running from the Anaconda prompt within
iblenv ensures the paths are correctly set (`conda activate iblenv`).

The below code will show the wheel data for a particular session, with the tongue and paw DLC
features overlaid:
  
```python
python wheel_dlc_viewer.py --eid 77224050-7848-4680-ad3c-109d3bcd562c --dlc tongue,paws
```
Pressing the space bar will toggle playing of the video and the left and right arrows allow you
to step frame by frame.  Key bindings also allow you to move between trials and toggle the legend.

When called without the `--eid` flag will cause a random session to be selected.  You can find a
full list of input arguments and key bindings by calling the function with the `--help` flag: 
```python
python wheel_dlc_viewer.py -h
```

### Running from within Python
You can also import the Viewer from within Python...

Example 1 - inspect trial 100 of a given session, looking at the right camera
```python
from wheel_dlc_viewer import Viewer
eid = '77224050-7848-4680-ad3c-109d3bcd562c'
v = Viewer(eid=eid, trial=100, camera='right')
```

Example 2 - pick a random session to inspect, showing all DLC
```python
from wheel_dlc_viewer import Viewer
Viewer(dlc_features='all')
 ```

For more details, see the docstring for the Viewer.

## Experiment reference strings
The included `exp_ref` module conveniently inter-converts experiment UUIDs ('eids') and human
readable experiment reference strings ('exp_refs').  Using exp_refs has the advantage of being
easily recognizable and sortable.  Lists of these exp_refs can be converted to session paths, eids
and DataJoint query restrictions (or _vice versa_):

**ref2eid**
```python
from exp_ref import *
from oneibl.one import ONE
base = 'https://test.alyx.internationalbrainlab.org'
one = ONE(username='test_user', password='TapetesBloc18', base_url=base)

# Lookup an eid from a reference dict 
ref = {'date': datetime(2018, 7, 13).date(), 'sequence': 1, 'subject': 'flowers'}
ref2eid(ref, one=one)
Out[0]: '4e0b3320-47b7-416e-b842-c34dc9004cf8'

# Convert a list of experiment reference strings to eids
ref2eid(['2018-07-13_1_flowers', '2019-04-11_1_KS005'], one=one)
Out[1]: ['4e0b3320-47b7-416e-b842-c34dc9004cf8',
         '7dc3c44b-225f-4083-be3d-07b8562885f4']
``` 

**ref2dj**
```python
# Fetch session information from DataJoint using an exp_ref
ref2dj('2020-06-20_2_CSHL046').fetch1()
Out[0]: {'subject_uuid': UUID('dffc24bc-bd97-4c2a-bef3-3e9320dc3dd7'),
         'session_start_time': datetime.datetime(2020, 6, 20, 13, 31, 47),
         'session_number': 2,
         'session_date': datetime.date(2020, 6, 20),
         'subject_nickname': 'CSHL046'}

# Restrict your query based on one or more exp_refs
from ibl_pipeline import acquisition, behavior
exp_refs = ['2020-06-20_2_CSHL046', '2019-11-01_1_ibl_witten_13']
query = behavior.TrialSet.Trial * (acquisition.Session & ref2dj(exp_refs))
```

**path2ref**
```python
from pathlib import Path

path_str = Path('E:/FlatIron/Subjects/zadorlab/flowers/2018-07-13/001')

# Return parsed dict from session path
path2ref(path_str)
Out[0]: {'subject': 'flowers', 'date': datetime.date(2018, 7, 13), 'sequence': 1}

# Returned dict of string elements from path
path2ref(path_str, parse=False)
Out[1]: {'subject': 'flowers', 'date': '2018-07-13', 'sequence': '001'}

# Returned exp_ref string from path
dict2ref(path2ref(path_str))
Out[1]: '2018-07-13_1_flowers'
```

For the full documentation and functions, run help on the module: `help(exp_ref)`. 