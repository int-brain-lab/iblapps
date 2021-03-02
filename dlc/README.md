## How to get example frames with DLC labeles painted on top
Avoiding to download lengthy videos, with DLC_labeled_video.py one can stream 5 example frames randomly picked across a whole video, specified by eid and video type ('body', 'left' or 'right'). 

To use it, start an ipython session, then type `run /path/to/DLC_labeled_video.py` to load the script. 

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
