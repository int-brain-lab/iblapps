## How to make DLC-labeled video
With the script DLC_labeled_video.py one can make DLC-labeled videos. The script downloads a specific IBL video ('body', 'left' or 'right') for some session (eid). It also downloads the wheel info and will print the wheel angle onto each frame.
To use it, start an ipython session, then type `run /path/to/DLC_labeled_video.py` to load the script. Next type 

`Viewer(eid, video_type, trial_range, save_video=True, eye_zoom=False)`

This will display and save a labeled video for a particular trial range. E.g. `Viewer('3663d82b-f197-4e8b-b299-7b803a155b84', 'left', [5,7])` will display and save a DLC labeled video for the left cam video of session with eid '3663d82b-f197-4e8b-b299-7b803a155b84' and range from trial 5 to trail 7. There's further the option to show a zoom of the pupil only. 

See `Example_DLC_access.ipynb` for more intructions and an example how to load DLC results.

