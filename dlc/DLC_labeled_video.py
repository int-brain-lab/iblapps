import numpy as np
from one.api import ONE
from pathlib import Path
import cv2
import os,fnmatch
import matplotlib
import pandas as pd
# conda install -c conda-forge pyarrow
import os

def Find(pattern, path):

    '''
    find a local video like so:
    flatiron='/home/mic/Downloads/FlatIron'      
    vids = Find('*.mp4', flatiron)
    '''
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def Viewer(eid, video_type, trial_range, save_video=True, eye_zoom=False):
    '''
    eid: session id, e.g. '3663d82b-f197-4e8b-b299-7b803a155b84'
    video_type: one of 'left', 'right', 'body'
    trial_range: first and last trial number of range to be shown, e.g. [5,7]
    save_video: video is saved this local folder

    Example usage to view and save labeled video with wheel angle:
    Viewer('3663d82b-f197-4e8b-b299-7b803a155b84', 'left', [5,7])
    3D example: 'cb2ad999-a6cb-42ff-bf71-1774c57e5308', [5,7]
    '''

    save_vids_here = Path.home()

    one = ONE()
    alf_path = one.eid2path(eid)

    # Download a single video
    video_path = (alf_path / 
        f'raw_video_data/_iblrig_{video_type}Camera.raw.mp4')
    
    if not os.path.isfile(video_path):
        print('mp4 not found locally, downloading it ...')
        video_path = one.load_dataset(eid,
            f'raw_video_data/_iblrig_{video_type}Camera.raw.mp4',
            download_only=True)

    # Get trials info
    trials = one.load_object(eid, 'trials', download_only=True)
    

    # Download DLC traces and stamps
    Times = one.load_dataset(eid,f'alf/_ibl_{video_type}Camera.times.npy') 
    cam = one.load_dataset(eid,f'alf/_ibl_{video_type}Camera.dlc.pqt')
                                                      
    # get video info
    cap = cv2.VideoCapture(video_path.as_uri())
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(3)), int(cap.get(4)))


    print(eid,
          ', ',
          video_type,
          ', fsp:',
          fps,
          ', #frames:',
          length,
          ', #stamps:',
          len(Times),
          ', #frames - #stamps = ',
          length - len(Times))

    # pick trial range for which to display stuff
    trials = one.load_object(eid, 'trials')
    num_trials = len(trials['intervals'])
    if trial_range[-1] > num_trials - 1:
        print('There are only %s trials' % num_trials)
    print('There are %s trials' % num_trials)
    frame_start = find_nearest(Times,
                               [trials['intervals'][trial_range[0]][0]])
    frame_stop = find_nearest(Times,
                              [trials['intervals'][trial_range[-1]][1]])

    '''
    load wheel
    '''

    wheel = one.load_object(eid, 'wheel')
    import brainbox.behavior.wheel as wh
    try:
        pos, t = wh.interpolate_position(
            wheel['timestamps'], wheel['position'], freq=1000)
    except BaseException:
        pos, t = wh.interpolate_position(
            wheel['times'], wheel['position'], freq=1000)

    w_start = find_nearest(t, trials['intervals'][trial_range[0]][0])
    w_stop = find_nearest(t, trials['intervals'][trial_range[-1]][1])

    # confine to interval
    pos_int = pos[w_start:w_stop]
    t_int = t[w_start:w_stop]

    # alignment of cam stamps and interpolated wheel stamps
    wheel_pos = []
    kk = 0
    for wt in Times[frame_start:frame_stop]:
        wheel_pos.append(pos_int[find_nearest(t_int, wt)])
        kk += 1
        if kk % 3000 == 0:
            print('iteration', kk)

    '''
    DLC related stuff
    '''
    Times = Times[frame_start:frame_stop]

    # some exception for inconsisitent data formats

 
    
    points = np.unique(['_'.join(x.split('_')[:-1]) for x in cam.keys()])
    if len(points) == 1:
        points = np.unique(['_'.join(x.split('_')[:-1]) for x in cam.keys()])

    if video_type != 'body':
        d = list(points)
        d.remove('tube_top')
        d.remove('tube_bottom')
        points = np.array(d)

    # Set values to nan if likelyhood is too low # for pqt: .to_numpy()
    XYs = {}
    for point in points:
        x = np.ma.masked_where(
            cam[point + '_likelihood'] < 0.9, cam[point + '_x'])
        x = x.filled(np.nan)
        y = np.ma.masked_where(
            cam[point + '_likelihood'] < 0.9, cam[point + '_y'])
        y = y.filled(np.nan)
        XYs[point] = np.array(
            [x[frame_start:frame_stop], y[frame_start:frame_stop]])

    # Just for 3D testing
    # return XYs

    # Zoom at eye
    if eye_zoom:
        pivot = np.nanmean(XYs['pupil_top_r'], axis=1)
        x0 = int(pivot[0]) - 33
        x1 = int(pivot[0]) + 33
        y0 = int(pivot[1]) - 28
        y1 = int(pivot[1]) + 38
        size = (66, 66)
        dot_s = 1  # [px] for painting DLC dots

    else:
        x0 = 0
        x1 = size[0]
        y0 = 0
        y1 = size[1]
        if video_type == 'left':
            dot_s = 10  # [px] for painting DLC dots
        else:
            dot_s = 5

    if save_video:
        loc = (save_vids_here / 
        f'{eid}_trials_{trial_range[0]}_{trial_range[-1]}_{video_type}.mp4')

        out = cv2.VideoWriter(str(loc),
                              cv2.VideoWriter_fourcc(*'mp4v'),
                              fps,
                              size)  # put , 0 if grey scale

    # writing stuff on frames
    font = cv2.FONT_HERSHEY_SIMPLEX

    if video_type == 'left':
        bottomLeftCornerOfText = (20, 1000)
        fontScale = 4
    else:
        bottomLeftCornerOfText = (10, 500)
        fontScale = 2

    lineType = 2

    # assign a color to each DLC point (now: all points red)
    cmap = matplotlib.cm.get_cmap('Set1')
    CR = np.arange(len(points)) / len(points)

    block = np.ones((2 * dot_s, 2 * dot_s, 3))

    # set start frame
    cap.set(1, frame_start)

    k = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        gray = frame

        # print wheel angle
        fontColor = (255, 255, 255)
        Angle = round(wheel_pos[k], 2)
        Time = round(Times[k], 3)
        cv2.putText(gray,
                    'Wheel angle: ' + str(Angle),
                    bottomLeftCornerOfText,
                    font,
                    fontScale / 2,
                    fontColor,
                    lineType)

        a, b = bottomLeftCornerOfText
        bottomLeftCornerOfText0 = (int(a * 10 + b / 2), b)
        cv2.putText(gray,
                    '  time: ' + str(Time),
                    bottomLeftCornerOfText0,
                    font,
                    fontScale / 2,
                    fontColor,
                    lineType)

        # print DLC dots
        ll = 0
        for point in points:

            # Put point color legend
            fontColor = (np.array([cmap(CR[ll])]) * 255)[0][:3]
            a, b = bottomLeftCornerOfText
            if video_type == 'right':
                bottomLeftCornerOfText2 = (a, a * 2 * (1 + ll))
            else:
                bottomLeftCornerOfText2 = (b, a * 2 * (1 + ll))
            fontScale2 = fontScale / 4
            cv2.putText(gray, point,
                        bottomLeftCornerOfText2,
                        font,
                        fontScale2,
                        fontColor,
                        lineType)

            X0 = XYs[point][0][k]
            Y0 = XYs[point][1][k]
            # transform for opencv?
            X = Y0
            Y = X0

            if not np.isnan(X) and not np.isnan(Y):
                col = (np.array([cmap(CR[ll])]) * 255)[0][:3]
                # col = np.array([0, 0, 255]) # all points red
                X = X.astype(int)
                Y = Y.astype(int)
#                cv2.imshow('frame', gray)
#                print(gray.shape)
#                print(X - dot_s,X + dot_s, Y - dot_s,Y + dot_s)
                gray[X - dot_s:X + dot_s, Y - dot_s:Y + dot_s] = block * col
            ll += 1

        gray = gray[y0:y1, x0:x1]
        if save_video:
            out.write(gray)
        #cv2.imshow('frame', gray)
        #cv2.waitKey(1)
        k += 1
        if k == (frame_stop - frame_start) - 1:
            break

    if save_video:
        out.release()
    cap.release()
    #cv2.destroyAllWindows()
    
    print(eid, video_type, frame_stop, frame_start)
    #return XYs, Times
