import numpy as np
import alf.io
from oneibl.one import ONE
from pathlib import Path
import cv2
import os
from oneibl.webclient import http_download_file_list
import matplotlib
import pandas as pd
# conda install -c conda-forge pyarrow

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def download_raw_video(eid, cameras=None):
    """
    Downloads the raw video from FlatIron or cache dir.
    This allows you to download just one of the
    three videos
    :param cameras: the specific camera to load
    (i.e. 'left', 'right', or 'body') If None all
    three videos are downloaded.
    :return: the file path(s) of the raw videos
    """
    one = ONE()
    if cameras:
        cameras = [cameras] if isinstance(cameras, str) else cameras
        cam_files = ['_iblrig_{}Camera.raw.mp4'.format(cam) for cam in cameras]
        datasets = one._alyxClient.get(
            'sessions/' + eid)['data_dataset_session_related']
        urls = [ds['data_url'] for ds in datasets if ds['name'] in cam_files]
        cache_dir = one.path_from_eid(eid).joinpath('raw_video_data')
        if not os.path.exists(str(cache_dir)):
            os.mkdir(str(cache_dir))
        else:  # Check if file already downloaded
            #cam_files = [fi[:-4] for fi in cam_files]  # Remove ext
            filenames = [f for f in os.listdir(str(cache_dir))
                         if any([cam in f for cam in cam_files])]
            if filenames:
                return [cache_dir.joinpath(file) for file in filenames]

        http_download_file_list(
            urls,
            username=one._par.HTTP_DATA_SERVER_LOGIN,
            password=one._par.HTTP_DATA_SERVER_PWD,
            cache_dir=str(cache_dir))        
        
        return 

    else:
        return one.load(eid, ['_iblrig_Camera.raw'], download_only=True)


def Viewer(eid, video_type, trial_range, save_video=True, eye_zoom=False):
    '''
    eid: session id, e.g. '3663d82b-f197-4e8b-b299-7b803a155b84'
    video_type: one of 'left', 'right', 'body'
    trial_range: first and last trial number of range to be shown, e.g. [5,7]
    save_video: video is displayed and saved in local folder

    Example usage to view and save labeled video with wheel angle:
    Viewer('3663d82b-f197-4e8b-b299-7b803a155b84', 'left', [5,7])
    3D example: 'cb2ad999-a6cb-42ff-bf71-1774c57e5308', [5,7]
    '''
 
    save_vids_here = '/home/mic/'
    if save_vids_here[-1] != '/':
        return 'Last character of save_vids_here must be slash' 
 
    one = ONE()
    dataset_types = ['camera.times',
                     'wheel.position',
                     'wheel.timestamps',
                     'trials.intervals',
                     'camera.dlc']

    a = one.list(eid, 'dataset-types')

    assert all([i in a for i in dataset_types]
               ), 'For this eid, not all data available'

    D = one.load(eid, dataset_types=dataset_types, dclass_output=True)
    alf_path = Path(D.local_path[0]).parent.parent / 'alf'

    # Download a single video
    video_data = alf_path.parent / 'raw_video_data'     
    download_raw_video(eid, cameras=[video_type])
    video_path = list(video_data.rglob('_iblrig_%sCamera.raw.*' % video_type))[0] 
    print(video_path) 

    # that gives cam time stamps and DLC output (change to alf_path eventually)
    
    cam = alf.io.load_object(alf_path, '%sCamera' % video_type, namespace = 'ibl')     


    # just to read in times for newer data (which has DLC results in pqt format
    #cam = alf.io.load_object(alf_path, '_ibl_%sCamera' % video_type)

    # set where to read and save video and get video info
    cap = cv2.VideoCapture(video_path.as_uri())
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(3)), int(cap.get(4)))

    assert length < len(cam['times']), '#frames > #stamps'
    print(eid,', ', video_type,', fsp:', fps,', #frames:', length, ', #stamps:', len(cam['times']), ', #frames - #stamps = ', length - len(cam['times'])) 


    # pick trial range for which to display stuff
    trials = alf.io.load_object(alf_path, 'trials', namespace = 'ibl')
    num_trials = len(trials['intervals'])
    if trial_range[-1] > num_trials - 1:
        print('There are only %s trials' % num_trials)

    frame_start = find_nearest(cam['times'],
                               [trials['intervals'][trial_range[0]][0]])
    frame_stop = find_nearest(cam['times'],
                              [trials['intervals'][trial_range[-1]][1]])

    '''
    wheel related stuff
    '''

    wheel = alf.io.load_object(alf_path, 'wheel', namespace ='ibl')
    import brainbox.behavior.wheel as wh
    try:
        pos, t = wh.interpolate_position(
            wheel['timestamps'], wheel['position'], freq=1000)
    except:
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
    for wt in cam['times'][frame_start:frame_stop]:
        wheel_pos.append(pos_int[find_nearest(t_int, wt)])
        kk += 1
        if kk % 3000 == 0:
            print('iteration', kk)

    '''
    DLC related stuff
    '''
    Times = cam['times'][frame_start:frame_stop] 
    del cam['times']      
    
    # some exception for inconsisitent data formats
    try:
        dlc_name = '_ibl_%sCamera.dlc.pqt' % video_type
        dlc_path = alf_path / dlc_name
        cam = pd.read_parquet(dlc_path ,engine = "fastparquet")    
    except:
        raw_vid_path = alf_path.parent / 'raw_video_data'
        cam = alf.io.load_object(raw_vid_path, '%sCamera' % video_type, namespace = 'ibl')  


    points = np.unique(['_'.join(x.split('_')[:-1]) for x in cam.keys()])
    if len(points)==1:
        cam = cam['dlc']
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
            cam[point + '_likelihood'].to_numpy() < 0.9, cam[point + '_x'].to_numpy())
        x = x.filled(np.nan)
        y = np.ma.masked_where(
            cam[point + '_likelihood'].to_numpy() < 0.9, cam[point + '_y'].to_numpy())
        y = y.filled(np.nan)
        XYs[point] = np.array(
            [x[frame_start:frame_stop], y[frame_start:frame_stop]])

    # Just for 3D testing
    #return XYs
    
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
        out = cv2.VideoWriter(save_vids_here + '%s_trials_%s_%s_%s.mp4' % (eid,
                                                          trial_range[0],
                                                          trial_range[-1],
                                                          video_type),
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
    cmap = matplotlib.cm.get_cmap('Spectral')
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
                    fontScale/2,
                    fontColor,
                    lineType)


        a,b = bottomLeftCornerOfText
        bottomLeftCornerOfText0 = (int(a*10 + b/2), b)
        cv2.putText(gray, 
                    '  time: ' + str(Time),
                    bottomLeftCornerOfText0,
                    font,
                    fontScale/2,
                    fontColor,
                    lineType)

            
        # print DLC dots
        ll = 0
        for point in points:
               
            # Put point color legend
            fontColor = (np.array([cmap(CR[ll])]) * 255)[0][:3]
            a ,b = bottomLeftCornerOfText
            if video_type == 'right':
                bottomLeftCornerOfText2 = (a, a * 2*(1 + ll))
            else: 
                bottomLeftCornerOfText2 = (b, a * 2*(1 + ll))
            fontScale2 = fontScale/4
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
                #col = np.array([0, 0, 255]) # all points red
                X = X.astype(int)
                Y = Y.astype(int)
                gray[X - dot_s:X + dot_s, Y - dot_s:Y + dot_s] = block * col
            ll += 1

        gray = gray[y0:y1, x0:x1]
        if save_video:
            out.write(gray)
        cv2.imshow('frame', gray)
        cv2.waitKey(1)
        k += 1
        if k == (frame_stop - frame_start) - 1:
            break

    if save_video:
        out.release()
    cap.release()
    cv2.destroyAllWindows()
