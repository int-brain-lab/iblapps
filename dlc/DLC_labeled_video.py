import numpy as np
from one.api import ONE
from pathlib import Path
import cv2
import os,fnmatch
import matplotlib
import pandas as pd
# conda install -c conda-forge pyarrow
import os
from ibldsp.smooth import smooth_interpolate_savgol
from brainbox.io.one import SessionLoader
from copy import deepcopy


#one = ONE(base_url='https://openalyx.internationalbrainlab.org',
#      password='international', silent=True)

one = ONE()

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


def load_lp(eid, cam, masked=True, paws=True,
            reso='128x102_128x128', flav='multi'):

    '''
    for a given session and cam, load all lp tracked points;
    that's paw specific now; 
    flav is either single or multi view EKS
    '''
    
    print(f'loading LP, {reso}, {cam}')
    print(f'{flav}, paws:{paws}, {eid}')
    
    if paws:
    
        pth = ('/mnt/8cfe1683-d974-40f3-a20b-b217cad4722a/lp_ens'
              f'/{reso}/{eid}/ensembles_{cam}Camera/'
              f'_iblrig_{cam}Camera.raw.paws.eks_{flav}.csv') 

        d0 = pd.read_csv(pth, low_memory=False)


        if reso[:7] == '128x102':
            scale = 10 if cam == 'left' else 5
        else:    
            scale = 4 if cam == 'left' else 2

        print('scale', scale)
       
        # concat column keys
        d = {}
        for k in d0:
            if (d0[k][1] in ['x','y']):
                d[d0[k][0]+'_'+d0[k][1]] = scale * np.array(
                                               d0[k][2:].values, 
                                               dtype=np.float32)
            else:
                d[d0[k][0]+'_'+d0[k][1]] = np.array(
                                               d0[k][2:].values, 
                                               dtype=np.float32)                
          
             
        del d['bodyparts_coords']
        
#        k0 = list(d.keys())
#        for k in k0:
#            if 'likelihood' in k:
#                del d[k]    

    
    d['times'] = np.load(one.eid2path(eid) / 'alf'
                    / f'_ibl_leftCamera.times.npy')
                    

    ls = [len(d[x]) for x in d]
    if not all(ls == np.mean(ls)):
        lsd = {x:len(d[x]) for x in d}
        print(f'length mismatch: {lsd}')
        print(eid, cam)
        print('cutting times')
        d['times'] = d['times'][:ls[0]]

    if (not paws and reso == '128x102_128x128'):
        # load old complete lp file        
        pth = ('/mnt/8cfe1683-d974-40f3-a20b-b217cad4722a/lp_ens'
              f'/{reso}/{eid}/_ibl_{cam}Camera.lightningPose.pqt') 

        d = pd.read_parquet(pth)    

        if masked:
            points = np.unique(['_'.join(x.split('_')[:-1]) 
                                for x in d.keys()])[1:]
        
            for point in points:
                cond = d[f'{point}_likelihood'] < 0.9
                d.loc[cond, [f'{point}_x', f'{point}_y']] = np.nan

    return d


def load_dlc(eid, cam, smoothed=False, manual=True):

    '''
    cam in left, right, body 
    '''

    if manual:
        pth = one.eid2path(eid)    
        d = pd.read_parquet(pth / 'alf' / f'_ibl_{cam}Camera.dlc.pqt')
        d['times'] = np.load(one.eid2path(eid) / 'alf'
                    / f'_ibl_{cam}Camera.times.npy')
                    
        ls = [len(d[x]) for x in d]
        if not all(ls == np.mean(ls)):
            lsd = {x:len(d[x]) for x in d}
            print(f'length mismatch: {lsd}')
            print(eid, cam)
            print('cutting times')
            d['times'] = d['times'][:ls[0]]            

    else:
        # load DLC
        sess_loader = SessionLoader(one, eid)
        sess_loader.load_pose(views=[cam])
        d = sess_loader.pose[f'{cam}Camera']
    
    if smoothed:
        print('smoothing dlc traces')
        window = 13 if cam == 'right' else 7
        sers = [x for x in d.keys() if (x[-1] in ['x','y'])]# and 'paw' in x
        for ser in sers:
            d[ser] = smooth_interpolate_savgol(
                d[ser].to_numpy(),
                window=window,order=3, interp_kind='linear')   

    return d




def Viewer(eid, video_type, frame_start, frame_stop, save_video=True, 
           eye_zoom=False, lp=False, ens=False,
           res = '128x102_128x128', masked=True, paws_only=False,
           smooth_dlc = False):
           
    '''
    eid: session id, e.g. '3663d82b-f197-4e8b-b299-7b803a155b84'
    video_type: one of 'left', 'right', 'body'
    save_video: video is saved this local folder

    Example usage to view and save labeled video with wheel angle:
    Viewer('3663d82b-f197-4e8b-b299-7b803a155b84', 'left', [5,7])
    3D example: 'cb2ad999-a6cb-42ff-bf71-1774c57e5308', [5,7]
    
    Different resolutions:
    128x102_128x128
    320x256_128x128
    320x256_256x256
    
    
    paws: paws only
    '''

    save_vids_here = Path.home()


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
    
    
    if lp and ens:
        print('either lp or ens must be False')
        return
     
    if lp:

        cam = load_lp(eid, video_type, paws=True,
              reso=res, flav='multi')
              
              
        print(cam.keys())        
   
    elif ens:
        print('load ensembling results')
        pth = Path('/mnt/8cfe1683-d974-40f3-a20b-b217cad4722a/lp_ens'
          f'/{res}/{eid}/ensembles_{video_type}Camera')

        cams = []
        nets = 4 if res[:7] == '320x256' else 5
        
        if res[:7] == '128x102':
            scale = 10 if video_type == 'left' else 5
        else:    
            scale = 4 if video_type == 'left' else 2

        for net in range(nets):   
            cam = {}
#            try:
#                df = pd.read_csv(pth / 
#                        f'_iblrig_{video_type}Camera.raw.eye{net}.csv')
#                cam = cam | {'_'.join([df[x][0],df[x][1]]): 
#                         scale* np.array(list(map(float, df[x][2:]))) 
#                         for x in df.keys() if df[x][1] in ['x','y','likelihood']}
#            except:
#                pass
                
            df = pd.read_csv(pth / f'_iblrig_{video_type}Camera.raw.paws{net}.csv')

            cam = cam | {'_'.join([df[x][0],df[x][1]]): 
                     scale* np.array(list(map(float, df[x][2:]))) 
                     for x in df.keys() if df[x][1] in ['x','y','likelihood']}
                     
            cams.append(cam)
                    
        cam = cams[0]           
        
    else: 
        print('loading dlc')
        cam = load_dlc(eid, video_type, smoothed=smooth_dlc)
                                                      
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

    print('frame start stop', frame_start, frame_stop)

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

    w_start = find_nearest(t, Times[frame_start])
    w_stop = find_nearest(t, Times[frame_stop])

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
    Frames = np.arange(frame_start, frame_stop)
    
    
    # liklihood threshold
    l_thr = 0.9 if masked else -1

    points = [x[:-2] for x in cam.keys() if x[-1] == 'x']

    if video_type != 'body':
        try:
            d = list(points)
            d.remove('tube_top')
            d.remove('tube_bottom')
            points = d
        except:
            pass

    if paws_only:
        p2 = deepcopy(points)
        for point in p2:
            if not 'paw' in point:
                points.remove(point)            

    points = np.array(points)
    
    if ens:

        XYss = []
        
        for cam in cams:
            # Set values to nan if likelyhood is too low # for pqt: .to_numpy()
            XYs = {}
            for point in points:
                x = np.ma.masked_where(
                    cam[point + '_likelihood'] < l_thr, cam[point + '_x'])
                x = x.filled(np.nan)
                y = np.ma.masked_where(
                    cam[point + '_likelihood'] < l_thr, cam[point + '_y'])
                y = y.filled(np.nan)
                XYs[point] = np.array(
                    [x[frame_start:frame_stop], y[frame_start:frame_stop]])        
        
            XYss.append(XYs)    
        cam = cams[0]
        
        
    # Set values to nan if likelyhood is too low # for pqt: .to_numpy()
    XYs = {}
    for point in points:
        x = np.ma.masked_where(
            cam[point + '_likelihood'] < l_thr, cam[point + '_x'])
        x = x.filled(np.nan)
        y = np.ma.masked_where(
            cam[point + '_likelihood'] < l_thr, cam[point + '_y'])
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
    
        rr = f'_{res}' if ens else ''
        loc = (save_vids_here / 
        f'{eid}_{video_type}_frames_{frame_start}_{frame_stop}'
        f'_lp_{lp}_ens_{ens}{rr}.mp4')

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
                    
                    
        bottomLeftCornerOfText1 = (a, b - 3* a)
        cv2.putText(gray,
                    'Frame: ' + str(Frames[k]),
                    bottomLeftCornerOfText1,
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

            if ens:
                for XYs in XYss:

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
                        gray[X - dot_s:X + dot_s, Y - 
                             dot_s:Y + dot_s] = block * col
                
                
            if not ens:
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
                    gray[X - dot_s:X + dot_s, Y - 
                         dot_s:Y + dot_s] = block * col            

                
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
