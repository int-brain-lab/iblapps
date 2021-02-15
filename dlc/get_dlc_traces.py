import alf.io
import numpy as np
from oneibl.one import ONE

def get_DLC(eid,video_type):
    '''load dlc traces
    load dlc traces for a given session and
    video type.

    :param eid: A session eid
    :param video_type: string in 'left', 'right', body'
    :return: array of times and dict with dlc points
             as keys and x,y coordinates as values,
             for each frame id
    '''   
    one = ONE()    
    D = one.load(eid, dataset_types = ['camera.dlc', 'camera.times'])
    alf_path = one.path_from_eid(eid) / 'alf'
    cam0 = alf.io.load_object(
            alf_path,
            '%sCamera' %
            video_type,
            namespace='ibl')
    Times = cam0['times']
    cam = cam0['dlc']
    points = np.unique(['_'.join(x.split('_')[:-1]) for x in cam.keys()])
    XYs = {}
    for point in points:
        x = np.ma.masked_where(
            cam[point + '_likelihood'] < 0.9, cam[point + '_x'])
        x = x.filled(np.nan)
        y = np.ma.masked_where(
            cam[point + '_likelihood'] < 0.9, cam[point + '_y'])
        y = y.filled(np.nan)
        XYs[point] = np.array([x, y])
        
    return Times, XYs
