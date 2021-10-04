import numpy as np
import alf.io
#pip install ONE-api
from one.api import ONE
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#os.environ['CUDA_VISIBLE_DEVICES'] = str(gputouse)
from ibllib.io.video import get_video_meta, get_video_frames_preload
import ibllib.io.video as vidio
import brainbox.behavior.wheel as wh
import math
from brainbox.processing import bincount2D
import time
import copy
import matplotlib.patches as patches
import string
from scipy.interpolate import interp1d
from scipy.stats import zscore

#Generate scatterplots, variances, 2-state AR-HMMs to summarize differences in behavior in different sessions. Develop a 1-page figure per session that provides a behavioral overview.  Would be great to generate these figs for the RS sessions noted in repro-ephys slides above
#lickogram
import matplotlib
# https://github.com/lindermanlab/ssm
import os
   
   
def get_all_sess_with_ME():
    one = ONE()
    # get all bwm sessions with dlc
    all_sess = one.alyx.rest('sessions', 'list', 
                              project='ibl_neuropixel_brainwide_01',
                              task_protocol="ephys", 
                              dataset_types='camera.ROIMotionEnergy')

    eids = [s['url'].split('/')[-1] for s in all_sess]
    
    return eids     
    

def get_repeated_sites():    
    one = ONE()
    STR_QUERY = 'probe_insertion__session__project__name__icontains,ibl_neuropixel_brainwide_01,' \
                'probe_insertion__session__qc__lt,50,' \
                '~probe_insertion__json__qc,CRITICAL,' \
                'probe_insertion__session__n_trials__gte,400'
    all_sess = one.alyx.rest('trajectories', 'list', provenance='Planned',
                                  x=-2243, y=-2000, theta=15,
                                  django=STR_QUERY)
    eids = [s['session']['id'] for s in all_sess]
    
    return eids
    
    
def get_bwm_sessions():
    one = ONE()
    traj_traced = one.alyx.rest('trajectories', 'list', provenance='Planned',
                         django='probe_insertion__session__project__name__'
                                'icontains,ibl_neuropixel_brainwide_01,'
                                'probe_insertion__session__qc__lt,50,'
                                'probe_insertion__session__extended_qc__behavior,1,'
                                'probe_insertion__json__extended_qc__tracing_exists,True,'
                                '~probe_insertion__session__extended_qc___task_stimOn_goCue_delays__lt,0.9,'
                                '~probe_insertion__session__extended_qc___task_response_feedback_delays__lt,0.9,'
                                '~probe_insertion__session__extended_qc___task_response_stimFreeze_delays__lt,0.9,'
                                '~probe_insertion__session__extended_qc___task_wheel_move_before_feedback__lt,0.9,'
                                '~probe_insertion__session__extended_qc___task_wheel_freeze_during_quiescence__lt,0.9,'
                                '~probe_insertion__session__extended_qc___task_error_trial_event_sequence__lt,0.9,'
                                '~probe_insertion__session__extended_qc___task_correct_trial_event_sequence__lt,0.9,'
                                '~probe_insertion__session__extended_qc___task_n_trial_events__lt,0.9,'
                                '~probe_insertion__session__extended_qc___task_reward_volumes__lt,0.9,'
                                '~probe_insertion__session__extended_qc___task_reward_volume_set__lt,0.9,'
                                '~probe_insertion__session__extended_qc___task_stimulus_move_before_goCue__lt,0.9,'
                                '~probe_insertion__session__extended_qc___task_audio_pre_trial__lt,0.9')
    eids = [[x['session']['id'],x['probe_name']] for x in traj_traced]
    return eids    
    
    
def check_progress():
    
    one = ONE()
    eids = get_repeated_sites()
    
    s = {}
    comp = []
    for eid in eids:
        
        task = one.alyx.rest('tasks', 'list', session=eid, name='EphysDLC')[0]
        try:
            s[eid] = task['version']
            print(task['version'])
            if task['version'][:3] != '1.1':
                comp.append(eid)    
        except:
            print(eid, 'has no task version')        
    
    return s    
    
    
def download_all_dlc():
    
    eids = get_repeated_sites()    
    one = ONE() 
    dataset_types = ['camera.dlc', 'camera.times']
    
    for eid in eids: 
        try:
                    
            a = one.list(eid,'dataset-types')
            # for newer iblib version do [x['dataset_type'] for x in a]
            if not all([x['dataset_type'] for x in a]):
                print('not all data available')    
                continue
            
                         
            one.load(eid, dataset_types = dataset_types)    
        except:
            continue    
    
    
def constant_reaction_time(eid, rt, st,stype='stim'):


    '''
    getting trial numbers, feedback time interval
    '''
     
    one = ONE()    
    if stype == 'motion':
        wheelMoves = one.load_object(eid, 'wheelMoves')
    trials = one.load_object(eid, 'trials')
    d = {} # dictionary, trial number and still interval    
    evts = ['goCue_times', 'feedback_times', 'probabilityLeft',
            'choice', 'feedbackType']
            
    for tr in range(len(trials['intervals'])):
        if stype == 'motion':
            a = wheelMoves['intervals'][:,0]
        b = trials['goCue_times'][tr]  
        c = trials['feedback_times'][tr]
        ch = trials['choice'][tr]
        pl = trials['probabilityLeft'][tr]
        ft = trials['feedbackType'][tr]
        
        
        if any(np.isnan([trials[k][tr] for k in evts])):
            continue
               
              
        if c-b>10: # discard too long trials
            continue 
                
        if stype == 'motion':       
            # making sure the motion onset time is in a coupled interval
            ind = np.where((a > b) & (a < c), True, False)
            if all(~ind):
                #print(f'non-still start, trial {tr} and eid {eid}')
                continue
                
            a = a[ind][0]       
            react = np.round(a - b,3)           
                    
        if np.isnan(trials['contrastLeft'][tr]):
            cont = trials['contrastRight'][tr]            
            side = 0
        else:   
            cont = trials['contrastLeft'][tr]         
            side = 1              
                
        if stype == 'feedback':
            d[tr] = [c + st,rt, cont, side, ch, ft]
        if stype == 'stim':        
            d[tr] = [b + st,rt, cont, side, ch, ft]
        if stype == 'motion':
            d[tr] = [a + st,rt, cont, side, ch, ft]            
                
        
    print(f"cut {len(d)} of {len(trials['intervals'])} full trials segments")
    return d  

    
def get_dlc_XYs(eid, video_type):

    #video_type = 'left'    
    one = ONE() 
    Times = one.load_dataset(eid,f'alf/_ibl_{video_type}Camera.times.npy') 
    cam = one.load_dataset(eid,f'alf/_ibl_{video_type}Camera.dlc.pqt')
    points = np.unique(['_'.join(x.split('_')[:-1]) for x in cam.keys()])

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
            [x, y])    

    return Times, XYs 

#    a = one.list_datasets(eid)
#    if not all([u in a for u in dataset_types]):
#        print('not all data available')    
#        return
        
def get_ME(eid, video_type):

    #video_type = 'left'    
    one = ONE()       
    
    
    Times = one.load_dataset(eid,f'alf/_ibl_{video_type}Camera.times.npy') 
    ME = one.load_dataset(eid,f'alf/{video_type}Camera.ROIMotionEnergy.npy')

    return Times, ME 


    Times = cam0['times']  

def get_example_images(eid):


    eids = get_repeated_sites()
#    eid = eids[23]
#    video_type = 'body'

    #eids = ['15f742e1-1043-45c9-9504-f1e8a53c1744']
    eids = ['4a45c8ba-db6f-4f11-9403-56e06a33dfa4']
    frts = {'body':30, 'left':60,'right':150}    

    one=ONE()   
    
    
    #for eid in eids:
    urls = vidio.url_from_eid(eid, one=one)
    for video_type in frts:
        
        frame_idx = [5 * 60 * frts[video_type]]    
        try:
        
            
            video_path = urls[video_type]
            frames = get_video_frames_preload(video_path,
                                              frame_idx,
                                              mask=np.s_[:, :, 0])
            np.save('/home/mic/reproducible_dlc/example_images/'
                    f'{eid}_{video_type}.npy', frames)
            print(eid, video_type, 'done')
        except:
            print(eid, video_type,'error')  
            continue


    #'/home/mic/reproducible_dlc/example_images'

def get_mean_positions(XYs):
    mloc = {} # mean locations
    for point in XYs:
        if point in ['tube_bottom', 'tube_top']:
            continue
        mloc[point] = [int(np.nanmean(XYs[point][0])),
                       int(np.nanmean(XYs[point][1]))]
    return mloc
               
    
def plot_paw_on_image(eid, video_type='left', XYs = None):

    #fig = plt.figure(figsize=(8,4)) 
 
    Cs = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA',
          '#FFA15A', '#19D3F3', '#FF6692', '#B6E880',
          '#FF97FF', '#FECB52','cyan']

    try:
        r = np.load(f'/home/mic/reproducible_dlc/example_images/'
                  f'{eid}_{video_type}.npy')[0]  
    except:
        get_example_images(eid)
        r = np.load(f'/home/mic/reproducible_dlc/example_images/'
                  f'{eid}_{video_type}.npy')[0]         
    
    try:        
        if XYs == None:          
            _, XYs =  get_dlc_XYs(eid, video_type)  
       
        Cs = dict(zip(XYs.keys(),Cs))
        
        ds = {'body':3,'left':6,'right':15}
        
        
        for point in XYs:# ['paw_l','paw_r']:
            if point in ['tube_bottom', 'tube_top']:
                continue
         
            # downsample; normalise number of points to be the same
            # across all sessions
            xs = XYs[point][0][0::ds[video_type]]
            ys = XYs[point][1][0::ds[video_type]]
        
            plt.scatter(xs,ys, alpha = 0.05, s = 2, 
                       label = point, c = Cs[point])   
                        
        plt.axis('off')
        plt.tight_layout()
        plt.imshow(r,cmap='gray')
        plt.tight_layout()    
                                
                                  
        if video_type != 'body':      
        
            if video_type == 'left':
                fa = 1
            elif video_type == 'right':       
                fa = 0.5      
                                         
            # plot whisker pad rectangle               
            mloc = get_mean_positions(XYs)
            p_nose = np.array(mloc['nose_tip'])
            p_pupil = np.array(mloc['pupil_top_r'])    

            # heuristic to find whisker area in side videos:
            # square with side length half the distance 
            # between nose and pupil and anchored on midpoint

            p_anchor = np.mean([p_nose,p_pupil],axis=0)
            squared_dist = np.sum((p_nose-p_pupil)**2, axis=0)
            dist = np.sqrt(squared_dist)
            whxy = [int(dist/2), int(dist/3), 
                    int(p_anchor[0] - dist/4), int(p_anchor[1])]     
            
            rect = patches.Rectangle((whxy[2], whxy[3]), whxy[0], whxy[1], 
                                     linewidth=1, 
                                     edgecolor='lime',
                                     facecolor='none')               
            ax = plt.gca()                                  
            ax.add_patch(rect)                                       


            
            # plot eye region zoom as inset in upper right corner
            pivot = np.nanmean(XYs['pupil_top_r'], axis=1)
            x0 = int(pivot[0] - 33 * fa)
            x1 = int(pivot[0] + 33 * fa)
            y0 = int(pivot[1] - 28 * fa)
            y1 = int(pivot[1] + 38 * fa)        
            
            if video_type == 'right':
                axins = ax.inset_axes([0, -0.5, 0.5, 0.5])
            elif video_type == 'left':
                axins = ax.inset_axes([0.5, -0.5, 0.5, 0.5])
                
            axins.spines['bottom'].set_color('white')
            axins.spines['top'].set_color('white') 
            axins.spines['right'].set_color('white')
            axins.spines['left'].set_color('white')
            axins.imshow(r, cmap='gray', origin="lower")
            
            for point in XYs:# ['paw_l','paw_r']:
                if point in ['tube_bottom', 'tube_top']:
                    continue
             
                # downsample; normalise number of points to be the same
                # across all sessions
                xs = XYs[point][0][0::ds[video_type]]
                ys = XYs[point][1][0::ds[video_type]]
            
                axins.scatter(xs,ys, alpha = 1, s = 0.001, 
                           label = point, c = Cs[point]) 
            axins.set_xlim(x0, x1)
            axins.set_ylim(y1, y0)
            axins.set_xticklabels('')
            axins.set_yticklabels('')
            #ax.indicate_inset_zoom(axins, edgecolor="white")
            
            # plot tongue region zoom as inset in llower right corner
            p1 = np.nanmean(XYs['tube_top'], axis=1)
            p2 = np.nanmean(XYs['tube_bottom'], axis=1)
            pivot = np.nanmean([p1,p2], axis=0)
            x0 = int(pivot[0] - 60 * fa)
            x1 = int(pivot[0] + 100 * fa)
            y0 = int(pivot[1] - 100 * fa)
            y1 = int(pivot[1] + 60 * fa) 
                 
            if video_type == 'right':
                axins = ax.inset_axes([0.5, -0.5, 0.5, 0.5])
            elif video_type == 'left':
                axins = ax.inset_axes([0, -0.5, 0.5, 0.5])
                
            axins.spines['bottom'].set_color('white')
            axins.spines['top'].set_color('white') 
            axins.spines['right'].set_color('white')
            axins.spines['left'].set_color('white')
            axins.imshow(r, cmap='gray', origin="upper")
            
            for point in XYs:# ['paw_l','paw_r']:
                if point in ['tube_bottom', 'tube_top']:
                    continue 
                # downsample; normalise number of points to be the same
                # across all sessions
                xs = XYs[point][0][0::ds[video_type]]
                ys = XYs[point][1][0::ds[video_type]]
            
                axins.scatter(xs,ys, alpha = 1, s = 0.001, 
                           label = point, color = Cs[point]) 
            axins.set_xlim(x0, x1)
            axins.set_ylim(y1, y0)
            axins.set_xticklabels('')
            axins.set_yticklabels('')
            #ax.indicate_inset_zoom(axins, edgecolor="white")        
        
    except:
   
        plt.imshow(r,cmap='gray')    
        ax = plt.gca() 
        plt.text(.5, .5,'DLC is nan',color='r',fontweight='bold',
                 bbox=dict(facecolor='white', alpha=0.5),
                 fontsize=10,transform=ax.transAxes)                 
        plt.tight_layout()
    #plt.show()
    plt.title(video_type)    


def plot_paw_on_imageL(eid):
    plot_paw_on_image(eid, video_type='left')
    
def plot_paw_on_imageR(eid):
    plot_paw_on_image(eid, video_type='right')    
    
def plot_paw_on_imageB(eid):
    plot_paw_on_image(eid, video_type='body')      


def paw_speed_PSTH(eid):

    rt = 2
    st = -0.5 
    fs = {'right':150,'left':60}   
    
    cols = {'left':{'1':['darkred','--'],'-1':['#1f77b4','--']},
            'right':{'1':['darkred','-'],'-1':['#1f77b4','-']}}
    # take speed from right paw only, i.e. closer to cam
    # for each video
    speeds = {}
    for video_type in ['right','left']:
        times, XYs = get_dlc_XYs(eid, video_type)
        x = XYs['paw_r'][0]
        y = XYs['paw_r'][1]   
        if video_type == 'left': #make resolution same
            x = x/2
            y = y/2
        
        # get speed in px/sec [half res]
        s = ((np.diff(x)**2 + np.diff(y)**2)**.5)*fs[video_type]
        
        
        speeds[video_type] = [times,s]   
      
    
    # that's centered at feedback time
    d = constant_reaction_time(eid, rt, st,stype='stim') 
    
    sc = {'left':{'1':[],'-1':[]},'right':{'1':[],'-1':[]}}
    for video_type in ['right','left']: 
    
        times,s = speeds[video_type]
        for i in d:

            start_idx = find_nearest(times,d[i][0])
            end_idx = find_nearest(times,d[i][0]+d[i][1])   

            sc[video_type][str(d[i][4])].append(s[start_idx:end_idx])

    # trim on e frame if necessary
    for video_type in ['right','left']:    
        for choice in ['-1','1']:
            m = min([len(x) for x in sc[video_type][choice]])
            q = [x[:m] for x in sc[video_type][choice]]
            xs = np.arange(m) / fs[video_type]
            xs = np.concatenate([
                 -1*np.array(list(reversed(xs[:int(abs(st)*fs[video_type])]))),
                 np.array(xs[:int((rt - abs(st))*fs[video_type])])])  
            m = min(len(xs),m)           

            c =  cols[video_type][choice][0]
            ls = cols[video_type][choice][1]  
            
            qm = np.nanmean(q,axis=0) 
            plt.plot(xs[:m], qm[:m], c = c,linestyle=ls, 
                     linewidth = 1, 
                     label = f'paw {video_type[0]},' + ' choice ' + choice)
            
    ax = plt.gca()
    ax.axvline(x=0, label='stimOn', linestyle = '--', c='g')
    plt.title('paw speed PSTH')
    plt.xlabel('time [sec]')
    plt.ylabel('speed [px/sec]') 
    plt.legend()#bbox_to_anchor=(1.05, 1), loc='upper left')    
        
        
def nose_speed_PSTH(eid,vtype='right'):

    '''
    nose speed PSTH
    eid = 'd0ea3148-948d-4817-94f8-dcaf2342bbbe' is good
    '''

    rt = 2
    st = -0.5 
    fs = {'right':150,'left':60}   
    cts = {'1':'correct trials', '-1':'incorrect trials'}
    
    
    cols = {'left':{'1':['k','--'],'-1':['gray','--']},
            'right':{'1':['k','-'],'-1':['gray','-']}}
    # take speed from right paw only, i.e. closer to cam
    # for each video
    speeds = {}
    for video_type in ['right','left']:
        times, XYs = get_dlc_XYs(eid, video_type)
        x = XYs['nose_tip'][0]
        y = XYs['nose_tip'][1]   
        if video_type == 'left': #make resolution same
            x = x/2
            y = y/2
        
        # get speed in px/sec [half res]
        s = ((np.diff(x)**2 + np.diff(y)**2)**.5)*fs[video_type]
        
        
        speeds[video_type] = [times,s]   
      
    # average nose tip speeds across cams
    
    
    
    # that's centered at feedback time
    d = constant_reaction_time(eid, rt, st,stype='feedback') 
    
    sc = {'left':{'1':[],'-1':[]},'right':{'1':[],'-1':[]}}
    for video_type in ['right','left']: 
    
        times,s = speeds[video_type]
        for i in d:

            start_idx = find_nearest(times,d[i][0])
            end_idx = find_nearest(times,d[i][0]+d[i][1])   

            sc[video_type][str(d[i][5])].append(s[start_idx:end_idx])

    # trim on e frame if necessary
    for video_type in [vtype]: # 'right','left'  
        for choice in ['-1','1']:
            m = min([len(x) for x in sc[video_type][choice]])
            q = [x[:m] for x in sc[video_type][choice]]
            xs = np.arange(m) / fs[video_type]
            xs = np.concatenate([
                 -1*np.array(list(reversed(xs[:int(abs(st)*fs[video_type])]))),
                 np.array(xs[:int((rt - abs(st))*fs[video_type])])])  
            m = min(len(xs),m)           

            c =  cols[video_type][choice][0]
            ls = cols[video_type][choice][1]  
            
            qm = np.nanmean(q,axis=0) 
            plt.plot(xs[:m], qm[:m], c = c,linestyle=ls, 
                     linewidth = 1, 
                     label = cts[choice])
            
    ax = plt.gca()
    ax.axvline(x=0, label='stimOn', linestyle = '--', c='r')
    plt.title('nose tip speed PSTH, '
              f'{vtype} vid')
    plt.xlabel('time [sec]')
    plt.ylabel('speed [px/sec]') 
    plt.legend(loc='lower right')     
    
       
def get_licks(XYs):

    '''
    define a frame as a lick frame if
    x or y for left or right tongue point
    change more than half the sdt of the diff
    '''

    licks = []
    for point in ['tongue_end_l', 'tongue_end_r']:
        for c in XYs[point]:
           thr = np.nanstd(np.diff(c))/4
           licks.append(set(np.where(abs(np.diff(c))>thr)[0]))
    return sorted(list(set.union(*licks)))


def plot_licks(eid, combine=True):

    '''
    lick PSTH
    eid = 'd0ea3148-948d-4817-94f8-dcaf2342bbbe' is good
    '''
    T_BIN = 0.02
    rt = 2
    st = -0.5  
    
    if combine:    
        # combine licking events from left and right cam
        lick_times = []
        for video_type in ['right','left']:
            times, XYs = get_dlc_XYs(eid, video_type)
            # for the case that there are less times than DLC points
            r = get_licks(XYs)
            idx = np.where(np.array(r)<len(times))[0][-1]            
            lick_times.append(times[r[:idx]])
        
        lick_times = sorted(np.concatenate(lick_times))
        
    else:
        times, XYs = get_dlc_XYs(eid, 'left')    
        r = get_licks(XYs)
        idx = np.where(np.array(r)<len(times))[0][-1]              
        lick_times = times[r[:idx]]
   
    
    R, t, _ = bincount2D(lick_times, np.ones(len(lick_times)), T_BIN)
    D = R[0]
    
    # that's centered at feedback time
    d = constant_reaction_time(eid, rt, st,stype='feedback') 
    
    licks_pos = []
    licks_neg = []
    
    for i in d:
      
        start_idx = find_nearest(t,d[i][0])
        end_idx = start_idx + int(d[i][1]/T_BIN)   

        if end_idx > len(D):
            break

        # split by feedback type)        
        if d[i][5] == 1:                                             
            licks_pos.append(D[start_idx:end_idx])
        if d[i][5] == -1:
            licks_neg.append(D[start_idx:end_idx])  

 
    licks_pos_ = np.array(licks_pos).mean(axis=0)
    licks_neg_ = np.array(licks_neg).mean(axis=0)

    xs = np.arange(int(rt/T_BIN))
    xs = np.concatenate([
                       -1*np.array(list(reversed(xs[:int(len(xs)*abs(st/rt))]))),
                       np.array(xs[:int(len(xs)*(1 - abs(st/rt)))])])
    xs = xs*T_BIN                   
    plt.plot(xs, licks_pos_, c = 'k', label = 'correct trial')
    plt.plot(xs, licks_neg_, c = 'gray', label = 'incorrect trial')     
    ax = plt.gca()
    ax.axvline(x=0, label='feedback time', linestyle = '--', c='r')
    plt.title('lick PSTH')
    plt.xlabel('time [sec]')
    plt.ylabel('lick events \n [a.u.]') 
    plt.legend(loc='lower right')
    


def lick_raster(eid, combine=True):

    #plt.figure(figsize=(4,4))
    
    T_BIN = 0.02
    rt = 2
    st = -0.5     
    
    if combine:    
        # combine licking events from left and right cam
        lick_times = []
        for video_type in ['right','left']:
            times, XYs = get_dlc_XYs(eid, video_type)
            r = get_licks(XYs)
            idx = np.where(np.array(r)<len(times))[0][-1]            
            lick_times.append(times[r[:idx]])
        
        lick_times = sorted(np.concatenate(lick_times))
        
    else:
        times, XYs = get_dlc_XYs(eid, 'left')    
        r = get_licks(XYs)
        idx = np.where(np.array(r)<len(times))[0][-1]
        lick_times = times[r[:idx]]
        
    R, t, _ = bincount2D(lick_times, np.ones(len(lick_times)), T_BIN)
    D = R[0]  
      
    # that's centered at feedback time
    d = constant_reaction_time(eid, rt, st,stype='feedback') 
    
    licks_pos = []
    licks_neg = []
    
    for i in d:
      
        start_idx = find_nearest(t,d[i][0])
        end_idx = start_idx + int(d[i][1]/T_BIN)   
        
        if end_idx > len(D):
            break
            
        # split by feedback type)        
        if d[i][5] == 1:                                             
            licks_pos.append(D[start_idx:end_idx])    
    
    licks_pos_ = np.array(licks_pos).mean(axis=0)
    
    y_dims, x_dims = len(licks_pos), len(licks_pos[0])
    plt.imshow(licks_pos,aspect='auto', extent=[-0.5,1.5,y_dims,0],
               cmap='gray_r')
    
    ax = plt.gca()
    ax.set_xticks([-0.5,0,0.5,1,1.5])
    ax.set_xticklabels([-0.5,0,0.5,1,1.5])
    plt.ylabel('trials')
    plt.xlabel('time [sec]')
    ax.axvline(x=0, label='feedback time', linestyle = '--', c='r')
    plt.title('lick events per correct trial')
    plt.tight_layout()



def find_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx-1
    else:
        return idx      


def plot_wheel_position(eid):

    '''
    illustrate wheel position next to distance plot
    '''
    T_BIN = 0.02       
    rt = 2
    st = -0.5
    
    d = constant_reaction_time(eid, rt, st) 
    
    one = ONE()   
    wheel = one.load_object(eid, 'wheel')
    pos, t = wh.interpolate_position(wheel.timestamps, 
                                     wheel.position, 
                                     freq=1/T_BIN)
    whe_left = []
    whe_right = []
    
    for i in d:
      
        start_idx = find_nearest(t,d[i][0])
        end_idx = start_idx + int(d[i][1]/T_BIN)   
     
        wheel_pos = pos[start_idx:end_idx]
        if len(wheel_pos) == 1:
            print(i, [start_idx,end_idx])
        
        wheel_pos = wheel_pos - wheel_pos[0]
    
        if d[i][4] == -1:                                             
            whe_left.append(wheel_pos)
        if d[i][4] == 1:
            whe_right.append(wheel_pos)    

    xs = np.arange(len(whe_left[0]))*T_BIN
    times = np.concatenate([
                       -1*np.array(list(reversed(xs[:int(len(xs)*abs(st/rt))]))),
                       np.array(xs[:int(len(xs)*(1 - abs(st/rt)))])])

   
    for i in range(len(whe_left)):
        plt.plot(times, whe_left[i],c='#1f77b4', alpha =0.5, linewidth = 0.05)
    for i in range(len(whe_right)):        
        plt.plot(times, whe_right[i],c='darkred', alpha =0.5, linewidth = 0.05)

    plt.plot(times, np.mean(whe_left,axis=0),c='#1f77b4', 
             linewidth = 2, label = 'left')
    plt.plot(times, np.mean(whe_right,axis=0),c='darkred', 
             linewidth = 2, label = 'right')

    plt.axhline(y=0.26, linestyle = '--', c = 'k')
    plt.axhline(y=-0.26, linestyle = '--', c = 'k', label = 'reward boundary')  
    plt.axvline(x=0, linestyle = '--', c = 'g', label = 'stimOn')  
    axes = plt.gca()
    #axes.set_xlim([0,rt])
    axes.set_ylim([-0.27,0.27])
    plt.xlabel('time [sec]')
    plt.ylabel('wheel position [rad]')
    plt.legend(loc='lower right')
    plt.title('wheel positions colored by choice')
    plt.tight_layout()          


def get_sniffs(XYs):

    '''
    define a frame as a sniff frame if
    x or y for left or right tongue point
    change more than half the sdt of the diff
    '''

    c = XYs['nose_tip'][1] # on
    thr = np.nanstd(np.diff(c))/4
    sniffs = np.where(abs(np.diff(c))>thr)[0]
    return sniffs


def plot_sniffPSTH(eid, combine=False):

    '''
    sniff PSTH
    eid = 'd0ea3148-948d-4817-94f8-dcaf2342bbbe' is good
    '''
    T_BIN = 0.02
    rt = 2
    st = -0.5   
    
    if combine:    
        # combine licking events from left and right cam
        lick_times = []
        for video_type in ['right','left']:
            times, XYs = get_dlc_XYs(eid, video_type)
            lick_times.append(times[get_sniffs(XYs)])
        
        lick_times = sorted(np.concatenate(lick_times))
        
    else:
        times, XYs = get_dlc_XYs(eid, 'left')    
        lick_times = times[get_sniffs(XYs)]
    
    
    R, t, _ = bincount2D(lick_times, np.ones(len(lick_times)), T_BIN)
    D = R[0]
    
    # that's centered at feedback time
    d = constant_reaction_time(eid, rt, st,stype='feedback') 
    
    licks_pos = []
    licks_neg = []
    
    for i in d:
      
        start_idx = find_nearest(t,d[i][0])
        end_idx = start_idx + int(d[i][1]/T_BIN)   

        # split by feedback type)        
        if d[i][5] == 1:                                             
            licks_pos.append(D[start_idx:end_idx])
        if d[i][5] == -1:
            licks_neg.append(D[start_idx:end_idx])  

 
    licks_pos = np.array(licks_pos).mean(axis=0)
    licks_neg = np.array(licks_neg).mean(axis=0)

    xs = np.arange(int(rt/T_BIN))

    xs = np.concatenate([
                       -1*np.array(list(reversed(xs[:int(len(xs)*abs(st/rt))]))),
                       np.array(xs[:int(len(xs)*(1 - abs(st/rt)))])])

    xs = xs*T_BIN                   
    plt.plot(xs, licks_pos, c = 'k', label = 'correct trial')
    plt.plot(xs, licks_neg, c = 'gray', label = 'incorrect trial')     
    ax = plt.gca()
    ax.axvline(x=0, label='feedback time', linestyle = '--', c='r')
    plt.title('sniff PSTH')
    plt.xlabel('time [sec]')
    plt.ylabel('sniff events \n [a.u.]') 
    plt.legend(loc='lower right')


def get_pupil_diameter(XYs, smooth=True):
    """get mean of pupil diameter in two different ways:
    d1 = top - bottom, d2 = left - right
    and in addition assume it's a circle and
    estimate diameter from other pairs of points
    Author: Michael Schartner
    """

    # direct diameters
    t = XYs['pupil_top_r'].T
    b = XYs['pupil_bottom_r'].T
    l = XYs['pupil_left_r'].T
    r = XYs['pupil_right_r'].T

    ds = []
    def distance(p1, p2):
        return ((p1[:, 0] - p2[:, 0]) ** 2 + (p1[:, 1] - p2[:, 1]) ** 2) ** 0.5
    # get diameter via top-bottom and left-right
    ds.append(distance(t, b))
    ds.append(distance(l, r))
    def dia_via_circle(p1, p2):
        # only valid for non-crossing edges
        u = distance(p1, p2)
        return u * (2 ** 0.5)
    # estimate diameter via circle assumption
    for side in [[t, l], [t, r], [b, l], [b, r]]:
        ds.append(dia_via_circle(side[0], side[1]))
            
        
    diam = np.nanmedian(ds, axis=0)
    if smooth:
        return smooth_pupil_diameter(diam)
    else:
        return diam


def smooth_pupil_diameter(diam, window=31, order=3, interp_kind='cubic'):
    
    signal_noisy_w_nans = np.copy(diam)
    timestamps = np.arange(signal_noisy_w_nans.shape[0])
    good_idxs = np.where(~np.isnan(signal_noisy_w_nans))
    # perform savitzky-golay filtering on non-nan points
    signal_smooth_nonans = non_uniform_savgol(
        timestamps[good_idxs], signal_noisy_w_nans[good_idxs], window=window, polynom=order)
    signal_smooth_w_nans = np.copy(signal_noisy_w_nans)
    signal_smooth_w_nans[good_idxs] = signal_smooth_nonans
    # interpolate nan points
    interpolater = interp1d(
        timestamps[good_idxs], signal_smooth_nonans, kind=interp_kind, fill_value='extrapolate')
    diam = interpolater(timestamps)
    return diam


def non_uniform_savgol(x, y, window, polynom):
    """
    Applies a Savitzky-Golay filter to y with non-uniform spacing
    as defined in x
    This is based on https://dsp.stackexchange.com/questions/1676/savitzky-golay-smoothing-filter-for-not-equally-spaced-data
    The borders are interpolated like scipy.signal.savgol_filter would do
    https://dsp.stackexchange.com/a/64313
    Parameters
    ----------
    x : array_like
        List of floats representing the x values of the data
    y : array_like
        List of floats representing the y values. Must have same length
        as x
    window : int (odd)
        Window length of datapoints. Must be odd and smaller than x
    polynom : int
        The order of polynom used. Must be smaller than the window size
    Returns
    -------
    np.array of float
        The smoothed y values
    """
    if len(x) != len(y):
        raise ValueError('"x" and "y" must be of the same size')
    if len(x) < window:
        raise ValueError('The data size must be larger than the window size')
    if type(window) is not int:
        raise TypeError('"window" must be an integer')
    if window % 2 == 0:
        raise ValueError('The "window" must be an odd integer')
    if type(polynom) is not int:
        raise TypeError('"polynom" must be an integer')
    if polynom >= window:
        raise ValueError('"polynom" must be less than "window"')
    half_window = window // 2
    polynom += 1
    # Initialize variables
    A = np.empty((window, polynom))  # Matrix
    tA = np.empty((polynom, window))  # Transposed matrix
    t = np.empty(window)  # Local x variables
    y_smoothed = np.full(len(y), np.nan)
    # Start smoothing
    for i in range(half_window, len(x) - half_window, 1):
        # Center a window of x values on x[i]
        for j in range(0, window, 1):
            t[j] = x[i + j - half_window] - x[i]
        # Create the initial matrix A and its transposed form tA
        for j in range(0, window, 1):
            r = 1.0
            for k in range(0, polynom, 1):
                A[j, k] = r
                tA[k, j] = r
                r *= t[j]
        # Multiply the two matrices
        tAA = np.matmul(tA, A)
        # Invert the product of the matrices
        tAA = np.linalg.inv(tAA)
        # Calculate the pseudoinverse of the design matrix
        coeffs = np.matmul(tAA, tA)
        # Calculate c0 which is also the y value for y[i]
        y_smoothed[i] = 0
        for j in range(0, window, 1):
            y_smoothed[i] += coeffs[0, j] * y[i + j - half_window]
        # If at the end or beginning, store all coefficients for the polynom
        if i == half_window:
            first_coeffs = np.zeros(polynom)
            for j in range(0, window, 1):
                for k in range(polynom):
                    first_coeffs[k] += coeffs[k, j] * y[j]
        elif i == len(x) - half_window - 1:
            last_coeffs = np.zeros(polynom)
            for j in range(0, window, 1):
                for k in range(polynom):
                    last_coeffs[k] += coeffs[k, j] * y[len(y) - window + j]
    # Interpolate the result at the left border
    for i in range(0, half_window, 1):
        y_smoothed[i] = 0
        x_i = 1
        for j in range(0, polynom, 1):
            y_smoothed[i] += first_coeffs[j] * x_i
            x_i *= x[i] - x[half_window]
    # Interpolate the result at the right border
    for i in range(len(x) - half_window, len(x), 1):
        y_smoothed[i] = 0
        x_i = 1
        for j in range(0, polynom, 1):
            y_smoothed[i] += last_coeffs[j] * x_i
            x_i *= x[i] - x[-half_window - 1]
    return y_smoothed


#def align_left_right_pupil(eid):

#    timesL, XYsL = get_dlc_XYs(eid, 'left')    
#    dL=get_pupil_diameter(XYsL)
#    timesR, XYsR = get_dlc_XYs(eid, 'right') 
#    dR=get_pupil_diameter(XYsR)

#    # align time series left/right
#    interpolater = interp1d(
#        timesR,
#        np.arange(len(timesR)),
#        kind="cubic",
#        fill_value="extrapolate",)

#    idx_aligned = np.round(interpolater(timesL)).astype(np.int)
#    dRa = dR[idx_aligned]
#    
#    plt.plot(timesL,zscore(dL),label='left')
#    plt.plot(timesL,zscore(dRa),label='right')
#    plt.title('smoothed, aligned, z-scored right/left pupil diameter \n'
#             f'{eid}')  
#    plt.legend(loc='lower right')
#    plt.ylabel('pupil diameter')
#    plt.xlabel('time [sec]')



def pupil_diameter_PSTH(eid, s=None, times=None, per_trial_norm=True):

    '''
    nose speed PSTH
    eid = 'd0ea3148-948d-4817-94f8-dcaf2342bbbe' is good
    '''

    rt = 2  # duration of window
    st = -0.5  # lag of window wrt to stype 
  
    if (s is None) or (times is None):
        times, XYs = get_dlc_XYs(eid, 'left')    
        s = get_pupil_diameter(XYs)
    
    D = {}
 
    xs = np.arange(rt*60)  # number of frames 
    xs[:int(abs(st)*60)] = -1*np.array(list(reversed(xs[:int(abs(st)*60)])))
    xs[int(abs(st)*60):] = np.arange(rt*60)[1:1+len(xs[int(abs(st)*60):])]
    xs = xs /60.
    
    cols = {'stim':'r','feedback':'b','motion':'g'}
    
    
    for stype in ['stim','feedback']:#,'motion'
        # that's centered at feedback time
        d = constant_reaction_time(eid, rt, st,stype=stype) 
        
        D[stype] = []
        
        for i in d:

            start_idx = find_nearest(times,d[i][0])
            end_idx = start_idx  + rt*60
              
            if per_trial_norm:
                # per trial baseline - at start = 0
                dia = s[start_idx:end_idx]
                dia = zscore(dia,nan_policy='omit')
                dia = dia - dia[0]
            else:
                dia = s[start_idx:end_idx]
            
           
            D[stype].append(dia)

    
        MEAN = np.mean(D[stype],axis=0)
        STD = np.std(D[stype],axis=0)/np.sqrt(len(d))  

        plt.plot(xs, MEAN, label=stype, color = cols[stype])
        plt.fill_between(xs, MEAN + STD, MEAN - STD, color = cols[stype],
                         alpha=0.5)
        
    ax = plt.gca()
    ax.axvline(x=0, label='align event', linestyle = '--', c='k')
    plt.title('left cam pupil diameter PSTH')
    plt.xlabel('time [sec]')
    plt.ylabel('pupil diameter [px]') 
    plt.legend(loc='lower right')


def motion_energy_PSTH(eid):

    '''
    ME PSTH
    canonical session
    eid = '15f742e1-1043-45c9-9504-f1e8a53c1744'
    '''

    rt = 2  # duration of window
    st = -0.5  # lag of window wrt to stype 
    stype = 'stimOn_times'

    ME = {}
    one = ONE()
    trials = one.load_object(eid,'trials')
    ts = trials.intervals[0][0]
    te = trials.intervals[-1][1]

    try:
        for video_type in ['left','right','body']:
            t,m = get_ME(eid, video_type)       
            m = zscore(m,nan_policy='omit') 

            sta, end = find_nearest(t,ts), find_nearest(t,te) 
            t = t[sta:end]
            m = m[sta:end]

            ME[video_type] = [t,m]

        # align to body cam
        for video_type in ['left','right']:

            # align time series camera/neural
            interpolater = interp1d(
                ME[video_type ][0],
                np.arange(len(ME[video_type ][0])),
                kind="cubic",
                fill_value="extrapolate")

            idx_aligned = np.round(interpolater(ME['body'][0])).astype(int)
            ME[video_type] = [ME['body'][0], ME[video_type][1][idx_aligned]]

      
        D = {}
     
        fs = 30
        xs = np.arange(rt*fs)  # number of frames 
        xs = np.concatenate([-1*np.array(list(reversed(xs[:int(abs(st)*fs)]))),
                              np.arange(rt*fs)[1:1+len(xs[int(abs(st)*fs):])]])
        xs = xs /float(fs)
        
        cols = {'left':'r','right':'b','body':'g'}
            
        for video_type in ME:
            # that's centered at feedback time
            
            
            D[video_type] = []
            
            times,s = ME[video_type]

            trs = trials[stype][20:-20]    
            for i in trs:

                start_idx = int(find_nearest(times,i) + st*30)
                end_idx = int(start_idx  + rt*30)  

                D[video_type].append(s[start_idx:end_idx])

        
            MEAN = np.mean(D[video_type],axis=0)
            STD = np.std(D[video_type],axis=0)/np.sqrt(len(trs)) 
           

            plt.plot(xs, MEAN, label=video_type, 
                     color = cols[video_type], linewidth = 2)
            plt.fill_between(xs, MEAN + STD, MEAN - STD, color = cols[video_type],
                             alpha=0.2)
            
        ax = plt.gca()
        ax.axvline(x=0, label='stimOn', linestyle = '--', c='k')
        plt.title('Motion Energy PSTH')
        plt.xlabel('time [sec]')
        plt.ylabel('z-scored motion energy [a.u.]') 
        plt.legend(loc='lower right')        
        
    except:
        plt.title('No motion energy available!')


def interp_nans(y):

    def nan_helper(y):
        return np.isnan(y), lambda z: z.nonzero()[0]    
    nans, yy = nan_helper(y)
    y2 = copy.deepcopy(y)
    y2[nans] = np.interp(yy(nans), yy(~nans), y[~nans])
    return y2


def add_panel_letter(k):

    '''
    k is the number of the subplot
    '''
    L = string.ascii_uppercase[k-1]
    ax = plt.gca()
    ax.text(-0.1, 1.15, L, transform=ax.transAxes,
      fontsize=16, fontweight='bold', va='top', ha='right')


def plot_all(eid):
    matplotlib.rcParams.update({'font.size': 10})
    # report eid =  '4a45c8ba-db6f-4f11-9403-56e06a33dfa4'
 
    panels = {'plot_paw_on_imageL':plot_paw_on_imageL,
            'plot_paw_on_imageR':plot_paw_on_imageR,
            'plot_paw_on_imageB':plot_paw_on_imageB,
            'plot_wheel_position':plot_wheel_position,
            'paw_speed_PSTH':paw_speed_PSTH,
            'plot_licks':plot_licks, 
            'lick_raster':lick_raster,
            'nose_speed_PSTH':nose_speed_PSTH,
            'pupil_diameter_PSTH':pupil_diameter_PSTH,
            'motion_energy_PSTH':motion_energy_PSTH}
 
    nrows = 2
    ncols = int(np.ceil(len(panels)/2))

    plt.ioff()
  
    plt.figure(figsize=(17,10)) 
    
    k = 1
    for panel in panels:
        plt.subplot(nrows,ncols,k)  
        add_panel_letter(k)  
        try:
            panels[panel](eid)
            #continue
        except:
            ax = plt.gca() 
            plt.text(.5, .5,f'error in \n {panel}',color='r',fontweight='bold',
                     bbox=dict(facecolor='white', alpha=0.5),
                     fontsize=10, transform=ax.transAxes)
        k += 1
    

    plt.tight_layout()
    
    s2 = ' '.join(l)
    
    # print QC outcome in title and DLC task version
    one = ONE()
   
    det = one.get_details(eid, True)['extended_qc']
    p = one.eid2path(eid)
    s1 = '_'.join([str(p).split('/')[i] for i in [4,6,7,8]])
    
    dlc_qcs = [ 'time_trace_length_match',
                'trace_all_nan',
                'mean_in_bbox',
                'pupil_blocked',
                'lick_detection']
    
    raw_qcs = ['focus','position','brightness',
               'resolution','timestamps','file_headers',
               'wheel_alignment','camera_times','framerate',
               'pin_state','dropped_frames']


    qcs = ['task','behavior','videoLeft','videoRight','videoBody',
           'dlcLeft','dlcRight','dlcBody']           
      
    l = []       
    for q in qcs:    
        try:
            if det[q] == 'FAIL':

                if ('dlc' in q):
                    l.append('\n')
                    l.append(q+':'+str(det[q])+'-->') 
                    video_type = q[3:]
                    for dlc_qc in dlc_qcs:
                        w = f'_dlc{video_type}_{dlc_qc}'
                        if not det[w]:
                            l.append(w+':'+str(det[w])+',')
                elif ('video' in q): 
                    l.append('\n')
                    l.append(q+':'+str(det[q])+'-->') 
                    video_type = q[5:]
                    for raw_qc in raw_qcs:
                        w = f'_video{video_type}_{raw_qc}'
                        if type(det[w]) == bool:
                            if (not det[w]):
                                l.append(w+':'+str(det[w])+',')    
                        if type(det[w]) != bool: 
                            if not det[w][0]:
                                l.append(w+':'+str(det[w])+',')                
                              
                else:            
                    l.append(q+':'+str(det[q])+',')               
        except:
            continue
    
    s2 = ' '.join(l)
    
    ntrials = len(one.load_object(eid, 'trials')['goCue_times'])
    
    task = one.alyx.rest('tasks', 'list', session=eid, name='EphysDLC')
    
    try:
        plt.suptitle(s1+'#Trials:'+str(ntrials)+', DLC version: '
                     +str(task[0]['version'])+' \n '+s2,
                     backgroundcolor= 'white', fontsize=6)
    except:
        plt.suptitle(s1+'#Trials:'+str(ntrials)+', DLC version: ??'
                     +' \n '+s2,
                     backgroundcolor= 'white', fontsize=6)        
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])               
    #plt.savefig(f'/home/mic/reproducible_dlc/overviewJune/{eid}.png')
    #plt.savefig(f'/home/mic/reproducible_dlc/all_DLC/{s1}_{eid}.png')
    plt.savefig(f'/home/mic/reproducible_dlc/miles_QC_update/{s1}_{eid}.png')    
    plt.close()
    

def inspection():
    '''
    go through computed png images
    to report QC issues
    ''' 
    one = ONE()
    a = os.listdir('/home/mic/reproducible_dlc/overviewJune/') 
    
    for i in a:
        plt.figure(figsize=(15,10))
        eid = i.split('.')[0]
        p = one.path_from_eid(eid)
        s1 = '/'.join([str(p).split('/')[i] for i in [4,5,6,7,8]])
        print(s1)
        print(eid)
        img = mpimg.imread('/home/mic/reproducible_dlc/overviewJune/'+i)
        plt.imshow(img)
        plt.tight_layout()
        plt.show()
        input("Press Enter to continue...")
        plt.close()
        
             
        
