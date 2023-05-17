import os
from os.path import join as pjoin
from glob import glob
import pandas as pd
import numpy as np
from labcams import parse_cam_log, unpackbits  # pip install labcams if it doesn't work
import h5py as h5
from scipy.interpolate import interp1d

default_preferences = {'datapath':[pjoin(os.path.expanduser('~'),'data')]}

def get_filepath(subject,
                 session,
                 subfolders,
                 datapath = None,
                 filename = '*', 
                 extension = ''):
    
    if datapath is None:
        datapath = default_preferences['datapath'][0]

    files = glob(pjoin(datapath, subject, session, subfolders, filename+extension))

    if len(files) == 1:
        files = files[0]
    if not len(files):
        files = None

    return files


def load_dlc_data(data_path):
    try:
        data = pd.read_hdf(data_path)
        bpts = np.array(data.columns.get_level_values("bodyparts").values[::3])
        dlc_coords_x, dlc_coords_y, dlc_coords_likelihood = data.values.reshape((len(data), -1, 3)).T
        return bpts, dlc_coords_x, dlc_coords_y, dlc_coords_likelihood
    except:
        return None, None, None, None
    

def get_times_for_state_onset(log, state = 'stim'):
    statetimes = []
    for i,l in log.iterrows():
        t = np.nan
        a = l['task_states']
        for s in a: # for each state
            if state in s:
                t = s[-1]
        statetimes.append(t)
    return np.array(statetimes)

def load_data_from_droplets(sessionfolder, 
                            dropletsfolder = 'DropletsTask', 
                            read_riglog = False):
    
    eyedata = {}
    camgpio = []
    mptrackerfile = glob(pjoin(sessionfolder,'*','*.mptracker.h5'))
    if len(mptrackerfile):
        with h5.File(mptrackerfile[0],'r') as pupildata:
            for k in pupildata.keys():
                eyedata[k] = pupildata[k][:]
    
    from labcams import parse_cam_log
    camlog,camcomm = parse_cam_log(glob(pjoin(sessionfolder,dropletsfolder,'*.camlog'))[0])

    camsyncmethod = 'trial_init'
    if 'var2' in camlog.columns:
        camgpio = unpackbits(np.array(camlog['var2'])) # the onsets
    
    # check the if the number of cam sync pulses is close to the number of trials if not the cable was not well connected
    trialdata = pd.read_hdf(glob(pjoin(sessionfolder,dropletsfolder,'*triallog.h5'))[0])
    if len(camgpio):
        if np.abs(len(camgpio) - len(trialdata))<5: # only if recorded most trials on the camera
            if 2 in camgpio[0].keys():
                gpiosync = camgpio[0][2]
                # check here if the trialstart was connected or it was the stim 
                if np.mean(camgpio[1][2]-camgpio[0][2]) > 15:
                    camsyncmethod='stim'
                    #print('Using the stim command to sync {0}.'.format(sessionfolder))
    # otherwise use the network command
    if not 'sync' in locals():
        # then use the network command
        sync = []
        for c in camcomm:
            if 'trial_start' in c:
                sync.append(int(c.strip('# ').split(',')[0]))
        camsyncmethod = 'trial_init'
        print('Using the network command to sync {0}.'.format(sessionfolder))
    trial_onsets = get_times_for_state_onset(trialdata,camsyncmethod)
    if camsyncmethod == 'trial_init':
        trial_onsets[:] = 0.
    trial_onsets += np.array(trialdata.task_start_time)  # add the start of each trial
    # get the frametimes syncronized with the trial onset.
    nframes = camlog.frame_id.iloc[-1]+1 # frame_id starts at zero
    if 'diameter' in eyedata.keys(): # use the size of diameter if available
        nframes = len(eyedata['diameter'])
    camtime = interp1d(sync[:len(trial_onsets)],trial_onsets, fill_value = 'extrapolate')(np.arange(nframes))

    rewarded_side = np.array(trialdata.rewarded_side == 'left',dtype = int)
    rewarded_side[rewarded_side==0] = 2

    trial_start = np.array(trialdata.task_start_time)
    stim_onset = np.array([r.task_start_time + r.task_states[2][-1] for i,r in trialdata.iterrows()])
    stim_duration = np.array(trialdata.stim_duration)
    response = np.array(trialdata.response)
    response[response==-1] = 2
    reward_time = np.array([r.task_start_time + r.task_states[4][-1]  if r.rewarded else np.nan for i,r in trialdata.iterrows()])    
    rewarded = np.array(trialdata.rewarded)
    response_time = np.array(trialdata.response_time)
    timeout_time = np.array([r.task_start_time + r.task_states[5][-1]  if ((not r.rewarded) and (r.response != 0)) else np.nan for i,r in trialdata.iterrows()])
    stim_intensity = np.array(trialdata.stim_intensity)
    stim_rates = np.array(trialdata.stim_rates)
    
    trialdata = dict({'trial_start':trial_start,
                      'stim_onset':stim_onset,
                      'stim_duration':stim_duration,
                      'response':response,
                      'reward_time':reward_time,
                      'rewarded':rewarded,
                      'response_time':response_time,
                      'timeout_time':timeout_time,
                      'stim_intensity':stim_intensity,
                      'stim_rates':stim_rates,
                      'pupil_diameter':eyedata['diameter'],
                      'frame_times':camtime})
    
    return trialdata,camlog,camcomm,camtime, eyedata


def moving_average(a, n=50): #n = window to average across
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n