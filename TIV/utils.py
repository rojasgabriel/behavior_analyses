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
    

def load_experiment_data_joao(analysis_folder, task_folder): #specific to Joao's experiment output
    with h5.File(glob(pjoin(analysis_folder,'*.mptracker.h5'))[0],'r') as pupildata:
        diam = pupildata['diameter'][:]
    camlog,comm = parse_cam_log(glob(pjoin(task_folder,'*.camlog'))[0])
    sync = unpackbits(np.array(camlog['var2']))[0][2]
    trialdata = pd.read_hdf(glob(pjoin(task_folder,'*triallog.h5'))[0])

    trial_onsets = np.array([r.task_start_time + r.task_states[2][-1] for i,r in trialdata.iterrows()])
    
    #get the frametimes syncronized with the trial onset.
    camtime = interp1d(sync[:len(trial_onsets)],trial_onsets+0.3, fill_value = 'extrapolate')(np.arange(len(diam)))

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
                      'pupil_diameter':diam,
                      'frame_times':camtime})
    
    return trialdata


def moving_average(a, n=50): #n = window to average across
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n