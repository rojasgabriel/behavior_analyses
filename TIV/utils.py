import os
from os.path import join as pjoin
from glob import glob
import pandas as pd
import numpy as np
from labcams import parse_cam_log, unpackbits  # pip install labcams if it doesn't work
import h5py as h5
from scipy.interpolate import interp1d
from scipy import stats

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
                      'frame_times':camtime})
    
    no_choice_trials = np.zeros(shape=(len(trialdata['trial_start']),1))
    for itrial,t in enumerate(trialdata['trial_start']):
        if trialdata['response'][itrial] == 0:
            no_choice_trials[itrial] = 1
        else:
            no_choice_trials[itrial] = 0


    return trialdata,camlog,camcomm,camtime,eyedata,no_choice_trials

def get_frame_times(camtime, trialdata):
    frame_rate = np.mean(1./np.diff(camtime))

    trial_start_frames = np.zeros(shape=(len(trialdata['trial_start']),1))
    stim_onset_frames = np.zeros(shape=(len(trialdata['stim_onset']),1))
    response_onset_frames = np.zeros(shape=(len(trialdata['stim_onset']),1))
    reward_onset_frames = np.zeros(shape=(len(trialdata['reward_time']),1))
    trial_end_frames = np.zeros(shape=(len(trialdata['trial_start']),1))
    trial_frame_length = np.zeros(shape=(len(trialdata['trial_start']),1))
    for itrial,t in enumerate(trialdata['trial_start']):
        if itrial == len(trialdata['trial_start'])-1:
            continue
        trial_start_frames[itrial] = np.floor(trialdata['trial_start'][itrial]*frame_rate)
        stim_onset_frames[itrial] = np.floor(trialdata['stim_onset'][itrial]*frame_rate)
        response_onset_frames[itrial] = np.floor(trialdata['stim_onset'][itrial]*frame_rate+frame_rate) #response period begins 1s after stim onset
        reward_onset_frames[itrial] = np.floor(trialdata['reward_time'][itrial]*frame_rate)
        trial_duration = trialdata['trial_start'][itrial+1] - trialdata['trial_start'][itrial]
        trial_frame_length[itrial] = np.floor(trial_duration*frame_rate)
        trial_end_frames[itrial] = np.floor((trialdata['trial_start'][itrial]*frame_rate)+trial_frame_length[itrial])

    trial_frame_times = dict({'trial_start_frames':trial_start_frames,
                              'stim_onset_frames':stim_onset_frames,
                              'response_onset_frames':response_onset_frames,
                              'reward_onset_frames':reward_onset_frames,
                              'trial_end_frames':trial_end_frames,
                              'trial_frame_length':trial_frame_length})

    return trial_frame_times

def visualize_frame_times(trial_frame_times):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))

    plt.subplot(3, 2, 1)
    plt.plot(trial_frame_times['trial_start_frames'])
    plt.title('Trial Start Frames')

    plt.subplot(3, 2, 2)
    plt.plot(trial_frame_times['stim_onset_frames'])
    plt.title('Stim Onset Frames')

    plt.subplot(3, 2, 3)
    plt.plot(trial_frame_times['response_onset_frames'])
    plt.title('Response Onset Frames')

    plt.subplot(3, 2, 4)
    plt.plot(trial_frame_times['reward_onset_frames'])
    plt.title('Reward Onset Frames')

    plt.subplot(3, 2, 5)
    plt.plot(trial_frame_times['trial_end_frames'])
    plt.title('Trial End Frames')

    plt.subplot(3, 2, 6)
    plt.plot(trial_frame_times['trial_frame_length'])
    plt.title('Trial Frame Length')

    plt.tight_layout()


def visualize_eye_data(eyedata):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(4, 5))
    axes[0].plot(eyedata['diameter'], label='diameter', color = 'r')
    axes[0].legend()
    axes[0].set_xlabel('Frames')
    axes[1].plot(eyedata['elevation'], label='elevation', color = 'b')
    axes[1].legend()
    axes[1].set_xlabel('Frames')
    axes[2].plot(eyedata['azimuth'], label='azimuth', color = 'k')
    axes[2].legend()
    axes[2].set_xlabel('Frames')
    plt.tight_layout()


def visualize_frame_rate(camtime, trialdata, params):
    import matplotlib.pyplot as plt
    import numpy as np
    print(len(camtime),np.mean(1./np.diff(camtime)))
    plt.figure(figsize=[10,2])
    plt.plot(camtime[:-1],1./np.diff(camtime),
                    'o',alpha = 0.5,clip_on = False,label='camera rate')
    plt.vlines(trialdata['trial_start'],25,30,color='r',lw = 0.5,label='trial start')
    plt.legend()
    plt.ylabel('Frame rate')
    plt.xlabel('Time from camera start')
    plt.ylim([0,np.max(plt.ylim())])
    plt.title(f"{params['subject']} - {params['session']}")
    plt.tight_layout()

def get_pupil_diameters_per_trial(trialdata, eyedata, trial_frame_times):
    from scipy import stats
    m = stats.mode(trial_frame_times['trial_frame_length'])

    # trial_pupil_diameters = np.zeros(shape=(len(trialdata['trial_start']), int(m.mode[0][0])))
    trial_pupil_diameters = []
    for itrial,t in enumerate(trialdata['trial_start']):
        if itrial == len(trialdata['trial_start'])-1:
            continue
        trial_pupil_diameters.append(eyedata['diameter'][int(trial_frame_times['trial_start_frames'][itrial]):int(trial_frame_times['trial_end_frames'][itrial])])
        trial_pupil_diameters[itrial] = trial_pupil_diameters[itrial][:int(m.mode[0][0])]

    return trial_pupil_diameters

def analyze_pupil_and_performance(trialdata, eyedata, trial_frame_times):
    import numpy as np
    import matplotlib.pyplot as plt

    trial_pupil_diameters = get_pupil_diameters_per_trial(trialdata, eyedata, trial_frame_times)
    trial_pupil_diameters_mean = np.array([np.nanmean(trial_pupil_diameters[i]) for i in range(len(trial_pupil_diameters))])
    convolved_pupil_diameters = np.convolve(trial_pupil_diameters_mean, np.ones(25)/25, mode='valid')
    convolved_rewarded = np.convolve(trialdata['rewarded'], np.ones(25)/25, mode='valid')

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.plot(convolved_pupil_diameters, label='Pupil', color='b')
    ax1.set_xlabel('Trials')
    ax1.set_ylabel('Pupil diameter')
    ax1.yaxis.label.set_color('b')
    ax1.tick_params(axis='y', colors='b')
    ax1.spines['left'].set_color('b')

    ax2.plot(convolved_rewarded, label='Performance', color='r')
    ax2.set_ylabel('Performance')
    ax2.yaxis.label.set_color('r')
    ax2.tick_params(axis='y', colors='r')
    ax2.spines['right'].set_color('r')

    plt.title(f"r^2 = {np.corrcoef(trial_pupil_diameters_mean[:min(len(trial_pupil_diameters_mean), len(trialdata['rewarded']))], trialdata['rewarded'][:min(len(trial_pupil_diameters_mean), len(trialdata['rewarded']))])[0,1]**2}")

    plt.tight_layout()