from utils import get_filepath, load_dlc_data, load_data_from_droplets, get_frame_times, get_pupil_diameters_per_trial
import os
from os.path import join as pjoin
import matplotlib.pyplot as plt
import numpy as np

#Define basic data lookup parameters
params = dict({'subject':'JC111', 
               'session':'20230324_131923',
               'subfolders':'*',
               'extension':'.h5'})

data_folder = pjoin(os.path.expanduser('~'),'data')

data_paths = get_filepath(subject = params['subject'], 
                          session = params['session'], 
                          subfolders = 'dlc_analysis', 
                          extension = params['extension'])

if type(data_paths) is str: #check if only one data_path is returned. convert to list if so
    tmp = []
    tmp.append(data_paths)
    data_paths = tmp
    del tmp

#Load DLC data
lateral_data_path=[]
bottom_data_path=[]
for path in data_paths:
    if 'cam0' in path: #check if it's a lateral or bottom view video
        lateral_data_path.append(path)
    elif 'cam1' in path:
        bottom_data_path.append(path)
del path

#Making sure there's no empty lists to avoid exceptions later on
lateral_data_path = lateral_data_path or [None]
bottom_data_path = bottom_data_path or [None]

#Reading lateral view data
lateral_bpts, lateral_dlc_coords_x, lateral_dlc_coords_y, lateral_dlc_coords_likelihood = load_dlc_data(lateral_data_path[0])
del lateral_data_path

#Reading bottom view data
bottom_bpts, bottom_dlc_coords_x, bottom_dlc_coords_y, bottom_dlc_coords_likelihood = load_dlc_data(bottom_data_path[0])
del bottom_data_path

#Load trial data and check video frame rate
sessionfolder = pjoin(data_folder,params['subject'],params['session'])
trialdata,camlog,camcomm,camtime,eyedata,no_choice_trials = load_data_from_droplets(sessionfolder)

#Visualize framerate
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

#Visualize eye data
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

#Get frame times and visualze frame-related variables
trial_frame_times = get_frame_times(camtime, trialdata)

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
plt.show()

#Get pupil data by trial
trial_pupil_diameters = get_pupil_diameters_per_trial(trialdata, eyedata, trial_frame_times)
#Trial pupil diameters are stored in a list of arrays, one array per trial
#Each array contains the pupil diameters for each frame of the trial
#Calculate the mean of each array to get the mean pupil diameter for each trial and ignore nans
#If the mean value is more than 3 standard deviations away from the mean of the whole session, ignore it
#Store the mean for each trial in an array called trial_pupil_diameters_mean
#Plot trial_pupil_diameters_mean using a moving average of 50 to smooth the data
trial_pupil_diameters_mean = np.array([np.nanmean(trial_pupil_diameters[i]) for i in range(len(trial_pupil_diameters))])
trial_pupil_diameters_mean = trial_pupil_diameters_mean[np.abs(trial_pupil_diameters_mean - np.nanmean(trial_pupil_diameters_mean)) < 3*np.nanstd(trial_pupil_diameters_mean)]
plt.plot(np.convolve(trial_pupil_diameters_mean, np.ones(50)/50, mode='valid'))
#Use a different y-axis for plotting trialdata['rewarded'] using a moving average of 50 to smooth the data
plt.twinx()
plt.plot(np.convolve(trialdata['rewarded'], np.ones(50)/50, mode='valid'), 'r')
#Calculate the correlation between the two signals and plot the r squared value
#If the size of the two signals is different, use the smaller size for the correlation calculation
plt.title(f"r^2 = {np.corrcoef(trial_pupil_diameters_mean[:min(len(trial_pupil_diameters_mean), len(trialdata['rewarded']))], trialdata['rewarded'][:min(len(trial_pupil_diameters_mean), len(trialdata['rewarded']))])[0,1]**2}")
plt.show()
