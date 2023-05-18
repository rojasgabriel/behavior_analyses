from utils import get_filepath, load_dlc_data, load_data_from_droplets, get_frame_times
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

# Reading lateral view data
lateral_bpts, lateral_dlc_coords_x, lateral_dlc_coords_y, lateral_dlc_coords_likelihood = load_dlc_data(lateral_data_path[0])
del lateral_data_path

# Reading bottom view data
bottom_bpts, bottom_dlc_coords_x, bottom_dlc_coords_y, bottom_dlc_coords_likelihood = load_dlc_data(bottom_data_path[0])
del bottom_data_path

#Load trial data and check the frame rate
sessionfolder = pjoin(data_folder,params['subject'],params['session'])
trialdata,camlog,camcomm,camtime,eyedata = load_data_from_droplets(sessionfolder)

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

#Get frame times
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

# if trialdata['response'][itrial] == 0:
#     no_choice_trials[itrial] = 1
# else:
#     no_choice_trials[itrial] = 0