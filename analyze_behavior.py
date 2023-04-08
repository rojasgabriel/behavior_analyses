# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 16:02:36 2023

@author: Gabriel

Behavioral Analyses for chipmunk_solo

"""
# %% Import functions

from os import path, makedirs
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from utils import *


# %% Convert .mat files to .h5

file_names = pick_files_multi_session("chipmunk", "*.mat")
converted_files = convert_specified_behavior_sessions(file_names, overwrite=True)

# %% Align behavior to video data

# Separate sessions with bad labcams logfiles
earlyGRBsessions = ['20230126_110314',
                     '20230127_114504',
                     '20230130_121533',
                     '20230131_113044',
                     '20230201_131437',
                     '20230201_131715',
                     '20230202_123237',
                     '20230202_123648',
                     '20230203_133526',
                     '20230207_153632',
                     '20230208_113843',
                     '20230209_115759',
                     '20230210_132751',
                     '20230213_132401',
                     '20230217_132430',
                     '20230221_112843']

camlog_file_names = pick_files_multi_session("chipmunk", "*.camlog", "BackStereo") #should rename this to camlog file names

# Align data
for camlog_file in camlog_file_names:
    if str.split(camlog_file, '/')[5] in earlyGRBsessions:
        t = str.split(camlog_file, '/')[5]
        print('------------------------------')
        print(f'Using early sessions alignment function for {t}.')
        try:
            align_behavioral_video_earlyGRBsessions(camlog_file)
        except:
            print("There was an issue with the camlog file for the current early session:", camlog_file)
            print("Continuing to the next one...")
            continue
    else:
        t = str.split(camlog_file, '/')[5]
        print('------------------------------')
        print(f'Using default alignment function for {t}.')
        try:
            align_behavioral_video(camlog_file)
        except:
            print("There was an issue with the camlog file for the current session:", camlog_file)
            print("Continuing to the next one...")
            continue

# %% Get aligned data file paths into a list

sessions = []
for file in camlog_file_names:
    sessions.append(str.split(file, '/')[5]) # could vary by operating systems

aligned_data_paths = []
count = 0
for file in camlog_file_names:
    analysis_folder = path.join(path.join(*str.split(file, '/')[0:6], 'analysis/')) .replace('home', '/home')
    if path.exists(analysis_folder):
        # get aligned data
        aligned_file = glob(analysis_folder + '*_video_alignment.npy')
        aligned_data_paths.append(aligned_file[0])
        print(f'Aligned data for session {sessions[count]} added to aligned_data_paths list.')
        print('------------------------------')
        count+=1
    else:
        print(f'No video aligned trial data found for {sessions[count]}')
        print('------------------------------')
        count+=1
        
#%% Sort variables of interest

# Load trial data 
trialdata = [pd.read_hdf(glob(path.split(file)[0] + "/*.h5")[0]) for file in file_names]
animalID = str.split(file, '/')[4]

# Get wait times
wait_times = [np.array(session['waitTime']) for session in trialdata]
actual_wait_times = [np.array(session['actual_wait_time']) for session in trialdata]

wait_time_diff = []
for array1, array2 in zip(wait_times, actual_wait_times):
    wait_time_diff.append(array2-array1)

# Define a list of colors that follows a gradient from red to blue based on the number of days
colors = plt.cm.get_cmap('RdYlBu', len(wait_time_diff))

# Set up the plot
fig, ax = plt.subplots()

# Iterate over the data and plot each histogram separately, assigning a color based on the day
for i, arr in enumerate(filter(lambda x: wait_times[x][0] != 0, range(len(wait_times)))):
    color = colors(i)
    ax.hist(wait_time_diff[arr], bins=20, alpha=0.4, color=color, label=f"Day {i+1}")

# Add figure details
ax.legend(fontsize=6)
ax.set_xlabel('Wait time difference (s)')
ax.text(0.5, -0.19, '(observed wait time-required wait time)', transform=ax.transAxes, ha='center', fontsize=8)
ax.set_ylabel('# of trials')
ax.set_title(f'Wait times across sessions for {animalID}')

# Create a ScalarMappable object for the colorbar
sm = plt.cm.ScalarMappable(cmap=colors)
sm.set_array(range(len(wait_time_diff)))

fig.tight_layout()
fig.show()

#%% Save figures

figures_dir = '/home/gabriel/figures-code/chipmunk_behavior/figures/'
fig.savefig(f'{figures_dir}{animalID}_wait_time_difference.pdf', dpi=300)


    

    
    
    
    
    
    
    
    
    
    
    
    
