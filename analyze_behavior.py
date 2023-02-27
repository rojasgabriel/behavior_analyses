# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 16:02:36 2023

@author: Gabriel

Behavioral Analyses for chipmunk_solo

"""
# %% Import functions

import sys
sys.path.insert(0, '/Users/gabriel')  # wherever the parent directory for chiCa lives. just need to run it once
from chiCa import *
import os

# %% Convert .mat files to .h5

file_names = pick_files_multi_session("chipmunk", "*.mat")
converted_files = convert_specified_behavior_sessions(file_names)

# %% Align behavior to video data

# Separate sessions with bad labcams logfiles
earlyGRBsessions = [session for session in sorted(os.listdir('/Users/gabriel/data/GRB001'))
                    if not session.startswith('.DS_Store') and '20230223' not in session and '20230224' not in session]

file_names = pick_files_multi_session("chipmunk", "*.camlog", "BackStereo")

for camlog_file in file_names:
    if str.split(camlog_file, '/')[5] in earlyGRBsessions:
        print('Using modified alignment function for early GRB sessions.')
        try:
            print(early)
            break
            align_behavioral_video_earlyGRBsessions(camlog_file)
        except:
            print("There was an issue with the camlog file for the current early session:", camlog_file)
            print("Continuing to the next one...")
            continue
    else:
        print('Using default alignment function.')
        try:
            print('normal')
            break
            align_behavioral_video(camlog_file)
        except:
            print("There was an issue with the camlog file for the current session:", camlog_file)
            print("Continuing to the next one...")
            continue

# %% Align trial data to frame times

# this is necessary because I screwed up the logfile from labcams for the first couple of sessions 
# this would normally be done by the align_behavioral_video function in the previous section

from labcams import parse_cam_log
import numpy as np
import pandas as pd

log, comm = parse_cam_log(
    'c:/Users/Anne/data/GRB001/20230208_113843/chipmunk/GRB001_20230208_113843_chipmunk_DemonstratorAudiTask_FrontStereoView_00000000.camlog')

# spaces = [pos for pos, char in enumerate(comments[0]) if char == ' '] #Find all the empty spaces that separate the words
# camera_name = comments[0][spaces[1]+1: spaces[2]] #The name comes between the second and the third space

ts = []
a = []
for l in comm:
    l = l.strip('#')

    if 'trial_start' in l:
        a = [int(l.split(':')[-1]), int(l.split(',')[0])]  # first give me trial num, second gives me frame
    if 'trial_end' in l:
        a.append(int(l.split(',')[0]))
        ts.append(a)

Front_times = pd.DataFrame(np.stack(ts), columns=['itrial', 'onset_frame', 'offset_frame'])
