# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 16:02:36 2023

@author: Gabriel

Behavioral Analyses for chipmunk_solo

"""
#%% Import functions

# import sys
# sys.path.insert(0, '~/chiCa')
from chiCa.chipmunk_analysis_tools import *


#%% Convert .mat files to .h5

file_names = pick_files_multi_session("chipmunk", "*.mat")
converted_files = convert_specified_behavior_sessions(file_names)


#%% Align behavior to video data

file_names = pick_files_multi_session("chipmunk", "*.camlog", "BackStereo")
for camlog_file in file_names:
    try:
        align_behavioral_video(camlog_file)
    except:
        print("There was an issue with the camlog file for the current session:", camlog_file)
        break
        # print("Continuing to the next one...")
        # continue
    
    
#%% Align trial data to frame times 

# this is necessary because I screwed up the logfile from labcams for the first couple of sessions 
# this would normally be done by the align_behavioral_video function in the previous section

from labcams import parse_cam_log
import numpy as np
import pandas as pd

log,comm = parse_cam_log('c:/Users/Anne/data/GRB001/20230208_113843/chipmunk/GRB001_20230208_113843_chipmunk_DemonstratorAudiTask_FrontStereoView_00000000.camlog')

ts = []
a = []
for l in comm:
    l = l.strip('#')
    
    if 'trial_start' in l:
        a = [int(l.split(':')[-1]),int(l.split(',')[0])]
    if 'trial_end' in l:
        a.append(int(l.split(',')[0]))
        ts.append(a)
        
Front_times = pd.DataFrame(np.stack(ts),columns = ['itrial','onset_frame','offset_frame'])