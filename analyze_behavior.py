# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 16:02:36 2023

@author: Gabriel

Behavioral Analyses for chipmunk_solo

"""
# %% Import functions

import sys
# sys.path.insert(0, r'C:\Users\Anne\chiCa')  # wherever the parent directory for chiCa lives. just need to run it once
from chiCa import *


# %% Convert .mat files to .h5

file_names = pick_files_multi_session("chipmunk", "*.mat")
converted_files = convert_specified_behavior_sessions(file_names)

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

file_names = pick_files_multi_session("chipmunk", "*.camlog", "BackStereo")

# Align data
for camlog_file in file_names:
    if str.split(camlog_file, '\\')[5] in earlyGRBsessions:
        t = str.split(camlog_file, '\\')[5]
        print('------------------------------')
        print(f'Using early sessions alignment function for {t}.')
        try:
            align_behavioral_video_earlyGRBsessions(camlog_file)
        except:
            print("There was an issue with the camlog file for the current early session:", camlog_file)
            print("Continuing to the next one...")
            continue
    else:
        t = str.split(camlog_file, '\\')[5]
        print('------------------------------')
        print(f'Using default alignment function for {t}.')
        try:
            align_behavioral_video(camlog_file)
        except:
            print("There was an issue with the camlog file for the current session:", camlog_file)
            print("Continuing to the next one...")
            continue


