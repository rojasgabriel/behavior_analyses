from utils import get_filepath, load_dlc_data, load_data_from_droplets, get_frame_times, visualize_frame_rate, visualize_eye_data, visualize_frame_times, analyze_pupil_and_performance
import os
from os.path import join as pjoin
import matplotlib.pyplot as plt
import numpy as np

#Define basic data lookup parameters
params = dict({'subject':'JC099', 
               'session':'20221116_123718',
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
visualize_frame_rate(camtime, trialdata, params)

#Visualize eye data
visualize_eye_data(eyedata)

#Get frame times and visualze frame-related variables
trial_frame_times = get_frame_times(camtime, trialdata)
visualize_frame_times(trial_frame_times)

#Analyze pupil data by trial
analyze_pupil_and_performance(trialdata, eyedata, trial_frame_times)