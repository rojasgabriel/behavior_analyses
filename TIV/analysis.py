from utils import *
import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt

#Define basic data lookup parameters
params = dict({'subject':'JC111', 
               'session':'20230324_131923',
               'subfolders':'*',
               'extension':'.h5'})

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

#Making sure there's no empty lists to avoid exceptions later on
lateral_data_path = lateral_data_path or [None]
bottom_data_path = bottom_data_path or [None]

# Reading lateral view data
lateral_bpts, lateral_dlc_coords_x, lateral_dlc_coords_y, lateral_dlc_coords_likelihood = load_dlc_data(lateral_data_path[0])

# Reading bottom view data
bottom_bpts, bottom_dlc_coords_x, bottom_dlc_coords_y, bottom_dlc_coords_likelihood = load_dlc_data(bottom_data_path[0])

#Load trial data
analysis_folders = [os.path.dirname(r) for r in glob(r'C:/Users/Anne/data/JC111/*/*/*.mptracker.h5')]
task_folders = [os.path.dirname(r) for r in glob(r'C:/Users/Anne/data/JC111/*/*/*.triallog.h5')]

for analysis_folder, task_folder in zip(analysis_folders, task_folders):
    if not len(analysis_folders) == 1:
        sessiondata = []
        sessiondata.append(trialdata)
    trialdata = load_experiment_data_joao(analysis_folder, task_folder)

plt.plot(moving_average(trialdata['rewarded']))