from utils import get_filepath, load_dlc_data, load_data_from_droplets, get_frame_times, visualize_frame_rate, visualize_eye_data, visualize_frame_times, analyze_pupil_and_performance
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
visualize_frame_rate(camtime, trialdata, params)

#Visualize eye data
visualize_eye_data(eyedata)

#Get frame times and visualze frame-related variables
trial_frame_times = get_frame_times(camtime, trialdata)
visualize_frame_times(trial_frame_times)

#Analyze pupil data by trial
analyze_pupil_and_performance(trialdata, eyedata, trial_frame_times)

#Generate a design matrix to perform regression analysis
#the design matrix is a matrix of regressors that will be used to predict the body part coordinates
#we will use trial data to predict the body part coordinates
#the variables we will use to predict the body part coordinates are: stimulus intensity, outcome, response, previous outcome, previous response

#first, we need to create a matrix of regressors
#the matrix of regressors will have a row for each frame
#the columns will be the variables we want to use to predict the body part coordinates
#each column has an array of values, one for each trial
#the first column will be a column of ones, which will be used to calculate the intercept
#the second column will be the stimulus intensity
#the third column will be the outcome
#the fourth column will be the response
#the fifth column will be the previous outcome
#the sixth column will be the previous response
design_matrix = np.zeros((len(trial_frame_times['trial_start_frames']),6))
design_matrix[:,1] = trialdata['stim_intensity']
design_matrix[:,2] = trialdata['rewarded']
design_matrix[:,3] = trialdata['response']
design_matrix[:,4] = np.roll(trialdata['rewarded'],1)
design_matrix[:,5] = np.roll(trialdata['response'],1)

#second, we need to create a matrix of body part coordinates
#the matrix of body part coordinates will have a row for each frame
#the columns will be the x and y coordinates of the body part
#the first column will be the x coordinates of the body part
#the second column will be the y coordinates of the body part
#the body part coords must be separated by trial and by lateral and bottom 
#to separate by trial use trial_frame_times['trial_start_frames'] as an integer for trial start and trial_frame_times['trial_end_frames'] as an integer for trial end
#to separate by lateral and bottom use lateral_dlc_coords_x and bottom_dlc_coords_x for x coordinates and lateral_dlc_coords_y and bottom_dlc_coords_y for y coordinates
lateral_body_part_coords = np.zeros((len(trial_frame_times['trial_start_frames']),2))
bottom_body_part_coords = np.zeros((len(trial_frame_times['trial_start_frames']),2))
for i in range(len(trial_frame_times['trial_start_frames'])):
    lateral_body_part_coords[i,:] = np.mean(lateral_dlc_coords_x[trial_frame_times['trial_start_frames'][i]:trial_frame_times['trial_end_frames'][i],:],axis=0)
    bottom_body_part_coords[i,:] = np.mean(bottom_dlc_coords_x[trial_frame_times['trial_start_frames'][i]:trial_frame_times['trial_end_frames'][i],:],axis=0)
body_part_coords = np.concatenate((lateral_body_part_coords,bottom_body_part_coords),axis=1)


#third, we need to perform regression analysis
#we will use the design matrix to predict the body part coordinates
#we will use ordinary least squares regression
#we will use the statsmodels package to perform the regression analysis
import statsmodels.api as sm
model = sm.OLS(body_part_coords,design_matrix)
results = model.fit()
print(results.summary())

#fourth, we need to plot the results
#we will plot the predicted body part coordinates against the actual body part coordinates
import matplotlib.pyplot as plt
plt.figure()
plt.plot(body_part_coords[:,0],body_part_coords[:,1],'o',label='actual')
plt.plot(results.predict(),body_part_coords[:,1],'o',label='predicted')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

#fifth, we need to plot the r squared values
#we will plot the r squared values for each body part
#we will plot the r squared values for each body part as a function of time, averaged across trials
plt.figure()
plt.plot(results.rsquared,label='r squared')
plt.xlabel('time')
plt.ylabel('r squared')
plt.legend()
plt.show()


