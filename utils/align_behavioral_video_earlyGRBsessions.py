# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 17:43:26 2023

@author: Lukas (modified by Gabriel)
"""

def align_behavioral_video_earlyGRBsessions(camlog_file):
    '''Function to extract video tracking information. Returns the name of the camera from the log file,
    the average interval between acquired frames and the frame indices of the trial starts.
     
    Parameters
    ----------
    camlog_file: The log file from the labcams acquisition. Make sure the data is stored in the specified folder structure
                 that is, the chipmunk folder contains the camlog file and the .mat or .obsmat file from the behavior.
    
    Retunrs
    -------
    video_alignment_data: dict with keys: camera_name (the labbel of the camera
                          perspective acquired), trial_starts (the frame, during
                          which the trial start), frame_interval (average interval
                          between frames, calculated from the entire timeseries)
                        
    Examples
    --------
    video_alignment_data = align_behavioral_video(camlog_file)
    
    TO-DO: ask user if they want to overwrite video alignment file when saving at the end.
    '''
    
    import numpy as np
    from labcams import parse_cam_log #Retrieve the correct channel and spot trial starts in frames
    from scipy.io import loadmat #Load behavioral data for ground truth on trial number
    from os import path, makedirs
    from glob import glob
    
    #First load the associated chipmunk file and get the number of trials to find the correct channel on the camera
    chipmunk_folder = path.split(camlog_file)[0] #Split the path to retrieve the folder only in order to load the mat or obsmat file belonging to the movies.
    file_list = glob(chipmunk_folder + '/*.mat') #There is only one behavioral recording file in the folder
    if not file_list:
        file_list = glob(chipmunk_folder + '/*.obsmat')
        if not file_list:
            raise ValueError('No behavioral event file was found in the folder.')
            
    chipmunk_file = file_list[0]
    sesdata = loadmat(chipmunk_file, squeeze_me=True,
                                      struct_as_record=True)['SessionData']
    trial_num = int(sesdata['nTrials'])
    
    #Extract data from the log file
    logdata, comments = parse_cam_log(camlog_file) #Parse inputs and separate comments from frame data
    
    #Get the camera name
    spaces = [pos for pos, char in enumerate(comments[0]) if char == ' '] #Find all the empty spaces that separate the words
    camera_name = comments[0][spaces[1]+1: spaces[2]] #The name comes between the second and the third space
    
    #Get the video frame interval in s
    video_frame_interval = np.mean(np.diff(logdata['timestamp'])) #Compute average time between frames
    
    #Find trial start frames
    ts = []
    a = []
    for l in comments:
        l = l.strip('#')
    
        if 'trial_start' in l:
            a = [int(l.split(':')[-1]),int(l.split(',')[0])]
        if 'trial_end' in l:
            a.append(int(l.split(',')[0]))
            ts.append(a)   
    tmp = np.stack(ts)
    
    #Check if trial numbers between Bpod and labcams are the same
    if tmp.shape[0] == trial_num + 1:
        pass
    elif tmp.shape[0] == trial_num:
        pass
    else:
        raise ValueError('The trial numbers between the Bpod output and the labcams logfile do not match. Please check the log files and camera setup.')
    
    trial_start_video_frames = tmp[:,1]
    
    #Create video alignment data structure to be saved         
    video_alignment_data = dict({'camera_name': camera_name, 'trial_starts': trial_start_video_frames, 'frame_interval': video_frame_interval})
    
    #Check if the proper directory exists to store the data
    directory = path.join(path.split(camlog_file)[0],'..', 'analysis')
    
    if not path.exists(directory):
        makedirs(directory)
        print(f"Directory {directory} created")
    else:
        print(f"Directory {directory} already exists")

    #Save data
    output_name = path.splitext(path.split(camlog_file)[1])[0]
    np.save(path.join(path.split(camlog_file)[0],'..', 'analysis', output_name + '_video_alignment.npy'), video_alignment_data)
    print(f'Video alignment file created for {camlog_file}!')  
    
    return video_alignment_data