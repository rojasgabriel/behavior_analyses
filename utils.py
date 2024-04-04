#%% Utils for my analyses
import numpy as np
import pandas as pd
import os
import glob

def get_file_names(animal_name, data_type, file_extension, file_keyword=None):
    '''Tool to select specified files of a data type over all sessions for a given animal.
    This relies on the hierarchical Churchland lab data folder structure with:
    animal_name -> session_datetime -> data_type
    
    Adapted from Lukas Oesch's `chipmunk_analysis_tools.py`.
       
    
    Parameters
    ----------
    animal_name: str, the name of the animal whose sessions are to be selected.
    data_type: str, the directory with the specific data type, for example chipumnk, caiman, etc.
    file_extension: str, file extension specifier, for example *.mat
    file_keyword: str, a pattern that should be detected inside the file name to
                  distinguish the desired files from other files with the same extension.
    
    Returns
    -------
    file_names: list, list of file names selected
    
    Examples
    --------
    file_names = get_file_names('GRB001', 'chipmunk', '*.h5')
    '''

    home_dir = os.path.expanduser("~")
    session_dirs = glob.glob(f"{home_dir}/data/{animal_name}/*/")
    
    file_names = []
    for session_dir in session_dirs:
        data_type_dir = os.path.join(session_dir, data_type)
        file_paths = glob.glob(os.path.join(data_type_dir, file_extension))
        for file_path in file_paths:
            if file_keyword is None or file_keyword in file_path:
                file_names.append(file_path)
    
    file_names.sort()
    
    return file_names

def get_session_trial_counts(file_names):
    """
    Calculate the number of trials in each session file.

    Parameters:
        file_names (list): A list of file paths to session data files.

    Returns:
        session_trial_counts (list): A list containing the number of trials in each session file.

    Example usage:
        file_names = ['/path/to/file1.h5', '/path/to/file2.h5', '/path/to/file3.h5']
        trial_counts = get_session_trial_counts(file_names)
        print(trial_counts)
        >>> [10, 15, 8]
    """

    session_trial_counts = []
    for file in file_names: 
        session_data = pd.read_hdf(file)
        ntrials = len(session_data)
        session_trial_counts.append(ntrials)

    return session_trial_counts

def get_filtered_session_averages_and_dates(file_names, min_stims=6):
    """
    Calculate filtered session averages and dates based on the given file names.

    Args:
        file_names (list): A list of file names.
        min_stims (int, optional): The minimum number of unique stimuli required. Defaults to 6.

    Returns:
        tuple:
            filtered_dates: A list of filtered dates.
            filtered_session_averages: A list of filtered session averages.
            stims: A list of unique stimuli.

    Example usage:
        file_names = ['/path/to/file1.hdf', '/path/to/file2.hdf', '/path/to/file3.hdf']
        dates, averages, unique_stims = get_filtered_session_averages_and_dates(file_names)
    """

    filtered_dates = []
    filtered_session_averages = []
    stims = []
    for file in file_names:
        session_data = pd.read_hdf(file)
        stim_rates = np.array([len(timestamps) for timestamps in session_data.stimulus_event_timestamps])
        unique_stims = list(np.unique(stim_rates))
        valid_trials = np.logical_or(stim_rates == 4, stim_rates == 20)
        if len(unique_stims) > min_stims:
            # allocate performance
            performance = np.array(session_data.outcome_record, dtype=float)
            performance[performance == -1] = np.nan  # setting early withdrawal trials as nans
            performance[performance == 2] = np.nan  # setting no response trials as nans
            filtered_session_averages.append(np.nanmean(performance[valid_trials]))

            # allocate date
            split_path = file.split('/')
            filtered_dates.append(split_path[5])

            # allocate unique_stims
            stims.append(unique_stims)

    return filtered_dates, filtered_session_averages, stims

def get_median_wait_time(file_names):
    """
    Calculate the median wait time from a list of file names.
    
    Parameters:
        file_names (list): A list of file names containing session data.
    
    Returns:
        list: A list of median wait times for each file.
    
    Example usage:
        file_names = ['data1.hdf', 'data2.hdf', 'data3.hdf']
        median_wait_times = get_median_wait_time(file_names)
        print(median_wait_times)
        >>> [10.5, 8.2, 12.0]
    """
    
    median_wait_times = []
    for file in file_names:
        session_data = pd.read_hdf(file)
        median_wait_times.append(np.nanmedian(session_data.waitTime))

    return median_wait_times

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size), 'valid') / window_size

def separate_axes(ax): #thanks Lukas!
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    yti = ax.get_yticks()
    yti = yti[(yti >= ax.get_ylim()[0]) & (yti <= ax.get_ylim()[1]+10**-3)] #Add a small value to cover for some very tiny added values
    ax.spines['left'].set_bounds([yti[0], yti[-1]])
    xti = ax.get_xticks()
    xti = xti[(xti >= ax.get_xlim()[0]) & (xti <= ax.get_xlim()[1]+10**-3)]
    ax.spines['bottom'].set_bounds([xti[0], xti[-1]])

    return

#%% Used for my version of the performance summaries

def categorize_trials(sessiondata, exclude=None):
    """
    Categorizes trials by response (left/right) and outcome (rewarded/unrewarded).

    Parameters:
        sessiondata (DataFrame): Loaded from the output file created with chipmunk_analysis_tools to convert .mat to .h5. 0 indicates left, 1 indicates right.
        exclude (int or list, optional): Specifies the stimulus rates that the user wants to exclude from the output data. Useful when wanting to exclude responses at chance (12 Hz in chipmunk).

    Returns:
        correct_left (list): Zeros and ones identifying the trials where the response was left and the outcome was rewarded.
        correct_right (list): Zeros and ones identifying the trials where the response was right and the outcome was rewarded.
        incorrect_left (list): Zeros and ones identifying the trials where the response was left and the outcome was unrewarded.
        incorrect_right (list): Zeros and ones identifying the trials where the response was right and the outcome was unrewarded.

    Example usage:
        sessiondata = pd.read_hdf('/path/to/sessiondata.h5')
        correct_left, correct_right, incorrect_left, incorrect_right, exclude = categorize_trials(sessiondata, exclude=[12, 24])
    """
    
    df = pd.DataFrame(sessiondata[['response_side', 'correct_side', 'outcome_record']])
    df = df.dropna()
    df = df.reset_index(drop=True)

    sel = sessiondata[sessiondata.response_side.isin([0,1])] # select only trials where the subject responded
    sel_stim_rates = np.array([len(timestamps) for timestamps in sel.stimulus_event_timestamps])

    if exclude is not None:
        correct_left = [1 if (df.response_side[i] == 0) & (df.outcome_record[i] == 1) & (not any(sel_stim_rates[i] == exclude)) else 0 for i,t in enumerate(df.response_side)]
        correct_right = [1 if (df.response_side[i] == 1) & (df.outcome_record[i] == 1) & (not any(sel_stim_rates[i] == exclude)) else 0 for i,t in enumerate(df.response_side)]
        incorrect_left = [1 if (df.response_side[i] == 0) & (df.outcome_record[i] == 0) & (not any(sel_stim_rates[i] == exclude)) else 0 for i,t in enumerate(df.response_side)]
        incorrect_right = [1 if (df.response_side[i] == 1) & (df.outcome_record[i] == 0) & (not any(sel_stim_rates[i] == exclude)) else 0 for i,t in enumerate(df.response_side)]
    else:
        correct_left = [1 if (df.response_side[i] == 0) & (df.outcome_record[i] == 1) else 0 for i,t in enumerate(df.response_side)]
        correct_right = [1 if (df.response_side[i] == 1) & (df.outcome_record[i] == 1) else 0 for i,t in enumerate(df.response_side)]
        incorrect_left = [1 if (df.response_side[i] == 0) & (df.outcome_record[i] == 0) else 0 for i,t in enumerate(df.response_side)]
        incorrect_right = [1 if (df.response_side[i] == 1) & (df.outcome_record[i] == 0) else 0 for i,t in enumerate(df.response_side)]

    return correct_left, correct_right, incorrect_left, incorrect_right

def allocate_response_times(response_times, correct_left, correct_right, incorrect_left, incorrect_right):
    """
    Allocates response times to different categories based on trial outcomes.

    Parameters:
        response_times (list): List of response times for each trial.
        correct_left (list): Zeros and ones identifying the trials where the response was left and the outcome was rewarded.
        correct_right (list): Zeros and ones identifying the trials where the response was right and the outcome was rewarded.
        incorrect_left (list): Zeros and ones identifying the trials where the response was left and the outcome was unrewarded.
        incorrect_right (list): Zeros and ones identifying the trials where the response was right and the outcome was unrewarded.

    Returns:
        tuple:
            correct_left_response_times (list): List of response times for correct left trials.
            correct_right_response_times (list): List of response times for correct right trials.
            incorrect_left_response_times (list): List of response times for incorrect left trials.
            incorrect_right_response_times (list): List of response times for incorrect right trials.

    Example usage:
        response_times = [0.5, 0.6, 0.7, 0.8, 0.9]
        correct_left = [1, 0, 0, 1, 0]
        correct_right = [0, 1, 0, 0, 1]
        incorrect_left = [0, 0, 1, 0, 0]
        incorrect_right = [0, 0, 0, 1, 0]
        correct_left_response_times, correct_right_response_times, incorrect_left_response_times, incorrect_right_response_times = allocate_response_times(response_times, correct_left, correct_right, incorrect_left, incorrect_right)
    """
    correct_left_response_times = [time for trial, time in enumerate(response_times) if correct_left[trial] == 1]
    correct_right_response_times = [time for trial, time in enumerate(response_times) if correct_right[trial] == 1]
    incorrect_left_response_times = [time for trial, time in enumerate(response_times) if incorrect_left[trial] == 1]
    incorrect_right_response_times = [time for trial, time in enumerate(response_times) if incorrect_right[trial] == 1]

    return correct_left_response_times, correct_right_response_times, incorrect_left_response_times, incorrect_right_response_times

def cutoff_response_times(response_times, cutoff):
    """
    Filters out response times greater than a specified cutoff value.

    Parameters:
        response_times (list): A list of response times.
        cutoff (float): The maximum response time allowed.

    Returns:
        list: A list of response times that are less than or equal to the cutoff value.

    Example usage:
        response_times = [0.5, 0.8, 1.2, 0.3, 0.9]
        cutoff = 1.0
        cutoff_response_times(response_times, cutoff)
        >>> [0.5, 0.8, 0.3, 0.9]
    """
    response_times_cutoff = []
    for trial in response_times:
        if trial > cutoff:
            continue
        else:
            response_times_cutoff.append(trial)

    return response_times_cutoff