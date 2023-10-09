def categorize_trials(sessiondata, exclude=None):
    """
    Categorizes trials by response (left/right) and outcome (rewarded/unrewarded).

    Inputs
    ------
    sessiondata: DataFrame, loaded from the output file created with chipmunk_analysis_tools to convert .mat to .h5. 0 indicates left, 1 indicates right.
    filter_out: int or list that specifies the stimulus rates that the user wants to exclude from the output data. Useful when wanting to exclude responses at chance (12 Hz in chipmunk)

    Outputs
    ------
    correct_left, correct_right, incorrect_left, incorrect_right: lists with zeros and ones identifying the trials of each type
    """

    import pandas as pd
    import numpy as np
    
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

    return correct_left, correct_right, incorrect_left, incorrect_right, exclude


def allocate_response_times(response_times, correct_left, correct_right, incorrect_left, incorrect_right):
    """
    Gets response times for each category of trials

    Inputs
    -----
    response_times: an array containing the response time for each trial that the animal responded (no early withdrawal or no choice trials; these might cause an error, but haven't tested this to confirm)
    correct_left, correct_right, incorrect_left, incorrect_right: lists with zeros and ones identifying the trials of each type

    Outputs
    -----
    correct_left_response_times, correct_right_response_times, incorrect_left_response_times, incorrect_right_response_times: response times categorized by the different types of responses and outcomes
    """

    correct_left_response_times = [time for trial, time in enumerate(response_times) if correct_left[trial] == 1]
    correct_right_response_times = [time for trial, time in enumerate(response_times) if correct_right[trial] == 1]
    incorrect_left_response_times = [time for trial, time in enumerate(response_times) if incorrect_left[trial] == 1]
    incorrect_right_response_times = [time for trial, time in enumerate(response_times) if incorrect_right[trial] == 1]

    return correct_left_response_times, correct_right_response_times, incorrect_left_response_times, incorrect_right_response_times


def cutoff_response_times(response_times, cutoff):
    """
    Filters response times larger than the cutoff value

    Inputs:
    -----
    response_times: the array of response times to be filtered
    cutoff: number in seconds to filter past

    Output:
    -----
    response_times_cutoff: filtered list of response times
    """

    response_times_cutoff = []
    for trial in response_times:
        if trial>cutoff:
            continue
        else:
            response_times_cutoff.append(trial[0])
            
    return response_times_cutoff