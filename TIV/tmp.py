from labcams import parse_cam_log, unpackbits  # pip install labcams if it doesn't work
import h5py as h5
from scipy.interpolate import interp1d
analysis_folders = [os.path.dirname(r) for r in glob(r'C:/Users/Anne/data/JC111/*/*/*.mptracker.h5')]
task_folders = [os.path.dirname(r) for r in glob(r'C:/Users/Anne/data/JC111/*/*/*.triallog.h5')]

def read_data_from_pupil_chaoqun(analysis_folders, task_folders):

    with h5.File(glob(pjoin(analysis_folder,'*.mptracker.h5'))[0],'r') as pupildata:
        diam = pupildata['diameter'][:]
    # camlog,comm = parse_cam_log(glob(pjoin(task_folder,'*.camlog'))[0])
    camlog = parse_cam_log(glob(pjoin(task_folder,'*.camlog'))[0])
    sync = unpackbits(np.array(camlog['var2']))[0][2]
    trialdata = pd.read_hdf(glob(pjoin(task_folder,'*triallog.h5'))[0])

    trial_onsets = np.array([r.task_start_time + r.task_states[2][-1] for i,r in trialdata.iterrows()])
    # get the frametimes syncronized with the trial onset.
    camtime = interp1d(sync[:len(trial_onsets)],trial_onsets+0.3, fill_value = 'extrapolate')(np.arange(len(diam)))


    rewarded_side = np.array(trialdata.rewarded_side == 'left',dtype = int)
    rewarded_side[rewarded_side==0] = 2

    trial_start = np.array(trialdata.task_start_time)
    stim_onset = np.array([r.task_start_time + r.task_states[2][-1] for i,r in trialdata.iterrows()])
    stim_duration = np.array(trialdata.stim_duration)
    response = np.array(trialdata.response)
    response[response==-1] = 2
    reward_time = np.array([r.task_start_time + r.task_states[4][-1]  if r.rewarded else np.nan for i,r in trialdata.iterrows()])    
    rewarded = np.array(trialdata.rewarded)
    response_time = np.array(trialdata.response_time)
    # timeout_time = np.array([r.task_start_time + r.task_states[5][-1]  if ((not r.rewarded) and (r.response != 0)) else np.nan for i,r in trialdata.iterrows()])
    stim_intensity = np.array(trialdata.stim_intensity)
    # stim_rates = np.array(trialdata.stim_rates)
    
    trackerfile = glob(pjoin(analysis_folder,'*.mptracker.h5'))[0]
    fname = trackerfile.replace('DropletsTask','analysis').replace('mptracker.h5','session_data.h5')
    if not os.path.exists(os.path.dirname(fname)):
        os.makedirs(os.path.dirname(fname))
    with h5.File(fname,'w') as fid:
        fid.create_dataset('trialstart_times',data = trial_start)
        fid.create_dataset('frame_times',data = camtime)
        fid.create_dataset('pupil_diameter',data = diam)
        fid.create_dataset('stim_onsets',data = stim_onset)
        fid.create_dataset('stim_duration',data = stim_duration)
        fid.create_dataset('reward_times',data = reward_time)
        fid.create_dataset('timeout_times',data = reward_time)
        fid.create_dataset('choice',data = response)
        fid.create_dataset('correct_side',data = rewarded_side)
        fid.create_dataset('rewarded', data = rewarded)
        fid.create_dataset('response_time', data = response_time)
        fid.create_dataset('stim_intensity', data = stim_intensity)
        #fid.create_dataset('stim_rates', data = stim_rates)
    return trialdata,camtime,diam

#[read_data_from_pupil_chaoqun(analysis_folders, task_folders) for analysis_folder, task_folder in zip(analysis_folders, task_folders)];

for analysis_folder, task_folder in zip(analysis_folders, task_folders):
    read_data_from_pupil_chaoqun(analysis_folders, task_folders)