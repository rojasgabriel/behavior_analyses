from utils import *
file_names = pick_files_multi_session("chipmunk", "*.mat")
h5_files = convert_specified_behavior_sessions(file_names, overwrite=False)