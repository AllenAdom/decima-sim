import numpy as np
query_idx = 1
query_path = '2g/'

adj_mat = np.load(
    query_path + 'adj_mat_' + str(query_idx) + '.npy', allow_pickle=True)
task_durations = np.load(
    query_path + 'task_duration_' + str(query_idx) + '.npy', allow_pickle=True)
task_durations_items = task_durations.item()


print('hello')