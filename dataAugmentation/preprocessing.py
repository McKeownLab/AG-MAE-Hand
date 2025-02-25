import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import random
import math

NPY_ADDRESS         = "./data/stmae_embeddings_pd_5.npy"
LEFT_DATA_ADDRESS   = "./data/left_best_segments_and_indexes.csv"
RIGHT_DATA_ADDRESS  = "./data/right_best_segments_and_indexes.csv"
SHUFFLED_DATA_RATIO = 3


def shuffle_tuples_list(tuples_list, number_of_shuffles):
    print(len(tuples_list))
    shuffles = []
    shuffles_num = number_of_shuffles
    len_tuple = len(tuples_list)
    
    if number_of_shuffles > math.factorial(len_tuple-2):
        shuffles_num = math.factorial(len_tuple-2)   

    if len(tuples_list) <= 3:
        return []
    
    while len(shuffles) < shuffles_num:
        temp_list = tuples_list[:]
        
        # Extract the elements to shuffle (excluding the first and last elements)
        middle_section = temp_list[1:-1]
        random.shuffle(middle_section)
        shuffled_list = [temp_list[0]] + middle_section + [temp_list[-1]]
        
        if shuffled_list not in shuffles and shuffled_list != tuples_list:
            shuffles.append(shuffled_list)
    
    return shuffles


with open(NPY_ADDRESS, 'rb') as f:
    data = np.load(f, allow_pickle=True)

data_embeddings = list(data.values())[0]  # shape: (460, 400, 21, 16)
data_labels     = list(data.values())[1]
data_indices    = list(data.values())[2]

# Convert labels so that class 4 becomes class 3
data_labels = np.array(data_labels)
data_labels[data_labels == 4] = 3  
data[list(data.keys())[1]] = data_labels

data_embeddings = torch.tensor(list(data.values())[0], dtype=torch.long)
data_labels = torch.tensor(list(data.values())[1], dtype=torch.long)
print(torch.unique(data_labels)) # should be from 0 to 3


#keeping only indices 1, 4, 8
selected_indices = [1, 4, 8]
data_embeddings = data_embeddings[:, :, selected_indices, :]
print(data_embeddings.shape)  # Output should be (460, 400, 3, 16)

# Reshape to merge the 3*16 dimension into a single 48 dimension
data_embeddings = data_embeddings.view(data_embeddings.shape[0], data_embeddings.shape[1], -1)
print(data_embeddings.shape)  # Output should be (460, 400, 48)

#read csv data
left_data = pd.read_csv(LEFT_DATA_ADDRESS)
right_data = pd.read_csv(RIGHT_DATA_ADDRESS)

print(left_data[:5])
left_tap_frames = left_data['tap_indices'].tolist()
right_tap_frames = right_data['tap_indices'].tolist()

tuples_list_left = []
tuples_list_right = []

#generate tuples data from the frames range
for tap_frame_str in left_tap_frames:
    numbers = list(map(int, tap_frame_str.split(',')))
    tuples_list = [(numbers[i] + 1, numbers[i+1]) for i in range(len(numbers) - 1) if numbers[i+1] < 399]
    tuples_list.append((tuples_list[-1][1] + 1, 399))
    tuples_list.insert(0, (0, 0))
    tuples_list_left.append(tuples_list)
    
for tap_frame_str in right_tap_frames:
    numbers = list(map(int, tap_frame_str.split(',')))
    tuples_list = [(numbers[i] + 1, numbers[i+1]) for i in range(len(numbers) - 1) if (numbers[i+1] < 399)]
    tuples_list.append((tuples_list[-1][1] + 1, 399))
    tuples_list.insert(0, (0, 0))
    tuples_list_right.append(tuples_list)
    

print(len(tuples_list_left), len(left_data['file_name'].tolist()))
print(len(tuples_list_right), len(right_data['file_name'].tolist()))

#Fix file names and sync it with the npy embeddings file
left_tap_file_names = left_data['file_name'].tolist()
right_tap_file_names = right_data['file_name'].tolist()
left_tap_file_names = [file_name.replace("_finger_tapping_distances.csv", "") for file_name in left_tap_file_names]
right_tap_file_names = [file_name.replace("_finger_tapping_distances.csv", "") for file_name in right_tap_file_names]


df_left = pd.DataFrame({'file_name': left_tap_file_names, 'tap_indices': tuples_list_left, 'label': [1 for _ in range(len(tuples_list_left))]})
df_right = pd.DataFrame({'file_name': right_tap_file_names, 'tap_indices': tuples_list_right, 'label': [1 for _ in range(len(tuples_list_left))]})


tmp_df_column_names = ['file_name', 'tap_indices', 'label']
tmp_df_left = pd.DataFrame(columns=tmp_df_column_names)
tmp_df_right = pd.DataFrame(columns=tmp_df_column_names)

for index, row in df_left.iterrows():
    print(index, row['file_name'])
    file_name = row['file_name']
    tap_indices = row['tap_indices']
    shuffled_tap_indices = shuffle_tuples_list(tap_indices, SHUFFLED_DATA_RATIO)
    new_row = {'file_name': file_name, 'tap_indices': tap_indices, 'label': 1}
    tmp_df_left = pd.concat([tmp_df_left, pd.DataFrame([new_row])], ignore_index=True)
    for shuffled in shuffled_tap_indices:
        new_row = {'file_name': file_name, 'tap_indices': shuffled, 'label': 0}
        tmp_df_left = pd.concat([tmp_df_left, pd.DataFrame([new_row])], ignore_index=True)
    
for index, row in df_right.iterrows():
    print(index, row['file_name'])
    file_name = row['file_name']
    tap_indices = row['tap_indices']
    shuffled_tap_indices = shuffle_tuples_list(tap_indices, SHUFFLED_DATA_RATIO)
    new_row = {'file_name': file_name, 'tap_indices': tap_indices, 'label': 1}
    tmp_df_right = pd.concat([tmp_df_right, pd.DataFrame([new_row])], ignore_index=True)
    for shuffled in shuffled_tap_indices:
        new_row = {'file_name': file_name, 'tap_indices': shuffled, 'label': 0}
        tmp_df_right = pd.concat([tmp_df_right, pd.DataFrame([new_row])], ignore_index=True)
    

tmp_df_left.to_csv('./left_test.csv')
tmp_df_right.to_csv('./right_test.csv')


