import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import random
import math
import ast

LEFT_DF_LOC  = './left_shuffled.csv'
RIGHT_DF_LOC = './right_shuffled.csv'
NPY_ADDRESS  = "./data/stmae_embeddings_pd_5.npy"

df_left = pd.read_csv(LEFT_DF_LOC)
df_right = pd.read_csv(RIGHT_DF_LOC)

with open(NPY_ADDRESS, 'rb') as f:
    data = np.load(f, allow_pickle=True)

data_embeddings = list(data.values())[0]  # shape: (460, 400, 21, 16)
data_labels     = list(data.values())[1]
data_indices    = list(data.values())[2]

# Convert labels so that class 4 becomes class 3
data_labels = np.array(data_labels)
data_labels[data_labels == 4] = 3  
data[list(data.keys())[1]] = data_labels

data_embeddings = torch.tensor(list(data.values())[0], dtype=torch.float)
data_labels = torch.tensor(list(data.values())[1], dtype=torch.float)
print(torch.unique(data_labels)) # should be from 0 to 3


#keeping only indices 1, 4, 8
selected_indices = [1, 4, 8]
data_embeddings = data_embeddings[:, :, selected_indices, :]
print(data_embeddings.shape)  # Output should be (460, 400, 3, 16)

# Reshape to merge the 3*16 dimension into a single 48 dimension
data_embeddings = data_embeddings.view(data_embeddings.shape[0], data_embeddings.shape[1], -1)
print(data_embeddings.shape)  # Output should be (460, 400, 48)


embeddings_list_left_all = []
embeddings_list_right_all = []


# convert data_embeddings back to a normal list
if data_embeddings.is_cuda:
    data_embeddings = data_embeddings.cpu()
data_embeddings = data_embeddings.numpy()
data_embeddings = data_embeddings.tolist()

error_count_left = 0
for index, row in df_left.iterrows():
    embeddings_list_left = []
    tap_indices = row['tap_indices']
    file_name = row['file_name']
    
    try:
        file_name_index = data_indices.index(file_name)
        file_name_embeddings = data_embeddings[file_name_index]
    except ValueError:
        error_count_left += 1  # Increment the error counter
        continue
    
    tap_indices = ast.literal_eval(tap_indices)
    # print(len(tap_indices))
    
    for start, end in tap_indices:
        if start == end:
            embeddings_list_left.append(file_name_embeddings[start])
        else:
            embeddings_list_left.extend(file_name_embeddings[start:end+1])
    
    embeddings_list_left_all.append(embeddings_list_left)
    
    # print(len(embeddings_list_left))    
    # x = input('interrupt: ')
print(f"Number of items not found in data_indices: {error_count_left}")
print(len(embeddings_list_left_all))