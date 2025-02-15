import numpy as np
import pickle

NPY_ADDRESS     = "./data/stmae_v2.npy_943MB/stmae_embeddings_pd_4.npy"
NEW_NPY_ADDRESS = "./data/stmae_v2.npy_943MB/stmae_embeddings_new.npy"

with open(NPY_ADDRESS, 'rb') as f:
    data = np.load(NPY_ADDRESS, allow_pickle=True) 
    
data_embeddings = list(data.values())[0]
data_labels     = list(data.values())[1]
data_indices    = list(data.values())[2]

data_labels = np.array(data_labels)

# uncomment to convert labels to integer
# data_labels = data_labels.astype(int)  # Convert to int to avoid float issues

data_labels[data_labels == 4] = 3  

data[list(data.keys())[1]] = data_labels

with open(NEW_NPY_ADDRESS, 'wb') as f:
    np.save(f, data)

print("Modified .npy file saved successfully!")