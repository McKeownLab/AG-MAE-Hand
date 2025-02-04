import numpy as np
from sklearn.model_selection import train_test_split

NPY_ADDRESS = "./data/stmae_v2.npy_943MB/stmae_embeddings_pd_4.npy"

def retrieve_data():
    with open(NPY_ADDRESS, 'rb') as f:
        data = np.load(NPY_ADDRESS, allow_pickle=True) 
    
    # step 1: data preprocessing
    data_embeddings = list(data.values())[0]
    data_labels     = list(data.values())[1]
    data_indices    = list(data.values())[2]

    data_embeddings_flat = data_embeddings.reshape(460, -1)  # New shape: (460, 400*21*64 = 537600)

    X_train, X_temp, y_train, y_temp = train_test_split(data_embeddings_flat, data_labels, 
                                                       test_size=0.3, 
                                                       random_state=42)

    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, 
                                                   test_size=0.5, 
                                                   random_state=42)

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test