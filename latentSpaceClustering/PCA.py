import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, f1_score
import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score
import seaborn as sns
from sklearn.decomposition import PCA


NPY_ADDRESS = "../../Datasets/stmae_embeddings_pd_4.npy"


with open(NPY_ADDRESS, 'rb') as f:
    data = np.load(NPY_ADDRESS, allow_pickle=True) 

data_embeddings = torch.tensor(list(data.values())[0])
data_labels     = torch.tensor(list(data.values())[1], dtype=torch.long)
data_indices    = list(data.values())[2]

hand_lr = [0 if 'left' in index else 1 for index in data_indices]
hand_lr = np.array(hand_lr)

n = 400 * 21 * 64

reshaped_data_embedding = data_embeddings.view(460, n)

reshaped_data_embedding_np = reshaped_data_embedding.numpy()
labels_np = data_labels.numpy()  

print(labels_np)

pca = PCA(n_components = 2)
tensor_pca_np = pca.fit_transform(reshaped_data_embedding_np)

print(tensor_pca_np)

plt.figure(figsize=(8, 6))

scatter = plt.scatter(
    tensor_pca_np[:, 0], tensor_pca_np[:, 1], 
    c=hand_lr, cmap='tab10', alpha=0.7
)
legend1 = plt.legend(*scatter.legend_elements(), title="Labels")
plt.gca().add_artist(legend1)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Projection with flattening")
plt.grid(True)
plt.show()
