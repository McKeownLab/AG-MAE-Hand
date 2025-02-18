import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset

from torch_geometric.loader import DataLoader


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, f1_score
import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score
import seaborn as sns

from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F

from torch.utils.data import random_split

import random


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


BATCH_SIZE = 256
NUM_CLASSES = 4
HIDDEN_DIM = 256
LEARNING_RATE = 0.001
NUM_EPOCHS = 200


class ParkinsonGNN(nn.Module):
    def __init__(self, in_feats = 64, hidden_dim = HIDDEN_DIM, num_classes = NUM_CLASSES):
        super(ParkinsonGNN, self).__init__()


        self.conv1 = GCNConv(in_feats, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.conv4 = GCNConv(hidden_dim, hidden_dim)

        self.in_feats = in_feats

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),       
            nn.ReLU(),                        
            nn.Dropout(0.3),                       
            nn.Linear(hidden_dim // 2, num_classes) 
        )

        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, batch):
        
        x, edge_index = batch.x, batch.edge_index 
        x = torch.tensor(x)
        x = x.reshape(-1, self.in_feats)
        edge_index = torch.tensor(edge_index)

        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = F.relu(self.conv4(x, edge_index))


        x = global_mean_pool(x, batch.batch) 


        x = self.classifier(x)

        return x
   

    
class ParkinsonDataset(Dataset):

    def __init__(self, data_list):
        self.data_list = data_list 

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        return self.data_list[idx]
    

NPY_ADDRESS =  "../../Datasets/stmae_embeddings_pd_4.npy"

with open(NPY_ADDRESS, 'rb') as f:
    data = np.load(NPY_ADDRESS, allow_pickle=True) 

data_embeddings = list(data.values())[0]
data_labels     = torch.tensor(list(data.values())[1], dtype=torch.long)
data_indices    = list(data.values())[2]

data_labels = np.array(data_labels)
data_labels[data_labels == 4] = 3  
data[list(data.keys())[1]] = data_labels
data_labels = torch.tensor(list(data.values())[1], dtype=torch.long)


frame_mean_embeddings = data_embeddings.mean(axis = 1)

batch_size, seq_len, num_nodes, feat_dim = data_embeddings.shape  # (460, 400, 21, 64)
data_embeddings = torch.tensor(data_embeddings)
frame_flattened_embeddings = data_embeddings.permute(0, 2, 1, 3).reshape(batch_size, num_nodes, seq_len * feat_dim)

finger_edges = torch.tensor([
    [0, 1], [1, 2], [2, 3], [3, 4], 
    [0, 5], [5, 6], [6, 7], [7, 8],
    [0, 9], [9, 10], [10, 11], [11, 12],
    [0, 13], [13, 14], [14, 15], [15, 16],
    [0, 17], [17, 18], [18, 19], [19, 20]
], dtype=torch.long).t()

graphs = []

for i in range(frame_mean_embeddings.shape[0]):

    x = frame_mean_embeddings[i]
    graph = Data(x = x, edge_index = finger_edges, y = data_labels[i])
    graphs.append(graph)

# parkinson_dataset = ParkinsonDataset(graphs)
batch_size = BATCH_SIZE

shuffled_graphs = torch.utils.data.RandomSampler(graphs)  # Shuffle the indices of the dataset
graphs = [graphs[i] for i in shuffled_graphs]

train_dataset, val_dataset, test_dataset = random_split(graphs, [0.7, 0.15, 0.15])

train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle=False)

# Training part

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ParkinsonGNN().to(device)
optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)
criterion = nn.CrossEntropyLoss()  # For classification

num_epochs = NUM_EPOCHS 
for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0

    for batch in train_loader:
        batch = batch.to(device) 
        optimizer.zero_grad() 

        out = model(batch) 
        loss = criterion(out, batch.y) 
        loss.backward()  
        optimizer.step()  

        total_train_loss += loss.item() 

    model.eval()
    correct = 0
    total_validation_loss = 0 

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            out = model(batch)
            loss = criterion(out, batch.y) 
            pred = out.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total_validation_loss += loss.item() 

    print(f"Epoch {epoch+1}, Train Loss: {total_train_loss / len(train_loader)}, Validation Loss: {total_validation_loss / len(train_loader)}, Number of Correct: {correct}")




model.eval()
preds_list, targets_list = [], []
    
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        outputs = model(batch)
        preds = outputs.argmax(dim=1)
        preds_list.extend(preds.cpu().numpy())
        targets_list.extend(batch.y.cpu().numpy())

print(preds_list)
print(targets_list)
acc = accuracy_score(targets_list, preds_list)
recall = recall_score(targets_list, preds_list, average='macro')
f1 = f1_score(targets_list, preds_list, average='macro')
print(f"Test Accuracy: {acc:.4f}")
print(f"Test Recall: {recall:.4f}")
print(f"Test F1 Score: {f1:.4f}")