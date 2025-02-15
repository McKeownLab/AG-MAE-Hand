import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import tqdm

# =============================================================================
# 1. Data Loading and Splitting
# =============================================================================
NPY_ADDRESS = "./data/stmae_v2.npy_943MB/stmae_embeddings_pd_4.npy"

with open(NPY_ADDRESS, 'rb') as f:
    data = np.load(f, allow_pickle=True)

# Expected: data dictionary with keys: embeddings, labels, indices.
data_embeddings = list(data.values())[0]  # shape: (460, 400, 21, 64)
data_labels     = list(data.values())[1]
data_indices    = list(data.values())[2]

# Convert labels so that class 4 becomes class 3 (since you have 4 classes)
data_labels = np.array(data_labels)
data_labels[data_labels == 4] = 3  
data[list(data.keys())[1]] = data_labels
data_labels = torch.tensor(list(data.values())[1], dtype=torch.long)

def split_data(data, labels, train_ratio=0.7, val_ratio=0.15):
    num_samples = data.shape[0]
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    
    train_end = int(train_ratio * num_samples)
    val_end   = train_end + int(val_ratio * num_samples)
    
    train_idx = indices[:train_end]
    val_idx   = indices[train_end:val_end]
    test_idx  = indices[val_end:]
    
    return (data[train_idx], labels[train_idx]), (data[val_idx], labels[val_idx]), (data[test_idx], labels[test_idx])

(train_data, train_labels), (val_data, val_labels), (test_data, test_labels) = split_data(data_embeddings, data_labels)

# =============================================================================
# 2. Define a Custom Dataset
# =============================================================================
class VideoDataset(Dataset):
    def __init__(self, videos, labels):
        """
        videos: numpy array of shape (num_videos, 400, 21, 64)
        labels: tensor of shape (num_videos,)
        """
        self.videos = videos
        self.labels = labels
    
    def __len__(self):
        return len(self.videos)
    
    def __getitem__(self, idx):
        video = torch.tensor(self.videos[idx], dtype=torch.float32)
        return video, self.labels[idx]

batch_size = 32
dataloaders = {
    'train': DataLoader(VideoDataset(train_data, train_labels), batch_size=batch_size, shuffle=True),
    'val': DataLoader(VideoDataset(val_data, val_labels), batch_size=batch_size, shuffle=False),
    'test': DataLoader(VideoDataset(test_data, test_labels), batch_size=batch_size, shuffle=False)
}

# =============================================================================
# 3. Create the Adjacency Matrix for the Hand Graph
# =============================================================================
def create_hand_adj(num_nodes=21):
    """
    Create and normalize an adjacency matrix for hand joints.
    Uses a typical MediaPipe hand connectivity.
    """
    A = np.zeros((num_nodes, num_nodes))
    # Add self-loops
    for i in range(num_nodes):
        A[i, i] = 1

    # Example connectivity for fingers
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),       # Index
        (0, 9), (9, 10), (10, 11), (11, 12),   # Middle
        (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
        (0, 17), (17, 18), (18, 19), (19, 20)   # Pinky
    ]
    for i, j in edges:
        A[i, j] = 1
        A[j, i] = 1  # Undirected graph

    # Symmetric normalization: A_norm = D^{-1/2} A D^{-1/2}
    d = np.sum(A, axis=1)
    d_inv_sqrt = np.power(d, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    D_inv_sqrt = np.diag(d_inv_sqrt)
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt
    return torch.tensor(A_norm, dtype=torch.float32)

# =============================================================================
# 4. Define Positional Encoding Module
# =============================================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=400):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create constant 'pe' matrix with values dependent on position and i
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices
        pe = pe.unsqueeze(1)  # (max_len, 1, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        x: Tensor of shape (T, B, d_model)
        """
        T = x.size(0)
        x = x + self.pe[:T]
        return self.dropout(x)

# =============================================================================
# 5. Define the GNN (GCN) Layer and the Combined Model
# =============================================================================
class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        """
        Simple Graph Convolution Layer.
        """
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        
    def forward(self, x, adj):
        """
        x: (batch, num_nodes, in_features)
        adj: (num_nodes, num_nodes)
        """
        x = torch.matmul(adj, x)  # Aggregate neighbors
        x = self.linear(x)
        return torch.relu(x)

class GNNTransformerClassifier(nn.Module):
    def __init__(self, 
                 num_joints=21, 
                 node_in_features=64, 
                 gnn_out_features=128, 
                 seq_len=400, 
                 num_classes=4,
                 num_heads=4,
                 num_transformer_layers=4):  # Increased transformer layers
        """
        The model first processes each frame's hand joints with a GNN,
        applies positional encoding, then uses a transformer to capture
        temporal relationships.
        """
        super(GNNTransformerClassifier, self).__init__()
        
        # GNN layer for spatial (hand joint) processing
        self.gnn = GCNLayer(in_features=node_in_features, out_features=gnn_out_features)
        self.gnn_out_features = gnn_out_features
        
        # Register the hand adjacency matrix as a buffer
        self.register_buffer('adj', create_hand_adj(num_joints))
        
        # Positional encoding module for the temporal sequence
        self.pos_encoder = PositionalEncoding(d_model=gnn_out_features, dropout=0.1, max_len=seq_len)
        
        # Transformer encoder expects input shape: (seq_len, batch, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=gnn_out_features, 
                                                   nhead=num_heads, 
                                                   dim_feedforward=256,
                                                   dropout=0.1,
                                                   activation='relu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        
        # Final classification layer
        self.fc = nn.Linear(gnn_out_features, num_classes)
        
    def forward(self, x):
        """
        x: (B, T, num_joints, node_in_features) = (batch, 400, 21, 64)
        """
        B, T, N, F = x.shape
        
        # --- GNN Processing (Spatial) ---
        # Process each frame independently:
        x = x.view(B * T, N, F)   # (B*T, 21, 64)
        x = self.gnn(x, self.adj)  # (B*T, 21, gnn_out_features)
        x = x.mean(dim=1)         # Mean pooling over joints: (B*T, gnn_out_features)
        x = x.view(B, T, self.gnn_out_features)  # (B, T, gnn_out_features)
        
        # --- Transformer Processing (Temporal) ---
        # Permute to shape (T, B, d_model) for transformer input:
        x = x.permute(1, 0, 2)      # (T, B, gnn_out_features)
        x = self.pos_encoder(x)     # Add positional encoding
        x = self.transformer(x)     # (T, B, gnn_out_features)
        x = x.mean(dim=0)          # Aggregate over time: (B, gnn_out_features)
        
        # --- Classification ---
        out = self.fc(x)           # (B, num_classes)
        return out

# =============================================================================
# 6. Training Setup
# =============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GNNTransformerClassifier(num_classes=4).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 100

train_losses, val_losses = [], []
train_accs, val_accs = [], []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, targets in dataloaders['train']:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
    
    train_losses.append(train_loss / len(dataloaders['train']))
    train_acc = correct / total
    train_accs.append(train_acc)
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in dataloaders['val']:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    val_losses.append(val_loss / len(dataloaders['val']))
    val_acc = correct / total
    val_accs.append(val_acc)
    
    print(f"Epoch {epoch+1}/{num_epochs} | "
          f"Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_acc:.4f} | "
          f"Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_acc:.4f}")
    
    # Plot confusion matrix every 5 epochs
    if (epoch + 1) % 5 == 0:
        cm = confusion_matrix(all_targets, all_preds)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=range(4), yticklabels=range(4))
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix at Epoch {epoch+1}')
        plt.show()

# Plot Loss & Accuracy curves
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs+1), val_losses, label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Over Epochs')

plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs+1), train_accs, label='Train Acc')
plt.plot(range(1, num_epochs+1), val_accs, label='Val Acc')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy Over Epochs')
plt.show()

# =============================================================================
# 7. Testing and Reporting Metrics
# =============================================================================
def evaluate(model, dataloader):
    model.eval()
    preds_list, targets_list = [], []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            preds_list.extend(preds.cpu().numpy())
            targets_list.extend(targets.cpu().numpy())
    
    acc = accuracy_score(targets_list, preds_list)
    recall = recall_score(targets_list, preds_list, average='macro')
    f1 = f1_score(targets_list, preds_list, average='macro')
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1 Score: {f1:.4f}")

print("Evaluating on Test Data:")
evaluate(model, dataloaders['test'])
