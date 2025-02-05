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


NPY_ADDRESS = "./data/stmae_v2.npy_943MB/stmae_embeddings_pd_4.npy"

with open(NPY_ADDRESS, 'rb') as f:
    data = np.load(NPY_ADDRESS, allow_pickle=True) 

data_embeddings = list(data.values())[0]
data_labels     = torch.tensor(list(data.values())[1], dtype=torch.long)
data_indices    = list(data.values())[2]


# Splitting dataset (70% train, 15% val, 15% test)
def split_data(data, labels, train_ratio=0.7, val_ratio=0.15):
    num_samples = data.shape[0]
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    
    train_end = int(train_ratio * num_samples)
    val_end = train_end + int(val_ratio * num_samples)
    
    train_idx, val_idx, test_idx = indices[:train_end], indices[train_end:val_end], indices[val_end:]
    
    return (data[train_idx], labels[train_idx]), (data[val_idx], labels[val_idx]), (data[test_idx], labels[test_idx])

(train_data, train_labels), (val_data, val_labels), (test_data, test_labels) = split_data(data_embeddings, data_labels)

# Dataloader setup
class VideoDataset(Dataset):
    def __init__(self, videos, labels):
        self.videos = videos
        self.labels = labels
    
    def __len__(self):
        return len(self.videos)
    
    def __getitem__(self, idx):
        return self.videos[idx], self.labels[idx]

batch_size = 32
dataloaders = {
    'train': DataLoader(VideoDataset(train_data, train_labels), batch_size=batch_size, shuffle=True),
    'val': DataLoader(VideoDataset(val_data, val_labels), batch_size=batch_size, shuffle=False),
    'test': DataLoader(VideoDataset(test_data, test_labels), batch_size=batch_size, shuffle=False)
}

# Transformer-based model
class TransformerClassifier(nn.Module):
    def __init__(self, input_dim=64, seq_len=400, feature_dim=21, num_classes=5, num_heads=4, num_layers=2):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Linear(input_dim * feature_dim, 128)
        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=num_heads, dim_feedforward=256)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(128, num_classes)
    
    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, x.shape[1], -1)  # Flatten (21, 64) -> (21*64)
        x = self.embedding(x)  # (batch, seq_len, embed_dim)
        x = x.permute(1, 0, 2)  # Required shape (seq_len, batch, embed_dim)
        x = self.transformer(x)
        x = x.mean(dim=0)  # Aggregate over sequence
        return self.fc(x)

# Training setup
model = TransformerClassifier()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100
train_losses, val_losses = [], []
train_accs, val_accs = [], []

for epoch in range(num_epochs):
    model.train()
    train_loss, correct, total = 0.0, 0, 0
    
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
    
    train_acc = correct / total
    train_losses.append(train_loss / len(dataloaders['train']))
    train_accs.append(train_acc)
    
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    all_preds, all_targets = [], []
    
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
    
    val_acc = correct / total
    val_losses.append(val_loss / len(dataloaders['val']))
    val_accs.append(val_acc)
    
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_acc:.4f}")
    
    # Plot confusion matrix every 5 epochs
    if (epoch + 1) % 5 == 0:
        cm = confusion_matrix(all_targets, all_preds)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(5), yticklabels=range(5))
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix at Epoch {epoch+1}')
        plt.show()
        
                
# Plot Loss & Accuracy
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

# Testing Phase
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
    print(f"Test Accuracy: {acc:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

evaluate(model, dataloaders['test'])
