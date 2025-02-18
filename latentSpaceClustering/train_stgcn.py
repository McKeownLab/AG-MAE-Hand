import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sys
from tqdm import tqdm
from google.colab import drive  # Import Google Drive

# ---------------------------
# Mount Google Drive
# ---------------------------
drive.mount('/content/drive')

# Update this path with the correct location in your Google Drive
NPY_FILE = "/content/drive/MyDrive/stmae_embeddings_pd_4_cp.npy"  # Modify this path

# ---------------------------
# Load and Modify Dataset Labels
# ---------------------------
with open(NPY_FILE, 'rb') as f:
    data = np.load(f, allow_pickle=True)

# Extract data from dictionary
data_embeddings = data['embeddings']  # shape: (460, 400, 21, 64)
data_labels = np.array(data['labels'])  # Convert labels to numpy array

# Modify labels: Convert class 4 to class 3
data_labels[data_labels == 4] = 3  
data['labels'] = data_labels  # Update labels in the dictionary

# ---------------------------
# Custom Dataset Definition
# ---------------------------
class STGCNDataset(Dataset):
    def __init__(self, data_dict, max_frames=400):
        self.embeddings = data_dict['embeddings']
        self.labels = data_dict['labels']
        self.max_frames = max_frames

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        video = self.embeddings[idx]  # shape: (T, 21, 64)
        label = self.labels[idx]
        T, V, C = video.shape

        # Pad with zeros if T < max_frames; crop if T > max_frames
        if T < self.max_frames:
            pad = np.zeros((self.max_frames - T, V, C), dtype=video.dtype)
            video = np.concatenate([video, pad], axis=0)
        elif T > self.max_frames:
            video = video[:self.max_frames]

        # Convert to torch tensor
        video = torch.tensor(video, dtype=torch.float32)
        label = torch.tensor(int(label), dtype=torch.long)
        return video, label

# ---------------------------
# Training and Validation Loops
# ---------------------------
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for videos, labels in pbar:
        videos = videos.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(videos)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * videos.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += videos.size(0)
        pbar.set_postfix(loss=loss.item())
    return running_loss / total, correct / total

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation", leave=False)
        for videos, labels in pbar:
            videos = videos.to(device)
            labels = labels.to(device)
            outputs = model(videos)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * videos.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += videos.size(0)
            pbar.set_postfix(loss=loss.item())
    return running_loss / total, correct / total

# ---------------------------
# Main Function
# ---------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create dataset and split into training and validation sets
    dataset = STGCNDataset(data, max_frames=400)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

    # Graph configuration for 21 joints
    graph_cfg = dict(layout='handmp', mode='spatial')

    # Import STGCN model from Drive
    # sys.path.append('/content/drive/MyDrive/model')  # Ensure the model folder is in Python path
    from stgcn import STGCN  # Import the STGCN model

    # Instantiate STGCN
    model = STGCN(graph_cfg,
                  in_channels=64,
                  base_channels=64,
                  num_classes=10,
                  ch_ratio=2,
                  num_stages=6,
                  inflate_stages=[3, 5],
                  down_stages=[3, 5],
                  task='class')
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    best_val_acc = 0.0
    for epoch in range(1, 51):
        print(f"Epoch {epoch}/50")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f}, Val   Acc: {val_acc:.4f}")

        # Save best model
        # if val_acc > best_val_acc:
        #     best_val_acc = val_acc
        #     save_path = os.path.join("/content/drive/MyDrive/checkpoints", "best_stgcn_model.pth")
        #     torch.save(model.state_dict(), save_path)
        #     print(f"Saved best model with val acc {best_val_acc:.4f} at epoch {epoch}")

if __name__ == '__main__':
    os.makedirs("/content/drive/MyDrive/checkpoints", exist_ok=True)
    main()
