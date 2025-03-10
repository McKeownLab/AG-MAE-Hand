import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# -------------------------------
# Hyperparameters
# -------------------------------
batch_size = 32
learning_rate = 1e-3
num_epochs = 50
nhead = 4
num_layers = 4
ff_dim = 128
dropout = 0.2
max_frames = 500         # Fixed number of frames per video
d_model = 48             # Transformer model dimension
input_dim = 3 * 21       # Raw frame feature dimension (3,21 -> 63)
num_classes = 4          # Classes: 0, 1, 2, 3

# -------------------------------
# 1. Load Train/Val/Test Data
# -------------------------------
split_file = "data/train_val_test_splits.npy"
splits = np.load(split_file, allow_pickle=True)
split0 = splits[0]
train_data = split0["train"]
val_data = split0["val"]
test_data = split0["test"]

# -------------------------------
# 2. Create a Custom Dataset Class with Optional Transform
# -------------------------------
class VideoDataset(Dataset):
    def __init__(self, file_names, labels, coordinates, max_frames=max_frames, transform=None):
        """
        coordinates: list of np.arrays, each of shape (n_frames, 3, 21)
        transform: a function that takes a tensor of shape (max_frames, 63) and returns a transformed tensor.
        """
        self.file_names = file_names
        self.labels = labels
        self.coordinates = coordinates
        self.max_frames = max_frames
        self.transform = transform

    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, idx):
        coords = self.coordinates[idx]  # shape: (n_frames, 3, 21)
        n_frames = coords.shape[0]
        
        # Truncate if more than max_frames
        if n_frames > self.max_frames:
            coords = coords[:self.max_frames, :, :]
        # Pad if less than max_frames
        elif n_frames < self.max_frames:
            pad_frames = self.max_frames - n_frames
            pad_array = np.zeros((pad_frames, coords.shape[1], coords.shape[2]), dtype=coords.dtype)
            coords = np.concatenate([coords, pad_array], axis=0)
        
        # Now coords is (max_frames, 3, 21); flatten each frame to get shape (max_frames, 63)
        coords = coords.reshape(self.max_frames, -1)
        # Convert to torch tensor
        coords = torch.tensor(coords, dtype=torch.float32)
        if self.transform is not None:
            coords = self.transform(coords)
        # For CrossEntropyLoss, labels should be Long tensors
        label = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return coords, label

# -------------------------------
# 3. Compute Normalization Statistics from Training Set
# -------------------------------
# First, create a temporary train dataset WITHOUT transform to compute mean and std.
temp_train_dataset = VideoDataset(train_data["file_names"], train_data["labels"], train_data["coordinates"], max_frames=max_frames, transform=None)
all_train_samples = [temp_train_dataset[i][0] for i in range(len(temp_train_dataset))]
all_train_tensor = torch.stack(all_train_samples, dim=0)  # shape: (N, max_frames, 63)
# Compute mean and std over all frames and samples (i.e., over dimensions 0 and 1)
global_mean = all_train_tensor.mean(dim=(0,1))
global_std = all_train_tensor.std(dim=(0,1))
print("Normalization stats -- Mean:", global_mean, "Std:", global_std)

# Define a normalization transform function
def normalize_tensor(x, mean=global_mean, std=global_std, eps=1e-8):
    return (x - mean) / (std + eps)

# -------------------------------
# 4. Create Dataset Instances (with normalization)
# -------------------------------
train_dataset = VideoDataset(train_data["file_names"], train_data["labels"], train_data["coordinates"],
                             max_frames=max_frames, transform=normalize_tensor)
val_dataset   = VideoDataset(val_data["file_names"], val_data["labels"], val_data["coordinates"],
                             max_frames=max_frames, transform=normalize_tensor)
test_dataset  = VideoDataset(test_data["file_names"], test_data["labels"], test_data["coordinates"],
                             max_frames=max_frames, transform=normalize_tensor)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# -------------------------------
# 5. Model Building: Advanced Transformer-Based Classifier
# -------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim=63, d_model=48, nhead=4, num_layers=4, ff_dim=128, dropout=0.2, seq_length=max_frames, num_classes=4):
        super(TransformerClassifier, self).__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_length)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=ff_dim, 
            dropout=dropout, 
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Advanced classification head with one extra layer
        self.fc1 = nn.Linear(d_model, 128)
        self.ln1 = nn.LayerNorm(128)
        self.fc2 = nn.Linear(128, 64)
        self.ln2 = nn.LayerNorm(64)
        self.fc3 = nn.Linear(64, 32)
        self.ln3 = nn.LayerNorm(32)
        self.fc4 = nn.Linear(32, num_classes)   # Output layer: logits for 4 classes
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch_size, seq_length, input_dim)
        x = self.input_proj(x)            # -> (batch_size, seq_length, d_model)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)             # -> (seq_length, batch_size, d_model)
        x = self.transformer_encoder(x)   # -> (seq_length, batch_size, d_model)
        x = x.mean(dim=0)                 # Global average pooling -> (batch_size, d_model)
        x = self.fc1(x)
        x = self.ln1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.ln2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.ln3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc4(x)
        return x  # logits

# -------------------------------
# 6. Training Setup
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerClassifier(input_dim=input_dim, d_model=d_model, nhead=nhead, num_layers=num_layers,
                              ff_dim=ff_dim, dropout=dropout, seq_length=max_frames, num_classes=num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

# -------------------------------
# 7. Training Loop
# -------------------------------
train_losses, val_losses = [], []
# We'll reset the per-epoch validation predictions each epoch for confusion matrix printing every 5 epochs.
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    correct_train = 0
    total_train = 0
    
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        train_loss += loss.item() * X_batch.size(0)
        preds = torch.argmax(outputs, dim=1)
        correct_train += (preds == y_batch).sum().item()
        total_train += y_batch.size(0)
    
    train_loss /= total_train
    train_losses.append(train_loss)
    train_acc = correct_train / total_train
    
    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0
    val_all_preds = []
    val_all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item() * X_batch.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct_val += (preds == y_batch).sum().item()
            total_val += y_batch.size(0)
            val_all_preds.extend(preds.cpu().numpy())
            val_all_labels.extend(y_batch.cpu().numpy())
    
    val_loss /= total_val
    val_losses.append(val_loss)
    val_acc = correct_val / total_val
    
    scheduler.step(val_loss)
    
    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
    
    # Print confusion matrix every 5 epochs for validation set
    if (epoch + 1) % 5 == 0:
        val_cm = confusion_matrix(val_all_labels, val_all_preds)
        print(f"Confusion Matrix for Validation Set (Epoch {epoch + 1}):")
        print(val_cm)

# -------------------------------
# 8. Evaluation on Test Set
# -------------------------------
model.eval()
test_loss = 0.0
correct_test = 0
total_test = 0
test_all_preds = []
test_all_labels = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        test_loss += loss.item() * X_batch.size(0)
        preds = torch.argmax(outputs, dim=1)
        correct_test += (preds == y_batch).sum().item()
        total_test += y_batch.size(0)
        test_all_preds.extend(preds.cpu().numpy())
        test_all_labels.extend(y_batch.cpu().numpy())

test_loss /= total_test
test_acc = correct_test / total_test

print("Test Loss:", test_loss)
print("Test Accuracy:", test_acc)
test_cm = confusion_matrix(test_all_labels, test_all_preds)
print("Confusion Matrix for Test Set:")
print(test_cm)

# -------------------------------
# 9. Plotting Learning Curve and Confusion Matrices
# -------------------------------
plt.figure(figsize=(8, 5))
plt.plot(range(1, num_epochs+1), train_losses, label="Train Loss")
plt.plot(range(1, num_epochs+1), val_losses, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Learning Curve")
plt.legend()
plt.show()

# Plot confusion matrices for Validation and Test sets with annotations
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Validation Confusion Matrix Plot
ax1.set_title("Validation Set Confusion Matrix")
cax1 = ax1.imshow(val_cm, interpolation='nearest', cmap=plt.cm.Blues)
fig.colorbar(cax1, ax=ax1)
ax1.set_xticks(np.arange(val_cm.shape[1]))
ax1.set_yticks(np.arange(val_cm.shape[0]))
ax1.set_xticklabels([0, 1, 2, 3])
ax1.set_yticklabels([0, 1, 2, 3])
ax1.set_xlabel('Predicted')
ax1.set_ylabel('True')
for i in range(val_cm.shape[0]):
    for j in range(val_cm.shape[1]):
        ax1.text(j, i, str(val_cm[i, j]), ha='center', va='center', color='black')

# Test Confusion Matrix Plot
ax2.set_title("Test Set Confusion Matrix")
cax2 = ax2.imshow(test_cm, interpolation='nearest', cmap=plt.cm.Blues)
fig.colorbar(cax2, ax=ax2)
ax2.set_xticks(np.arange(test_cm.shape[1]))
ax2.set_yticks(np.arange(test_cm.shape[0]))
ax2.set_xticklabels([0, 1, 2, 3])
ax2.set_yticklabels([0, 1, 2, 3])
ax2.set_xlabel('Predicted')
ax2.set_ylabel('True')
for i in range(test_cm.shape[0]):
    for j in range(test_cm.shape[1]):
        ax2.text(j, i, str(test_cm[i, j]), ha='center', va='center', color='black')

plt.tight_layout()
plt.show()
