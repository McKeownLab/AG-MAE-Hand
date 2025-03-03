import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import ast
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Hyperparameters
BATCH_SIZE      = 8
LEARNING_RATE   = 1e-4
NUM_EPOCHS      = 30
NHEAD           = 4
NUM_LAYERS      = 3
FF_DIM          = 128
DROPOUT         = 0.2
NUM_CLASSES     = 2

# -------------------------------
# 1. Data Loading & Preprocessing
# -------------------------------
# Load the .npy file. Adjust NPY_ADDRESS as needed.
NPY_ADDRESS = "data/stmae_embeddings_pd_5.npy"

with open(NPY_ADDRESS, 'rb') as f:
    data = np.load(f, allow_pickle=True)

# Extract embeddings, labels, and indices from the dictionary-like structure
data_embeddings = list(data.values())[0]  # shape: (460, 400, 21, 16)
data_labels     = list(data.values())[1]
data_indices    = list(data.values())[2]

# Convert labels: Change any label with value 4 to 1 (so that only 0 and 1 remain)
data_labels = np.array(data_labels)
data_labels[data_labels == 4] = 1
data_labels[data_labels == 3] = 1 
data_labels[data_labels == 2] = 1 
data[list(data.keys())[1]] = data_labels

# Convert embeddings and labels to numpy arrays if needed
if isinstance(data_embeddings, torch.Tensor):
    data_embeddings = data_embeddings.numpy()
if isinstance(data_labels, torch.Tensor):
    data_labels = data_labels.numpy()

# Select only indices 1, 4, and 8 from the third dimension
selected_indices = [1, 4, 8]
data_embeddings = data_embeddings[:, :, selected_indices, :]
# Reshape to merge the last two dimensions: from (460, 400, 3, 16) to (460, 400, 48)
data_embeddings = data_embeddings.reshape(data_embeddings.shape[0], data_embeddings.shape[1], -1)

print("Data embeddings shape:", data_embeddings.shape)  # Expected: (460, 400, 48)
print("Unique labels:", np.unique(data_labels))  # Expected: [0, 1]

# Normalize the data (normalize each feature across samples)
X = data_embeddings.astype(np.float32)
X_mean = np.mean(X, axis=0, keepdims=True)
X_std = np.std(X, axis=0, keepdims=True)
X = (X - X_mean) / (X_std + 1e-8)

y = data_labels  # shape: (460,)

# Split data into train (70%), validation (15%), and test (15%) sets using stratification
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

print("Train shape:", X_train.shape, "Validation shape:", X_val.shape, "Test shape:", X_test.shape)

# -------------------------------
# 2. Dataset & DataLoader
# -------------------------------
class EmbeddingsDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)  # (N, 400, 48)
        self.y = torch.tensor(y, dtype=torch.long)       # (N,)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = EmbeddingsDataset(X_train, y_train)
val_dataset   = EmbeddingsDataset(X_val, y_val)
test_dataset  = EmbeddingsDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# -------------------------------
# 3. Model Building: Transformer-Based Classifier
# -------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=400):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        return x + self.pe[:, :x.size(1)]

class TransformerClassifier(nn.Module):
    def __init__(self, d_model=48, nhead=NHEAD, num_layers=NUM_LAYERS, ff_dim=FF_DIM, dropout=DROPOUT, seq_length=400, num_classes=NUM_CLASSES):
        super(TransformerClassifier, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_length)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=ff_dim, 
            dropout=dropout, 
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(d_model, 128)
        self.ln1 = nn.LayerNorm(128)  # Using LayerNorm for stability
        self.fc2 = nn.Linear(128, 64)
        self.ln2 = nn.LayerNorm(64)   # Using LayerNorm for stability
        self.fc3 = nn.Linear(64, num_classes)   # Output layer now outputs num_classes logits
        self.relu = nn.ReLU()

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # (seq_len, batch_size, d_model)
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)      # Global average pooling: (batch_size, d_model)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.ln1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.ln2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)        # (batch_size, num_classes)
        return x  # raw logits

# -------------------------------
# 4. Training Setup
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerClassifier().to(device)

# Compute class weights based on training set distribution
unique, counts = np.unique(y_train, return_counts=True)
print("Training label distribution:", dict(zip(unique, counts)))
total_samples = len(y_train)
# Weight for each class: total_samples / (num_classes * count)
class_weights_np = total_samples / (NUM_CLASSES * counts)
class_weights = torch.tensor(class_weights_np, dtype=torch.float32).to(device)
print("Class weights:", class_weights)

# Use CrossEntropyLoss with computed class weights
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

# -------------------------------
# 5. Training Loop
# -------------------------------
train_losses, val_losses = [], []

for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss = 0.0
    correct_train = 0
    total_train = 0
    
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)  # shape: (batch_size, num_classes)
        loss = criterion(outputs, y_batch)
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        train_loss += loss.item() * X_batch.size(0)
        _, preds = torch.max(outputs, 1)
        correct_train += (preds == y_batch).sum().item()
        total_train += y_batch.size(0)
    
    train_loss /= total_train
    train_acc = correct_train / total_train
    train_losses.append(train_loss)
    
    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item() * X_batch.size(0)
            _, preds = torch.max(outputs, 1)
            correct_val += (preds == y_batch).sum().item()
            total_val += y_batch.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    
    val_loss /= total_val
    val_acc = correct_val / total_val
    val_losses.append(val_loss)
    
    scheduler.step(val_loss)
    
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
    
    if (epoch + 1) % 5 == 0:
        cm = confusion_matrix(all_labels, all_preds)
        print(f"\nConfusion Matrix at epoch {epoch+1}:")
        print(cm)

# -------------------------------
# 6. Evaluation on Test Set
# -------------------------------
model.eval()
test_loss = 0.0
correct_test = 0
total_test = 0

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        test_loss += loss.item() * X_batch.size(0)
        _, preds = torch.max(outputs, 1)
        correct_test += (preds == y_batch).sum().item()
        total_test += y_batch.size(0)

test_loss /= total_test
test_acc = correct_test / total_test

print("Test Loss:", test_loss)
print("Test Accuracy:", test_acc)

# -------------------------------
# 7. Learning Curve Plot
# -------------------------------
plt.plot(range(1, NUM_EPOCHS + 1), train_losses, label='Train Loss')
plt.plot(range(1, NUM_EPOCHS + 1), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Learning Curve')
plt.legend()
plt.show()
