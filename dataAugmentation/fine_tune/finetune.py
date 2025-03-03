import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import ast
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# 1. Data Loading & Preprocessing
# -------------------------------

# Load the fine-tuning CSV file (update the path if needed)
data = pd.read_csv("data/finetune.csv")

# Parse the embeddings column (assuming each cell is a string representation of a (400, 48) list)
def parse_embedding(x):
    return np.array(ast.literal_eval(x))

data["embedding_parsed"] = data["embeddings"].apply(parse_embedding)

# Stack embeddings: shape = (226, 400, 48)
X = np.stack(data["embedding_parsed"].values)
y = data["label"].values  # Labels: 0, 1, 2, or 3

# Normalize the data (feature-wise normalization)
X_mean = np.mean(X, axis=0, keepdims=True)
X_std = np.std(X, axis=0, keepdims=True)
X = (X - X_mean) / (X_std + 1e-8)

# Split data into train (70%), validation (15%), and test (15%) sets (using stratification)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print("Train shape:", X_train.shape, "Validation shape:", X_val.shape, "Test shape:", X_test.shape)

# Create a custom Dataset class (for multi-class, targets are Long tensors)
class EmbeddingsDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)  # shape: (N, 400, 48)
        self.y = torch.tensor(y, dtype=torch.long)       # shape: (N,)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

batch_size = 16  # Adjusted for the small dataset size
train_dataset = EmbeddingsDataset(X_train, y_train)
val_dataset   = EmbeddingsDataset(X_val, y_val)
test_dataset  = EmbeddingsDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# -------------------------------
# 2. Model Building: Fine-Tuning the Transformer
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

# Define a new Transformer classifier for fine-tuning (4-class output)
class TransformerClassifierFT(nn.Module):
    def __init__(self, d_model=48, nhead=4, num_layers=3, ff_dim=128, dropout=0.2, seq_length=400, num_classes=4):
        super(TransformerClassifierFT, self).__init__()
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
        self.ln1 = nn.LayerNorm(128)
        self.fc2 = nn.Linear(128, 64)
        self.ln2 = nn.LayerNorm(64)
        self.fc3 = nn.Linear(64, num_classes)   # Adjusted to output 4 logits
        self.relu = nn.ReLU()

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # (seq_len, batch_size, d_model)
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)      # Global average pooling over the sequence: (batch_size, d_model)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.ln1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.ln2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x  # raw logits (no activation; CrossEntropyLoss applies softmax)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerClassifierFT().to(device)

# -------------------------------
# 3. Loading Pre-Trained Weights
# -------------------------------
# Load pre-trained weights from the binary classifier, ignoring the final layer (fc3)
pretrained_path = "data/pretrained_transformer.pth"
pretrained_dict = torch.load(pretrained_path, map_location=device)
model_dict = model.state_dict()

# Filter out parameters for fc3 since shapes differ (binary vs. 4-class)
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'fc3' not in k}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)
print("Loaded pre-trained weights (excluding final layer) from", pretrained_path)

# -------------------------------
# 4. Fine-Tuning Setup
# -------------------------------
# For multi-class classification, we use CrossEntropyLoss.
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)  # Lower learning rate for fine-tuning
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
num_epochs = 25

train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

# -------------------------------
# 5. Fine-Tuning Training Loop
# -------------------------------
for epoch in range(num_epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)  # shape: (batch_size, 4)
        loss = criterion(outputs, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        running_loss += loss.item() * X_batch.size(0)
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)
        
    train_loss = running_loss / total
    train_acc = correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    
    # Validation phase
    model.eval()
    val_running_loss, val_correct, val_total = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            val_running_loss += loss.item() * X_batch.size(0)
            preds = torch.argmax(outputs, dim=1)
            val_correct += (preds == y_batch).sum().item()
            val_total += y_batch.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
            
    val_loss = val_running_loss / val_total
    val_acc = val_correct / val_total
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)
    
    scheduler.step(val_loss)
    
    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
    
    # Plot confusion matrix every 5 epochs
    if (epoch + 1) % 5 == 0:
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0,1,2,3], yticklabels=[0,1,2,3])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix at Epoch {epoch+1}')
        plt.show()

# -------------------------------
# 6. Final Evaluation on Test Set
# -------------------------------
model.eval()
test_running_loss, test_correct, test_total = 0.0, 0, 0
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        test_running_loss += loss.item() * X_batch.size(0)
        preds = torch.argmax(outputs, dim=1)
        test_correct += (preds == y_batch).sum().item()
        test_total += y_batch.size(0)
test_loss = test_running_loss / test_total
test_acc = test_correct / test_total
print(f"Test Loss: {test_loss:.4f} - Test Accuracy: {test_acc:.4f}")

# -------------------------------
# 7. Plot Learning Curves
# -------------------------------
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()

plt.subplot(1,2,2)
plt.plot(range(1, num_epochs+1), train_accuracies, label='Train Accuracy')
plt.plot(range(1, num_epochs+1), val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Curve')
plt.legend()

plt.tight_layout()
plt.show()
