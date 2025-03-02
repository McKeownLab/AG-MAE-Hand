import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import ast
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# -------------------------------
# 1. Data Loading & Preprocessing
# -------------------------------

# Load the Excel file (adjust the file name/path as necessary)
data = pd.read_csv("final_data/left_embeddings_shuffled.csv")

# Parse the embeddings column: assume each cell is a string representation of a 2D list of shape (400, 48)
def parse_embedding(x):
    return np.array(ast.literal_eval(x))

data["embedding_parsed"] = data["embeddings"].apply(parse_embedding)

# Stack the parsed embeddings into a numpy array: shape = (num_samples, 400, 48)
X = np.stack(data["embedding_parsed"].values)
y = data["label"].values  # Binary label: 1 (original) or 0 (shuffled)

print("Data shape:", X.shape)

# Split data into train (70%), validation (15%), and test (15%) sets with stratification
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print("Train shape:", X_train.shape, "Validation shape:", X_val.shape, "Test shape:", X_test.shape)

# Create a custom Dataset class
class EmbeddingsDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)  # shape: (N, 400, 48)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # shape: (N, 1)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Create DataLoaders
batch_size = 32
train_dataset = EmbeddingsDataset(X_train, y_train)
val_dataset   = EmbeddingsDataset(X_val, y_val)
test_dataset  = EmbeddingsDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# -------------------------------
# 2. Model Building: Transformer-Based Classifier
# -------------------------------

# Positional Encoding module
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=400):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)  # shape: (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        return x + self.pe[:, :x.size(1)]

# Define the Transformer classifier model
class TransformerClassifier(nn.Module):
    def __init__(self, d_model=48, nhead=4, num_layers=3, ff_dim=128, dropout=0.2, seq_length=400):
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
        self.ln1 = nn.LayerNorm(128)  # Layer Normalization instead of BatchNorm
        self.fc2 = nn.Linear(128, 64)
        self.ln2 = nn.LayerNorm(64)  # Layer Normalization instead of BatchNorm
        self.fc3 = nn.Linear(64, 1)  
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # (seq_len, batch_size, d_model)
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)  # (batch_size, d_model)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.ln1(x)  # LayerNorm instead of BatchNorm
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.ln2(x)  # LayerNorm instead of BatchNorm
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x  # raw logits


# Initialize model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerClassifier().to(device)
pos_weight = torch.tensor(3.0)  # roughly the ratio of negatives to positives
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# -------------------------------
# 3. Training Loop with Confusion Matrix Callback
# -------------------------------

num_epochs = 50

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    correct_train = 0
    total_train = 0
    
    # Training phase
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)  # shape: (batch_size, 1)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * X_batch.size(0)
        preds = (torch.sigmoid(outputs) > 0.5).float()
        correct_train += (preds == y_batch).sum().item()
        total_train += y_batch.size(0)
        
    train_loss /= total_train
    train_acc = correct_train / total_train
    
    # Validation phase
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
            
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct_val += (preds == y_batch).sum().item()
            total_val += y_batch.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    
    val_loss /= total_val
    val_acc = correct_val / total_val
    
    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f} - " \
          f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
    
    # Every 5 epochs, display the confusion matrix on the validation set
    if (epoch + 1) % 5 == 0:
        cm = confusion_matrix(all_labels, all_preds)
        print(f"\nConfusion Matrix at epoch {epoch+1}:")
        print(cm)
        # print()

# -------------------------------
# 4. Evaluation on Test Set
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
        
        preds = (torch.sigmoid(outputs) > 0.5).float()
        correct_test += (preds == y_batch).sum().item()
        total_test += y_batch.size(0)
        
test_loss /= total_test
test_acc = correct_test / total_test
print("Test Loss:", test_loss)
print("Test Accuracy:", test_acc)
