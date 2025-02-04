import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm

# ---------------------------------------------------
# 1. Data Preparation (Flattened)
# ---------------------------------------------------
# Original data shape: (460, 400, 21, 64)
# Flatten to (460, 400*21*64) = (460, 537600)




NPY_ADDRESS = "./data/stmae_v2.npy_943MB/stmae_embeddings_pd_4.npy"

with open(NPY_ADDRESS, 'rb') as f:
    data = np.load(NPY_ADDRESS, allow_pickle=True) 


# step 1: data preprocessing
data_embeddings = list(data.values())[0]
data_labels     = list(data.values())[1]
data_indices    = list(data.values())[2]

# Flatten the data
data_embeddings_flat = data_embeddings.reshape(460, -1)

# Split dataset
X_train, X_temp, y_train, y_temp = train_test_split(data_embeddings_flat, data_labels, 
                                                   test_size=0.3, 
                                                   random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, 
                                               test_size=0.5, 
                                               random_state=42)

# Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_t = torch.FloatTensor(X_train_scaled)
y_train_t = torch.LongTensor(y_train)
X_val_t = torch.FloatTensor(X_val_scaled)
y_val_t = torch.LongTensor(y_val)
X_test_t = torch.FloatTensor(X_test_scaled)
y_test_t = torch.LongTensor(y_test)

# Create DataLoaders
train_dataset = TensorDataset(X_train_t, y_train_t)
val_dataset = TensorDataset(X_val_t, y_val_t)
test_dataset = TensorDataset(X_test_t, y_test_t)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# ---------------------------------------------------
# 2. Simple Feed-Forward Network Architecture
# ---------------------------------------------------
class SimpleNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        return self.layers(x)

# Initialize model
input_size = X_train_scaled.shape[1]
num_classes = len(np.unique(data_labels))
model = SimpleNN(input_size, num_classes)

# ---------------------------------------------------
# 3. Training Configuration
# ---------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

# ---------------------------------------------------
# 4. Training Loop with Early Stopping
# ---------------------------------------------------
best_val_loss = float('inf')
patience = 10
patience_counter = 0

for epoch in tqdm.tqdm(range(100)):
    # Training
    model.train()
    train_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)
    
    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
    
    # Update learning rate
    scheduler.step(val_loss)
    
    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model_simple.pth')
    else:
        patience_counter += 1
    
    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        break

# ---------------------------------------------------
# 5. Evaluation
# ---------------------------------------------------
def evaluate_model(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    return all_labels, all_preds

# Load best model
model.load_state_dict(torch.load('best_model_simple.pth'))

# Test evaluation
test_labels, test_preds = evaluate_model(model, test_loader)

# Calculate metrics
accuracy = accuracy_score(test_labels, test_preds)
recall = recall_score(test_labels, test_preds, average='macro')
f1 = f1_score(test_labels, test_preds, average='macro')
cm = confusion_matrix(test_labels, test_preds)

print(f"\nTest Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Recall:   {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()