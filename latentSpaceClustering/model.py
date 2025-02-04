import numpy as np
from sklearn.model_selection import train_test_split
import pickle as pk
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm

NPY_ADDRESS = "./stmae_v2.npy_943MB/stmae_embeddings_pd_4.npy"

with open(NPY_ADDRESS, 'rb') as f:
    data = np.load(NPY_ADDRESS, allow_pickle=True) 


# step 1: data preprocessing
data_embeddings = list(data.values())[0]
data_labels     = list(data.values())[1]
data_indices    = list(data.values())[2]

data_embeddings_flat = data_embeddings.reshape(460, -1)  # New shape: (460, 400*21*64 = 537600)

X_train, X_temp, y_train, y_temp = train_test_split(data_embeddings_flat, data_labels, 
                                                   test_size=0.3, 
                                                   random_state=42)

X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, 
                                               test_size=0.5, 
                                               random_state=42)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Validation set: {X_val.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")


#step 2: SVM model implementation
# Data Standardization
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

# Create DataLoader
train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# SVM Model (Linear)
class SVM(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SVM, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)
        
    def forward(self, x):
        return self.fc(x)

# Initialize model
input_dim = X_train_scaled.shape[1]
num_classes = len(np.unique(data_labels))
model = SVM(input_dim, num_classes)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()  # Using cross-entropy for multi-class classification
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 100
for epoch in tqdm.tqdm(range(num_epochs)):
    model.train()
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

# Evaluation function
def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        outputs = model(X)
        _, preds = torch.max(outputs, 1)
        acc = accuracy_score(y.numpy(), preds.numpy())
        cm = confusion_matrix(y.numpy(), preds.numpy())
    return acc, cm

# Evaluate on all sets
train_acc, train_cm = evaluate(model, X_train_t, y_train_t)
val_acc, val_cm = evaluate(model, X_val_t, y_val_t)
test_acc, test_cm = evaluate(model, X_test_t, y_test_t)

print(f"Training Accuracy: {train_acc:.4f}")
print(f"Validation Accuracy: {val_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

# Plot confusion matrix
def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

plot_confusion_matrix(test_cm, "Test Set Confusion Matrix")