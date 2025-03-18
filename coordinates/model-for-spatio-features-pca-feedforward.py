import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from collections import Counter

# Assuming `train_data` is already loaded, and we have the `coordinates` and `labels` as part of the data
# Extract features (coordinates) and labels from the training dataset
coordinates = []
labels = []

split_file = "../../Datasets/spatio-features.npy"
splits = np.load(split_file, allow_pickle=True)
split0 = splits[0]

train_data = split0["train"]
val_data = split0["val"]
test_data = split0["test"]

# Extract coordinates and labels
for sample in train_data['coordinates']:
    x = np.array(sample)  # Shape is (n, 4)
    
    # Handle padding for samples with fewer than 400 frames
    if len(x) > 400:
        x = x[:400]  # Truncate to 400 frames
    else:
        # Pad the data to have 400 samples, each with 4 features (e.g., [x, y, z, w])
        padding_needed = 400 - len(x)  # Calculate how many samples need to be padded
        # Pad at the end of the sequence to get a total of 400 frames
        x = np.pad(x, ((0, padding_needed), (0, 0)), mode='constant', constant_values=0.0)
    
    # Flatten the sample to 1D (400 frames * 4 features = 1600 elements)
    x = x.flatten()
    coordinates.append(x)

# Extract the labels and change labels 2, 3, 4 to 1
for label in train_data['labels']:
    labels.append(label)

# Convert labels to numpy array
labels = np.array(labels)

# Change labels 2, 3, and 4 to be 1 (label 0 remains as 0)
labels[labels > 0] = 1

# Convert coordinates to numpy array
coordinates = np.array(coordinates)

# Apply PCA to reduce the dimensionality of the features
pca = PCA(n_components=2)  # We want to reduce to 2D for visualization
reduced_coordinates = pca.fit_transform(coordinates)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(reduced_coordinates, labels, test_size=0.2, random_state=42)

# Convert the data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Define the neural network architecture
class FCNN(nn.Module):
    def __init__(self):
        super(FCNN, self).__init__()
        self.layer1 = nn.Linear(2, 64)  # Input layer (2 PCA features) -> Hidden layer 1 (64 units)
        self.layer2 = nn.Linear(64, 32) # Hidden layer 2 (32 units)
        self.output_layer = nn.Linear(32, 2) # Output layer (2 classes)
        self.relu = nn.ReLU()  # Activation function

    def forward(self, x):
        x = self.relu(self.layer1(x))  # Apply layer1 and ReLU activation
        x = self.relu(self.layer2(x))  # Apply layer2 and ReLU activation
        x = self.output_layer(x)  # Output layer
        return x

# Initialize the model
model = FCNN()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()  # Loss function for classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the model
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()  # Zero the gradients
    outputs = model(X_train_tensor)  # Forward pass
    loss = criterion(outputs, y_train_tensor)  # Calculate the loss
    loss.backward()  # Backward pass (compute gradients)
    optimizer.step()  # Update weights

    # Calculate training loss and accuracy
    _, y_train_pred = torch.max(outputs, 1)
    train_accuracy = accuracy_score(y_train_tensor.numpy(), y_train_pred.numpy())

    # Evaluate on test set
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        outputs_test = model(X_test_tensor)  # Forward pass on test data
        test_loss = criterion(outputs_test, y_test_tensor)  # Calculate test loss
        _, y_test_pred = torch.max(outputs_test, 1)  # Get the predicted class for test data
        test_accuracy = accuracy_score(y_test_tensor.numpy(), y_test_pred.numpy())

    # Print the loss and accuracy for train and test sets
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, Train Accuracy: {train_accuracy:.4f}, Test Loss: {test_loss.item():.4f}, Test Accuracy: {test_accuracy:.4f}")

# Final evaluation on test set
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    y_pred_tensor = model(X_test_tensor)  # Forward pass
    _, y_pred = torch.max(y_pred_tensor, 1)  # Get the predicted class

# Convert the predictions to numpy for evaluation
y_pred = y_pred.numpy()

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Final Test Accuracy: {accuracy:.4f}")

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Plot the decision boundary and scatter plot
plt.figure(figsize=(8, 6))

# Plot the decision boundary
h = .02  # Step size in the mesh
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = model(torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32))
_, Z = torch.max(Z, 1)
Z = Z.numpy().reshape(xx.shape)

# Plot the decision boundary
plt.contourf(xx, yy, Z, alpha=0.75)

# Scatter plot of the training points
plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], label='Class 0', color='blue', alpha=0.6)
plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], label='Class 1', color='red', alpha=0.6)

# Add labels and title
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('FCNN Classification with PCA-reduced Coordinates')
plt.legend()
plt.grid(True)
plt.show()
