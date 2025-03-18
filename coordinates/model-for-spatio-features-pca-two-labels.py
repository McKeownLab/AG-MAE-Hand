import numpy as np
import torch
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

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

# Create a scatter plot
plt.figure(figsize=(8, 6))

# Plot for label 0 (class 0)
plt.scatter(reduced_coordinates[labels == 0, 0], reduced_coordinates[labels == 0, 1], label='Class 0', color='blue', alpha=0.6)

# Plot for label 1 (class 1) (this includes labels 1, 2, 3, and 4)
plt.scatter(reduced_coordinates[labels == 1, 0], reduced_coordinates[labels == 1, 1], label='Class 1', color='red', alpha=0.6)

# Add labels and title
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Hand Joint Coordinates (Class 0 vs Class 1)')
plt.legend()
plt.grid(True)
plt.show()
