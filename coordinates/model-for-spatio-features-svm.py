import numpy as np
import torch
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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

# Calculate class weights based on the distribution of the labels
class_counts = Counter(labels)
total_samples = len(labels)
class_weights = {class_label: total_samples / count for class_label, count in class_counts.items()}

print("Class Counts:", class_counts)
print("Class Weights:", class_weights)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(reduced_coordinates, labels, test_size=0.2, random_state=42)

# Create and train the SVM classifier with manually calculated class weights
svm_clf = SVC(kernel='linear', class_weight=class_weights)  # Use the class weights
svm_clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = svm_clf.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

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
Z = svm_clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary
plt.contourf(xx, yy, Z, alpha=0.75)

# Scatter plot of the training points
plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], label='Class 0', color='blue', alpha=0.6)
plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], label='Class 1', color='red', alpha=0.6)

# Add labels and title
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('SVM Classification with PCA-reduced Coordinates')
plt.legend()
plt.grid(True)
plt.show()
