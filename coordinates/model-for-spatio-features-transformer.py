import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import copy
import random

BATCH_SIZE = 32
SPLIT_INDEX = 0
LEARNING_RATE = 0.001

def augment_sample(sample, noise_level=0.02):

    sample = np.array(sample)  # Ensure it's a NumPy array
    noise = np.random.normal(loc=0, scale=noise_level, size=sample.shape)  # Small Gaussian noise
    return (sample + noise).tolist()  # Convert back to list

class SpatioFeaturesDataset(Dataset):
    def __init__(self, data, mean=None, std=None, seq_length=100):
        self.data = data
        self.seq_length = seq_length
        self.mean = mean
        self.std = std

        if mean is None or std is None:
            self.compute_mean_std()

        # If balance_data is True, perform undersampling to balance the classes

    def compute_mean_std(self):
        all_coordinates = []

        for sample in self.data['coordinates']:
            x = np.array(sample)
            if len(x) > self.seq_length:
                x = x[:self.seq_length]
            else:
                x = np.pad(x, ((0, 100 - len(x)), (0, 0)), mode='constant', constant_values=0.0)

            all_coordinates.append(x)

        all_coordinates = np.array(all_coordinates)
        self.mean = np.mean(all_coordinates, axis=0)
        self.std = np.std(all_coordinates, axis=0)

    def __len__(self):
        return len(self.data['coordinates'])

    def __getitem__(self, idx):
        x, y = self.data["coordinates"][idx], self.data["labels"][idx]
        x = np.array(x)

        if len(x) > self.seq_length:
            x = x[:self.seq_length]
        else:
            x = np.pad(x, ((0, 100 - len(x)), (0, 0)), mode='constant', constant_values=0.0)

        # Z-score normalization
        x = (x - self.mean) / (self.std + 1e-6)
        
        y = 0 if y == 0 else 1

        
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).long()


class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=6, num_heads=8, ff_dim=512, dropout=0.1):
        super(TransformerModel, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.zeros(1, 100, input_dim))  # Assuming seq_length is 400

        # Transformer layers
        self.transformer = nn.Transformer(
            d_model=input_dim, 
            nhead=num_heads, 
            num_encoder_layers=num_layers, 
            dim_feedforward=ff_dim, 
            dropout=dropout, 
            batch_first=True
        )
        
        self.fc = nn.Linear(input_dim, hidden_dim)
        # Final layer for binary classification (one output)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Add positional encoding to the input sequence
        x = x + self.positional_encoding

        # Pass through the Transformer
        x = self.transformer(x, x)  # Self-attention on the input
        
        # Use the output from the last token in the sequence
        x = x[:, -1, :]  # Take the last output (corresponding to the final position in the sequence)

        # Fully connected layers
        x = self.fc(x)
        x = self.fc_out(x)
        return x


def calculate_class_weights(dataset):
    # Calculate the number of occurrences of each class in the dataset
    class_counts = {0: 0, 1: 0}
    for _, label in dataset:
        class_counts[label.item()] += 1

    total_samples = len(dataset)
    
    # Calculate the inverse frequency weights
    class_weights = {
        0: total_samples / class_counts[0],
        1: total_samples / class_counts[1]
    }

    # Normalize weights so they sum to 1 or similar (optional)
    total_weight = class_weights[0] + class_weights[1]
    normalized_class_weights = torch.tensor([
        class_weights[0] / total_weight,
        class_weights[1] / total_weight
    ])

    return normalized_class_weights

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs = 20):
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        for x, y in train_loader:   
            optimizer.zero_grad()
            outputs = model(x)

            # Flatten the outputs to match target labels
            loss = criterion(outputs.squeeze(), y.float())  # Squeeze the outputs to shape [batch_size]
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Accuracy Calculation for training
            _, predicted = torch.max(outputs, 1)
            total_train += y.size(0)
            correct_train += (predicted == y).sum().item()

            
        train_losses.append(running_loss / len(train_loader))
        train_accuracy = 100 * correct_train / total_train

        model.eval()
        val_loss, y_true, y_pred = 0.0, [], []
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for x, y in val_loader:
                outputs = model(x)
                # Flatten the outputs to match target labels
       
                loss = criterion(outputs.squeeze(), y.float())  # Squeeze the outputs to shape [batch_size]
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total_val += y.size(0)
                correct_val += (predicted == y).sum().item()

                y_true.extend(y.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
                
        val_losses.append(val_loss / len(val_loader))
        val_accuracy = 100 * correct_val / total_val

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, Train Accuracy: {train_accuracy:.2f}%, Val Accuracy: {val_accuracy:.2f}%')

        if epoch % 5 == 0:
            cm = confusion_matrix(y_true, y_pred)
            ConfusionMatrixDisplay(cm).plot()
            plt.title(f'Confusion Matrix - Epoch {epoch+1}')
            plt.show()

    return train_losses, val_losses


def balance_dataset(data, noise_level=0.02):

    balanced_data = copy.deepcopy(data)
    coordinates = balanced_data['coordinates']
    labels = [0 if label == 0 else 1 for label in balanced_data['labels']]  # Map 1,2,3,4 to 1

    class_0_indices = [i for i, label in enumerate(labels) if label == 0]
    class_1_indices = [i for i, label in enumerate(labels) if label == 1]

    num_class_0, num_class_1 = len(class_0_indices), len(class_1_indices)

    # Oversample class 0 if it has fewer samples than class 1
    if num_class_0 < num_class_1 and num_class_0 > 0:
        extra_indices = np.random.choice(class_0_indices, size=(num_class_1 - num_class_0), replace=True)
        augmented_samples = [augment_sample(coordinates[i], noise_level) for i in extra_indices]
        
        coordinates.extend(augmented_samples)
        labels.extend([0] * len(augmented_samples))

    # Shuffle the dataset to mix new samples
    combined = list(zip(coordinates, labels))
    random.shuffle(combined)
    coordinates, labels = zip(*combined)

    return {'coordinates': list(coordinates), 'labels': list(labels)}

def evaluate_model(model, data_loader, title):
    y_true, y_pred = [], []
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in data_loader:
            outputs = model(x)
            _, predicted = torch.max(outputs, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
                
            y_true.extend(y.cpu().numpy())  
            y_pred.extend(predicted.cpu().numpy())

    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')

    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(cm).plot()
    plt.title(title)
    plt.show()

# Load the data
split_file = "../../Datasets/spatio-features.npy"
splits = np.load(split_file, allow_pickle=True)
split0 = splits[SPLIT_INDEX]

train_data = split0["train"]
val_data = split0["val"]
test_data = split0["test"]

train_data = balance_dataset(split0["train"])

train_dataset = SpatioFeaturesDataset(train_data)
val_dataset = SpatioFeaturesDataset(val_data)
test_dataset = SpatioFeaturesDataset(test_data)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

class_weights = calculate_class_weights(train_dataset)
print(class_weights)

# Initialize the model
model = TransformerModel(input_dim=6, hidden_dim=64, output_dim=1, num_heads=2)  # Output dimension = 1 for binary classification

# Use BCEWithLogitsLoss for binary classification
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Train the model
train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer)

# Plot the loss curves
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.title('Loss Curve')
plt.legend()
plt.show()

# Evaluate the model
evaluate_model(model, val_loader, 'Final Validation Confusion Matrix')
evaluate_model(model, test_loader, 'Final Test Confusion Matrix')
