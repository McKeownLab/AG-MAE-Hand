import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import random
import copy


BATCH_SIZE = 16
SPLIT_INDEX = 0
LEARNING_RATE = 0.01
NUM_FEATURES = 6


def augment_sample(sample, noise_level=0.02):

    sample = np.array(sample)  # Ensure it's a NumPy array
    noise = np.random.normal(loc=0, scale=noise_level, size=sample.shape)  # Small Gaussian noise
    return (sample + noise).tolist()  # Convert back to lisst

class SpatioFeaturesDataset(Dataset):
    def __init__(self, data, mean=None, std=None, seq_length=100):
        self.data = data
        self.seq_length = seq_length
        self.mean = mean
        self.std = std

        if mean is None or std is None:
            self.compute_mean_std()

    def compute_mean_std(self):
        # Compute the mean and std of the input data (training set only)
        all_coordinates = []

        for sample in self.data['coordinates']:
            x = np.array(sample)
            # Pad to fixed length (seq_length) or truncate if necessary
            print(len(x))
            if len(x) > self.seq_length:
                x = x[:self.seq_length]
            else:
                x = np.pad(x, ((0, 400 - len(x)), (0, 0)), mode='constant', constant_values=0.0)
            
            all_coordinates.append(x)

        all_coordinates = np.array(all_coordinates)
        self.mean = np.mean(all_coordinates, axis=0)
        self.std = np.std(all_coordinates, axis=0)

    def __len__(self):
        return len(self.data['coordinates'])

    def __getitem__(self, idx):
        x, y = self.data["coordinates"][idx], self.data["labels"][idx]
        x = np.array(x)

        # Pad/truncate the sequence to a fixed length
        if len(x) > self.seq_length:
            x = x[:self.seq_length]
        else:
            x = np.pad(x, ((0, 400 - len(x)), (0, 0)), mode='constant', constant_values=0.0)

        # Z-score normalization (standardization)
        x = (x - self.mean) / (self.std + 1e-6)  # Add small value to prevent division by zero

        # Change labels: 0 -> 0, and 1,2,3,4 -> 1
        y = 0 if y == 0 else 1
        
        
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


class LSTMModel(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)


    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        out = self.fc2(out)
        return out.squeeze(1)
    
class ComplexLSTMModel(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=5, dropout_prob=0.5, bidirectional=False):
        super(ComplexLSTMModel, self).__init__()
        
        # Define the bidirectional flag, and calculate the output dimension of the LSTM layers
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                            batch_first=True, dropout=dropout_prob, bidirectional=bidirectional)
        
        # Calculate the output dimension based on whether it's bidirectional or not
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # Add additional fully connected layers for more capacity
        self.fc1 = nn.Linear(lstm_output_dim, hidden_dim)  # Hidden layer 1
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # Hidden layer 2
        self.fc3 = nn.Linear(hidden_dim, output_dim)  # Output layer
        
        # Optional: Batch Normalization (after the first fully connected layer)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        
        # Dropout layer after the fully connected layers
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        # Pass the input through LSTM
        lstm_out, (hn, _) = self.lstm(x)
        
        # Get the last hidden state from the LSTM layers (from the last time step)
        if self.bidirectional:
            # Concatenate the last forward and backward hidden states
            hn = torch.cat((hn[-2], hn[-1]), dim=1)
        else:
            hn = hn[-1]

        # Pass through the first fully connected layer + BatchNorm (optional) + Dropout
        out = self.fc1(hn)
        out = self.batch_norm(out)  # Optional: Remove if batch normalization is not needed
        out = torch.relu(out)
        out = self.dropout(out)

        # Pass through the second fully connected layer + Dropout
        out = self.fc2(out)
        out = torch.relu(out)
        out = self.dropout(out)
        
        # Final output layer
        out = self.fc3(out)
        return out.squeeze(1)

    
class GRUModel(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, hn = self.gru(x)
        out = self.fc(hn[-1])
        return out.squeeze(1)
    
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

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs = 50):

    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            predicted = torch.sigmoid(outputs) > 0.5 
            total_train += y.size(0)
            correct_train += (predicted.view(-1) == y).sum().item()  

        train_losses.append(running_loss / len(train_loader))
        train_accuracy = 100 * correct_train / total_train
        # scheduler.step()

        model.eval()
        val_loss, y_true, y_pred = 0.0, [], []
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for x, y in val_loader:
                outputs = model(x)
                loss = criterion(outputs, y)
                val_loss += loss.item()

                predicted = torch.sigmoid(outputs) > 0.5  
                total_val += y.size(0)
                correct_val += (predicted.view(-1) == y).sum().item()

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


def evaluate_model(model, data_loader, title, type_test):

    y_true, y_pred = [], []
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in data_loader:
               outputs = model(x)

               predicted = torch.sigmoid(outputs) > 0.5  
               total += y.size(0)
               correct += (predicted.view(-1) == y).sum().item()

               y_true.extend(y.cpu().numpy())
               y_pred.extend(predicted.cpu().numpy())

    accuracy = 100 * correct / total
    print(f'Accuracy {type_test}: {accuracy:.2f}%')

    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(cm).plot()
    plt.title(title)
    plt.show()

split_file = "../../Datasets/spatio-features.npy"
splits = np.load(split_file, allow_pickle=True)
split0 = splits[SPLIT_INDEX]

train_data = split0["train"]
val_data = split0["test"]
test_data = split0["val"]

train_data = balance_dataset(split0["train"])
# val_data = balance_dataset(split0["val"])
# test_data = balance_dataset(split0["test"])


train_dataset = SpatioFeaturesDataset(train_data)
val_dataset = SpatioFeaturesDataset(val_data)
test_dataset = SpatioFeaturesDataset(test_data)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Calculate class weights based on the training dataset

class_weights = calculate_class_weights(train_dataset)
model = LSTMModel(input_dim= NUM_FEATURES, hidden_dim = 8, output_dim=1)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer)

plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.title('Loss Curve')
plt.legend()
plt.show()

evaluate_model(model, val_loader, 'Final Validation Confusion Matrix', 'validation')
evaluate_model(model, test_loader, 'Final Test Confusion Matrix', 'test')
