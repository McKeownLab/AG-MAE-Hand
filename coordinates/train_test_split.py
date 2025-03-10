import numpy as np
import os
import random
from sklearn.model_selection import train_test_split

# Load .npy data
NPY_FILE_LOCATION = 'data/coordinates.npy'  # Change to your actual file path
loaded_data = np.load(NPY_FILE_LOCATION, allow_pickle=True).item()

file_names = np.array(loaded_data["file_names"])  # Ensure this is a NumPy array
labels = np.array(loaded_data["labels"])
coordinates = np.array(loaded_data["coordinates"], dtype=object)  # Handle varying lengths

# Extract unique base names (without _Right or _Left)
base_name_to_files = {}
for file in file_names:
    base_name = file.rsplit('_', 1)[0]  # Remove _Right or _Left
    if base_name not in base_name_to_files:
        base_name_to_files[base_name] = []
    base_name_to_files[base_name].append(file)

# Convert to list and shuffle for random splitting
random.seed(42)

# Number of splits
NUM_SPLITS = 3
split_sets = []

for split_num in range(NUM_SPLITS):
    base_names_list = list(base_name_to_files.keys())
    random.shuffle(base_names_list)

    # Train-test-validation split (70:15:15)
    train_base, temp_base = train_test_split(base_names_list, test_size=0.3, random_state=split_num)
    val_base, test_base = train_test_split(temp_base, test_size=0.5, random_state=split_num)

    # Helper function to retrieve files, labels, and coordinates
    def get_data_splits(base_list):
        split_files = []
        split_labels = []
        split_coords = []
        
        for base in base_list:
            for file in base_name_to_files[base]:  # Get _Right and/or _Left files
                index = np.where(file_names == file)[0][0]  # Find index correctly
                split_files.append(file)
                split_labels.append(labels[index])
                split_coords.append(coordinates[index])  # Shape (num_rows, 3, 21)

        return split_files, split_labels, split_coords

    # Get train, val, test sets
    train_files, train_labels, train_coords = get_data_splits(train_base)
    val_files, val_labels, val_coords = get_data_splits(val_base)
    test_files, test_labels, test_coords = get_data_splits(test_base)

    # Store the split
    split_sets.append({
        "train": {"file_names": train_files, "labels": train_labels, "coordinates": train_coords},
        "val": {"file_names": val_files, "labels": val_labels, "coordinates": val_coords},
        "test": {"file_names": test_files, "labels": test_labels, "coordinates": test_coords}
    })

    print(f"Split {split_num + 1}: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")

# Save the splits for later use
SPLIT_FILE = "train_val_test_splits.npy"
np.save(SPLIT_FILE, split_sets)

print(f"Splits saved to {SPLIT_FILE}")

# Sample code to load splits
print("\nSample code to read splits:")
print("""
import numpy as np

SPLIT_FILE = 'train_val_test_splits.npy'
split_sets = np.load(SPLIT_FILE, allow_pickle=True)

# Example: Get the first split
split_1 = split_sets[0]
print("Train files in first split:", split_1['train']['file_names'][:5])  # Print first 5 train files
print("First file label:", split_1['train']['labels'][0])
print("First file coordinate shape:", split_1['train']['coordinates'][0].shape)  # (num_rows, 3, 21)
""")
