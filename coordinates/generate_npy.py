import os
import numpy as np
import pandas as pd

# Paths
CSV_FOLDER  = "data/final"  # Change to your CSV folder
NPY_ADDRESS = "data/stmae_embeddings_pd_5.npy" # Change to your .npy file location
OUTPUT_NPY  = "coordinates.npy"

# Load .npy file
with open(NPY_ADDRESS, 'rb') as f:
    data = np.load(f, allow_pickle=True)

# Extract data
data_indices = list(data.values())[2]  # Names of the files (without .csv)
data_labels = list(data.values())[1]   # Corresponding labels

# Convert class 4 to class 3
data_labels = np.array(data_labels)
data_labels[data_labels == 4] = 3  

# Dictionary to store results
data_dict = {
    "file_names": [],
    "labels": [],
    "coordinates": []
}

# Columns to ignore
IGNORE_COLUMNS = ['Unnamed: 0', 'frame_number', 'hand_id', 'hand_label', 'hand_width', 'hand_height']

# Iterate through CSV files in the folder
for filename in os.listdir(CSV_FOLDER):
    if filename.endswith(".csv"):
        file_base_name = filename.rsplit('.', 1)[0]  # Remove .csv extension
        # print(file_base_name)
        
        # file_base_name = file_base_name.replace('_finger_tapping', '')
        # Find corresponding label
        if file_base_name in data_indices:
            label_index = data_indices.index(file_base_name)
            label = data_labels[label_index]
            
            # Read the CSV file
            file_path = os.path.join(CSV_FOLDER, filename)
            df = pd.read_csv(file_path)

            # Drop unnecessary columns
            df = df.drop(columns=[col for col in IGNORE_COLUMNS if col in df.columns], errors='ignore')

            # Reshape data into (num_rows, 3, 21)
            print("# of columns", df.shape[1])
            num_rows = df.shape[0]
            coords_array = df.values.reshape(num_rows, 21, 3).transpose(0, 2, 1)  # Shape (num_rows, 3, 21)

            # Store data
            data_dict["file_names"].append(filename)
            data_dict["labels"].append(label)
            data_dict["coordinates"].append(coords_array)

# Convert lists to NumPy arrays for efficient storage
data_dict["coordinates"] = np.array(data_dict["coordinates"], dtype=object)  # Different row counts -> object array
data_dict["labels"] = np.array(data_dict["labels"])
data_dict["file_names"] = np.array(data_dict["file_names"])

# Save dictionary as .npy file

np.save(OUTPUT_NPY, data_dict)

print(f"Data saved to {OUTPUT_NPY}")

# Sample code to load and read the .npy file
print("\nSample code to read the .npy file:")
print("""
import numpy as np

# Load data
loaded_data = np.load('output_data.npy', allow_pickle=True).item()

# Access elements
file_names = loaded_data['file_names']
labels = loaded_data['labels']
coordinates = loaded_data['coordinates']

# Example usage
print(f"First file: {file_names[0]}, Label: {labels[0]}")
print("Shape of first file's coordinates:", coordinates[0].shape)  # (num_rows, 3, 21)
""")
