import numpy as np
import random
from collections import defaultdict
from sklearn.model_selection import train_test_split
import pandas as pd

NUMBER_OF_TAPS = 10

NPY_FILE_LOCATION = '../../Datasets/coordinates.npy'  
TAPPING_LEFT_ADDRESS = '../../Datasets/left_best_segments_and_indexes.csv'
TAPPING_RIGHT_ADDRESS = '../../Datasets/right_best_segments_and_indexes.csv'

loaded_data = np.load(NPY_FILE_LOCATION, allow_pickle=True).item()

right_tappings = pd.read_csv(TAPPING_RIGHT_ADDRESS)
left_tappings = pd.read_csv(TAPPING_LEFT_ADDRESS)

file_names = np.array(loaded_data["file_names"]) 
labels = np.array(loaded_data["labels"])
coordinates = np.array(loaded_data["coordinates"], dtype=object) 
file_names_prev = file_names.copy()

num_loop = len(file_names_prev)

label_sums = defaultdict(int)
label_counts = defaultdict(int)

deleted_indices = []
for idx in range(num_loop):
    new_file_name = file_names_prev[idx][0:-4] + '_finger_tapping_distances.csv'
    if('right' in new_file_name):
        search_pd = right_tappings
    else:
        search_pd = left_tappings
    row_need = search_pd[search_pd['file_name'] == new_file_name]
    if(row_need.empty):
        print(f"cant find {new_file_name}")
    tap_indices = row_need['tap_indices'].apply(lambda x: list(map(int, str(x).split(','))) if pd.notnull(x) else []).tolist()[0]
    if(len(tap_indices) < NUMBER_OF_TAPS):
        deleted_indices.append(idx)
    else:
        # print("prev", len(coordinates[idx]))
        
        coordinates[idx] = coordinates[idx][0:tap_indices[NUMBER_OF_TAPS - 1]]
        label_sums[labels[idx]] += tap_indices[NUMBER_OF_TAPS - 1]
        label_counts[labels[idx]] += 1
        # print("now", len(coordinates[idx]))
        # input()
label_averages = {label: label_sums[label] / label_counts[label] for label in label_sums}
print(label_averages)


file_names = np.delete(file_names, deleted_indices)
labels = np.delete(labels, deleted_indices)
coordinates = np.delete(coordinates, deleted_indices)


base_name_to_files = defaultdict(list)
base_name_to_labels = defaultdict(set)  


for i, file in enumerate(file_names):
    base_name = file.rsplit('_', 1)[0]  
    base_name_to_files[base_name].append(file)
    base_name_to_labels[base_name].add(labels[i]) 

base_names_label_0 = [b for b in base_name_to_labels if 0 in base_name_to_labels[b]]
base_names_label_nz = [b for b in base_name_to_labels if any(l != 0 for l in base_name_to_labels[b])]

random.seed(42)
NUM_SPLITS = 3
split_sets = []

for split_num in range(NUM_SPLITS):
    random.shuffle(base_names_label_0)
    random.shuffle(base_names_label_nz)

    train_size_0 = int(0.7 * len(base_names_label_0))
    train_size_nz = int(0.7 * len(base_names_label_nz))

    train_base_0 = base_names_label_0[:train_size_0]
    train_base_nz = base_names_label_nz[:train_size_nz]

    remaining_0 = base_names_label_0[train_size_0:] 
    remaining_nz = base_names_label_nz[train_size_nz:]

    val_base_0, test_base_0 = train_test_split(remaining_0, test_size=0.5, random_state=split_num)
    val_base_nz, test_base_nz = train_test_split(remaining_nz, test_size=0.5, random_state=split_num)

    train_base = train_base_0 + train_base_nz
    val_base = val_base_0 + val_base_nz
    test_base = test_base_0 + test_base_nz

    def get_data_splits(base_list):
        split_files, split_labels, split_coords = [], [], []
        for base in base_list:
            for file in base_name_to_files[base]: 
                index = np.where(file_names == file)[0][0]  
                split_files.append(file)
                split_labels.append(labels[index])
                split_coords.append(coordinates[index]) 
        return split_files, split_labels, split_coords

    train_files, train_labels, train_coords = get_data_splits(train_base)
    val_files, val_labels, val_coords = get_data_splits(val_base)
    test_files, test_labels, test_coords = get_data_splits(test_base)

    def balance_labels(labels_list, files_list, coords_list):
        """ Ensure equal label 0 and non-zero samples in val and test sets """
        label_0_indices = [i for i, lbl in enumerate(labels_list) if lbl == 0]
        label_nz_indices = [i for i, lbl in enumerate(labels_list) if lbl != 0]

        min_count = min(len(label_0_indices), len(label_nz_indices)) 
        selected_0 = random.sample(label_0_indices, min_count)
        selected_nz = random.sample(label_nz_indices, min_count)

        selected_indices = sorted(selected_0 + selected_nz)

        labels_balanced = [labels_list[i] for i in selected_indices]
        files_balanced = [files_list[i] for i in selected_indices]
        coords_balanced = [coords_list[i] for i in selected_indices]

        return files_balanced, labels_balanced, coords_balanced

    val_files, val_labels, val_coords = balance_labels(val_labels, val_files, val_coords)
    test_files, test_labels, test_coords = balance_labels(test_labels, test_files, test_coords)

    split_sets.append({
        "train": {"file_names": train_files, "labels": train_labels, "coordinates": train_coords},
        "val": {"file_names": val_files, "labels": val_labels, "coordinates": val_coords},
        "test": {"file_names": test_files, "labels": test_labels, "coordinates": test_coords}
    })

    count_0_val = val_labels.count(0)
    count_nz_val = len(val_labels) - count_0_val
    count_0_test = test_labels.count(0)
    count_nz_test = len(test_labels) - count_0_test

    print(f"Split {split_num + 1}: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")
    print(f"    Train - Label 0: {train_labels.count(0)}, Non-zero: {len(train_labels) - train_labels.count(0)}")
    print(f"    Val   - Label 0: {count_0_val}, Non-zero: {count_nz_val}")
    print(f"    Test  - Label 0: {count_0_test}, Non-zero: {count_nz_test}")

    assert count_0_val == count_nz_val, "Validation should have equal label 0 and non-zero samples"
    assert count_0_test == count_nz_test, "Test should have equal label 0 and non-zero samples"

SPLIT_FILE = "../../Datasets/train_val_test_splits.npy"
np.save(SPLIT_FILE, split_sets)

print(f"Splits saved to {SPLIT_FILE}")
