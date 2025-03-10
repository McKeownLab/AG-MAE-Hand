import pandas as pd
import numpy as np
import os
import shutil

READ_DATASET_LOCATION           = 'original/'
WRITE_DATASET_LOCATION          = 'tmp/'
FINAL_DATASET_LOCATION          = 'final/'
MULTIPLE_HAND_IDS_THRESHOLD     = 0.3
MAX_FRAME_DISPARITY_THRESHOLD   = 2


def check_for_duplicates_in_frames(dataset_loc):
    duplicates_cnt = 0
    for file_name in os.listdir(dataset_loc):
        if file_name.endswith('.csv'): 
            file_path = os.path.join(dataset_loc, file_name)
            data = pd.read_csv(file_path)
            
            frames_list = data['frame_number'].tolist()
            if len(frames_list) != len(set(frames_list)):
                print("The list has duplicates. File Name:", file_name)
                duplicates_cnt += 1
            else:
                print("The list has no duplicates.")
    print("total CSVs with duplicate frames:", duplicates_cnt)


def check_max_frame_number_disparity(dataset_loc):
    results = {}
    for file_name in os.listdir(dataset_loc):
        if file_name.endswith('.csv'):  # Check if the file is a CSV
            file_path = os.path.join(dataset_loc, file_name)
            data = pd.read_csv(file_path)

            if 'frame_number' in data.columns:
                frame_differences = data['frame_number'].diff().dropna()  # .diff() computes differences
                max_difference = frame_differences.max()
                results[file_name] = max_difference

                # print(f"File: {file_name}, Max Frame Difference: {max_difference}")
            else:
                print(f"File: {file_name} does not have a 'frame_number' column. Skipping.")

    print("\nSummary of Results:")
    for file, max_diff in results.items():
        print(f"{file}: Max Frame Difference = {max_diff}")

    if results:
        overall_max_difference = max(results.values())
        print(f"\nThe maximum difference across all files is: {overall_max_difference}")


def check_frames_length_in_dataset(dataset_loc):
    less_than_400_cnt = 0
    for file_name in os.listdir(dataset_loc):
        if file_name.endswith('.csv'): 
            file_path = os.path.join(dataset_loc, file_name)
            data = pd.read_csv(file_path)
            if data.shape[0] < 400:
                less_than_400_cnt += 1
            
    print("total CSVs with less than 400 frames:", less_than_400_cnt)


def clean_for_hand_labels(read_dataset_loc, write_dataset_loc):
    cnt=0
    for file_name in os.listdir(read_dataset_loc):
        if file_name.endswith('.csv'):
            file_path = os.path.join(read_dataset_loc, file_name)
            data = pd.read_csv(file_path)

            non_zero_count = (data['hand_id'] != 0).sum()
            total_rows = len(data)
            if non_zero_count != 0:
                if(non_zero_count/total_rows) > MULTIPLE_HAND_IDS_THRESHOLD:
                    cnt += 1
                else:
                    if 'left' in file_name:
                        filtered_df = data[data['hand_label'] == 'Left']
                        filtered_df.to_csv(os.path.join(write_dataset_loc, file_name))
                    elif 'right' in file_name:
                        filtered_df = data[data['hand_label'] == 'Right']
                        filtered_df.to_csv(os.path.join(write_dataset_loc, file_name))
                    else:
                        print('an exception occurs')
            else:
                data.to_csv(os.path.join(write_dataset_loc, file_name))


def clean_for_hand_ids(read_dataset_loc, write_dataset_loc):
    for file_name in os.listdir(read_dataset_loc):
        if file_name.endswith('.csv'): 
            file_path = os.path.join(read_dataset_loc, file_name)
            data = pd.read_csv(file_path)

            most_repeated_hand_id = data['hand_id'].mode()[0]
            # print(f"Processing {file_name}: Most repeated hand_id is {most_repeated_hand_id}")

            filtered_data = data[data['hand_id'] == most_repeated_hand_id]
            filtered_file_path = os.path.join(write_dataset_loc, file_name)
            filtered_data.to_csv(filtered_file_path, index=False)
            # print(f"Filtered file saved as: {filtered_file_path}")
    
    
def remove_files_with_high_frame_disparity(read_dataset_loc, write_dataset_loc, max_frame_disparity_threshold):
    results = {}
    for file_name in os.listdir(read_dataset_loc):
        if file_name.endswith('.csv'):
            file_path = os.path.join(read_dataset_loc, file_name)
            data = pd.read_csv(file_path)

            if 'frame_number' in data.columns:
                frame_differences = data['frame_number'].diff().dropna()  # .diff() computes differences
                max_difference = frame_differences.max()

                results[file_name] = max_difference

                if max_difference < max_frame_disparity_threshold:
                    target_path = os.path.join(write_dataset_loc, file_name)
                    shutil.copy(file_path, target_path)
                    print(f"File {file_name} moved to {write_dataset_loc}.")
                else:
                    print(f"File {file_name} exceeds the threshold and was not moved.")
            else:
                print(f"File {file_name} does not have a 'frame_number' column. Skipping.")


# clean_for_hand_labels(READ_DATASET_LOCATION, WRITE_DATASET_LOCATION)

# clean_for_hand_ids(WRITE_DATASET_LOCATION, WRITE_DATASET_LOCATION)

# check_frames_length_in_dataset(WRITE_DATASET_LOCATION)

# check_for_duplicates_in_frames(WRITE_DATASET_LOCATION)

# check_max_frame_number_disparity(WRITE_DATASET_LOCATION)

# remove_files_with_high_frame_disparity(WRITE_DATASET_LOCATION, FINAL_DATASET_LOCATION, MAX_FRAME_DISPARITY_THRESHOLD)

check_for_duplicates_in_frames(FINAL_DATASET_LOCATION)

check_max_frame_number_disparity(FINAL_DATASET_LOCATION)