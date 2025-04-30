import os
import numpy as np
import pandas as pd
import shutil

ALL_PLOTS_LOCATION = './data/plots/all'
LEFT_PLOTS_LOCATION = './data/plots/left'
RIGHT_PLOTS_LOCATION = './data/plots/right'
RESULTS_DIRECTORY = './data/results/'
SUPERVISION_FILE = './data/weak_supervision_results.csv'


def check_id_integrity(left_plots_location, right_plots_location):
    left_ids = {file.replace("_left_finger_tapping_distances.png", "") for file in os.listdir(left_plots_location) if file.endswith("_left_finger_tapping_distances.png")}
    right_files = set(os.listdir(right_plots_location))

    missing_right_files = [id for id in left_ids if f"{id}_right_finger_tapping_distances.png" not in right_files]
    if missing_right_files:
        print("Files missing in right directory for IDs:", missing_right_files)
    else:
        print("All files have corresponding right files.")
 
 
def seperate_by_supervision_results(supervision_file, results_dir, left_plots_dir, right_plots_dir, plots_dir):
    left_ids = {file.replace("_left_finger_tapping_distances.png", "_left") for file in os.listdir(left_plots_dir) if file.endswith("_left_finger_tapping_distances.png")}
    right_ids = {file.replace("_right_finger_tapping_distances.png", "_right") for file in os.listdir(right_plots_dir) if file.endswith("_right_finger_tapping_distances.png")}
    all_plots_id = left_ids | right_ids
    
    print('total plots number:', len(all_plots_id))
    
    supervision_df = pd.read_csv(supervision_file)
    filtered_rows = supervision_df[supervision_df["ID"].isin(all_plots_id)]
    predictions = filtered_rows[["ID", "Prediction"]]
    print('predictions len:', len(predictions))
    
    for file_name in os.listdir(plots_dir):
        if file_name.endswith(".png"):
            file_id = file_name.replace("_finger_tapping_distances.png", "")
            match = supervision_df[supervision_df["ID"] == file_id]  
        
            if not match.empty:
                prediction = match["Prediction"].values[0]
                target_dir = os.path.join(results_dir, str(prediction))
            else:
                target_dir = os.path.join(results_dir, "unknown")
                
            os.makedirs(target_dir, exist_ok=True)  
            shutil.copy(os.path.join(plots_dir, file_name), os.path.join(target_dir, file_name))  

    

# check_id_integrity(LEFT_PLOTS_LOCATION, RIGHT_PLOTS_LOCATION)

seperate_by_supervision_results(SUPERVISION_FILE, RESULTS_DIRECTORY, LEFT_PLOTS_LOCATION, RIGHT_PLOTS_LOCATION, ALL_PLOTS_LOCATION)