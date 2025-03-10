import os

DIRECTORY_PATH = 'final/'

for file_name in os.listdir(DIRECTORY_PATH):
    if file_name.endswith('.csv'):  
        if '_finger_tapping' in file_name:
            new_file_name = file_name.replace('_finger_tapping', '')

            old_file_path = os.path.join(DIRECTORY_PATH, file_name)
            new_file_path = os.path.join(DIRECTORY_PATH, new_file_name)
            
            os.rename(old_file_path, new_file_path)
            print(f"Renamed: {file_name} -> {new_file_name}")
        else:
            print(f"No '_finger_tapping' in: {file_name}, skipping.")
