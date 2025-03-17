import matplotlib.pyplot as plt
import numpy as np

def jittering(input_finger_tapping):

    jitter = np.random.normal(loc=0, scale = 0.01, size = input_finger_tapping.shape)
    input_finger_tapping_jittered = input_finger_tapping + jitter
    return input_finger_tapping_jittered


def draw_plot_finger_tap(input_coordinates, person_id, applied_augmentation='No'):
    input_finger_tapping = np.array(input_coordinates[person_id])
    
    if applied_augmentation == "Jittering":
        input_finger_tapping = jittering(input_finger_tapping)
    
    joint_4 = np.array(input_finger_tapping[:, :, 4])
    joint_8 = np.array(input_finger_tapping[:, :, 8])
    
    distance = np.linalg.norm(joint_8 - joint_4, axis=1)
    
  
    min_val = np.min(distance)
    max_val = np.max(distance)
    if max_val != min_val:
        normalized_distance = (distance - min_val) / (max_val - min_val)
    else:
        normalized_distance = distance 
    
    plt.figure(figsize=(10, 6))
    plt.plot(normalized_distance, label='Normalized Euclidean Distance')
    plt.title(f'Normalized Euclidean Distance between Joint 4 and Joint 8 over Time for Person ID {person_id} with {applied_augmentation} Augmentation')
    plt.xlabel('Frame')
    plt.ylabel('Normalized Euclidean Distance')
    plt.legend()
    plt.grid(True)
    plt.show()

def add_augmentation_data(new_train, train_data, index, ratio_augmentation, applied_augmentation):

    for i in range(ratio_augmentation):
        if applied_augmentation == "Jittering":       
            new_train['coordinates'].append(jittering(train_data['coordinates'][index]))
        new_train['labels'].append(train_data['labels'][index])
        new_train['file_names'].append(train_data['file_names'][index])

def create_new_dataset(applied_augmentation, splits, ratio_augmentation, path_new_npy):

    new_split_sets = []
    
    for i in range(len(splits)):
        train_data = splits[i]['train']
        new_train = {"file_names": [], "labels": [], "coordinates": []}
        new_train['coordinates'] = train_data['coordinates']

        for j in range(len(train_data['coordinates'])):
            add_augmentation_data(new_train, train_data, j, ratio_augmentation, applied_augmentation)
        new_split_sets.append({
            "train": new_train,
            "val": splits[i]['val'],
            "test": splits[i]['test']
        })
    np.save(path_new_npy, new_split_sets)

split_file = "../../Datasets/train_val_test_splits.npy"
splits = np.load(split_file, allow_pickle=True)
# split0 = splits[0]


# print(split0)
# train_data = split0["train"]
# val_data = split0["val"]
# test_data = split0["test"]

# coordinates = train_data['coordinates']
# draw_plot_finger_tap(coordinates, 1, 'No')
# draw_plot_finger_tap(coordinates, 1, 'Jittering')

create_new_dataset('Jittering', splits, 5, '../../Datasets/jittering.npy')