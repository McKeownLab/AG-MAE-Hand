import numpy as np

def find_spatio_features(input_finger_tappings):

    new_features = []

    for i in range(len(input_finger_tappings)):
        joint_4 = np.array(input_finger_tappings[i][:, :, 4])
        joint_8 = np.array(input_finger_tappings[i][:, :, 8])
        
        distance = np.linalg.norm(joint_8 - joint_4, axis=1)
        new_features.append(distance)
    return new_features

def create_new_dataset(splits, new_npy_path):
    new_split_sets = []
    
    for i in range(len(splits)):
        new_train = find_spatio_features(splits[i]['train']['coordinates'])
        new_val = find_spatio_features(splits[i]['val']['coordinates'])
        new_test = find_spatio_features(splits[i]['test']['coordinates'])

        new_split_sets.append({
            'train': {"file_names": splits[i]['train']['file_names'], "labels": splits[i]['train']['labels'], "coordinates": new_train},
            'val': {"file_names": splits[i]['val']['file_names'], "labels": splits[i]['val']['labels'], "coordinates": new_val},
            'test': {"file_names": splits[i]['test']['file_names'], "labels": splits[i]['test']['labels'], "coordinates": new_test}

        })
    np.save(new_npy_path, new_split_sets)

split_file = "../../Datasets/train_val_test_splits.npy"
splits = np.load(split_file, allow_pickle=True)

create_new_dataset(splits, '../../Datasets/spatio-features.npy')