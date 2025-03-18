import numpy as np

def get_distance(joint1_ind, joint2_ind, input_finger_tappings, index):
        joint1 = np.array(input_finger_tappings[index][:, :, joint1_ind])
        joint2 = np.array(input_finger_tappings[index][:, :, joint2_ind])
        
        distance = np.linalg.norm(joint1 - joint2, axis=1)
        return distance

def find_spatio_features(input_finger_tappings):

    new_features = []

    for i in range(len(input_finger_tappings)):
        new_feature = []
        new_feature.append(get_distance(5, 8, input_finger_tappings, i))
        new_feature.append(get_distance(4, 8, input_finger_tappings, i))
        new_feature.append(get_distance(1, 4, input_finger_tappings, i))
        new_feature.append(get_distance(0, 5, input_finger_tappings, i))
        new_feature = np.array(new_feature)
        new_feature = np.transpose(new_feature)
        new_features.append(new_feature)

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