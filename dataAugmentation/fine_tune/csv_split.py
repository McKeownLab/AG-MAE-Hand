import pandas as pd
import numpy as np

NPY_ADDRESS = "../data/stmae_embeddings_pd_5.npy"

def get_right_file_name_for_left(left_file_name):
    return left_file_name.replace('left', 'right')

data_left = pd.read_csv('../final_data/left_embeddings_shuffled.csv')
data_right = pd.read_csv('../final_data/right_embeddings_shuffled.csv')

file_names_pretrain = []
file_names_fine_tune = []

shuffled_data_left_indices = np.random.permutation(data_left.index)
shuffled_df_left = data_left.loc[shuffled_data_left_indices].reset_index(drop=True)

file_names_right = data_right['file_name'].to_list()

for index, row in shuffled_df_left.iterrows():
    if(str(row['label'])=="0.0"):
        continue
    corresponding_file_name_right = get_right_file_name_for_left(row['file_name'])
    if(len(file_names_pretrain) < 230):
        file_names_pretrain.append(row['file_name'])
        if corresponding_file_name_right in file_names_right:
            file_names_pretrain.append(corresponding_file_name_right)
    else:
        file_names_fine_tune.append(row['file_name'])
        if corresponding_file_name_right in file_names_right:
            file_names_fine_tune.append(corresponding_file_name_right)
            
print(len(file_names_pretrain))
print(len(file_names_fine_tune))
print(file_names_pretrain[:5])
print(file_names_fine_tune[:5])

df_pretrain = pd.DataFrame({'file_name': [], 'label': [], 'embeddings': []})
df_finetune = pd.DataFrame({'file_name': [], 'label': [], 'embeddings': []})

complete_csv_data = pd.read_csv('../final_data/complete_embeddings.csv')
print(complete_csv_data.columns.tolist())
complete_csv_data = complete_csv_data.drop(['Unnamed: 0', 'tap_indices'], axis=1)
print(complete_csv_data.head())


for pretrain_record_file_name in file_names_pretrain:
    match_records = complete_csv_data[complete_csv_data['file_name']==pretrain_record_file_name]
    df_pretrain = pd.concat([df_pretrain, match_records], ignore_index=True)


for finetune_record_file_name in file_names_fine_tune:
    match_records = complete_csv_data[complete_csv_data['file_name']==finetune_record_file_name]
    for _, match_rec in match_records.iterrows():
        if str(match_rec['label']) == "1.0":
            df_finetune = pd.concat([df_finetune, match_rec.to_frame().T], ignore_index=True)


with open(NPY_ADDRESS, 'rb') as f:
    data = np.load(f, allow_pickle=True)

data_embeddings = list(data.values())[0]  # shape: (460, 400, 21, 16)
data_labels     = list(data.values())[1]
data_indices    = list(data.values())[2]

# Convert labels so that class 4 becomes class 3
data_labels = np.array(data_labels)
data_labels[data_labels == 4] = 3  
data[list(data.keys())[1]] = data_labels


for index, row in df_finetune.iterrows():
    file_name = row['file_name']
    if file_name in data_indices:
        label = data_labels[data_indices.index(file_name)]
        df_finetune.at[index, 'label'] = label
    else:
        label = None  # Or some default value if not found
    

df_pretrain.to_csv('pretrain.csv')
df_finetune.to_csv('finetune.csv')