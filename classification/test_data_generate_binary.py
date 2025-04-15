import pandas as pd
from collections import Counter

FILE_LABELS_PATH = "./data/weak_supervision_results.csv"

people = ['KW', 'MG', 'SA', 'WM']


def at_least_two_same_label(row):

    labels = row[people].dropna()
    label_counts = labels.value_counts()
    repeated_labels = label_counts[label_counts >= 2]
    if not repeated_labels.empty:
        return repeated_labels.index[0]
    return None

df_labels = pd.read_csv(FILE_LABELS_PATH)
# print(df_labels)

file_labels = []
file_names = []

for _, row in df_labels.iterrows():
    label_row = at_least_two_same_label(row)
    if(label_row is not None):
        if(label_row >= 1):
            label_row = 1
        pred = row['Prediction']
        if(pred >= 1):
            pred = 1
        if(pred == label_row):
            file_labels.append((label_row, row['ID'], pred))
            file_names.append(row['ID'])

file_labels_copy = file_labels.copy()
print(len(file_labels))
for label, file_name, pred in file_labels_copy:
    file_name_other_hand = file_name
    if('right' in file_name):
        file_name_other_hand = file_name_other_hand.replace('right', 'left')
    else:
        file_name_other_hand = file_name_other_hand.replace('left', 'right')
    if(not file_name_other_hand in file_names):
        file_labels.remove((label, file_name, pred))

print(len(file_labels))

list_labels = [label for label, file_name, pred in file_labels]

count_labels = {i:0 for i in range(2)}
test_file = []
for label, file_name, pred in file_labels:
    if(not 'right' in file_name):
        continue
    index = [index for index, (first, second, third) in enumerate(file_labels) if second == file_name.replace('right', 'left')]
    index = index[0]
    label_left = file_labels[index][0]
    pred_left = file_labels[index][2]
    print(label_left)
    if(count_labels[label] < 20 and count_labels[label_left] < 20):
        if(label_left == label and count_labels[label] == 19):
            continue
        test_file.append((file_name, label, pred))
        test_file.append((file_name.replace('right', 'left'), label_left, pred_left))
        count_labels[label] += 1
        count_labels[label_left] += 1

test_df = pd.DataFrame(test_file, columns=['file_name', 'label_neurologists', 'label_predicted'])
test_df.to_csv('./data/test_binary.csv', index=False) 
# with open('test.txt', 'w') as file:
#     for item in test_file:
#         file.write(f"{item}\n") 