import pandas as pd

# Define file paths
left_hand_file = "final_data/left_embeddings_shuffled.csv"
right_hand_file = "final_data/right_embeddings_shuffled.csv"
output_file = "final_data/complete_embeddings.csv"

# Load the two Excel files
df_left = pd.read_csv(left_hand_file)
df_right = pd.read_csv(right_hand_file)

# Merge the dataframes row-wise
df_merged = pd.concat([df_left, df_right], ignore_index=True)

print(len(df_merged))

# Save the merged dataframe to a new Excel file
df_merged.to_csv(output_file, index=False)

print(f"Successfully merged {left_hand_file} and {right_hand_file} into {output_file}")
