import numpy as np

def load_stored_data(file_path):
    """
    Load stored data with labels and indexes from a file.

    Parameters:
    - file_path (str): Path to the .npy file containing the data.

    Returns:
    - embeddings (np.ndarray): Array of stored embeddings.
    - labels (np.ndarray): Array of corresponding labels.
    - indexes (list): List of indexes (metadata for each embedding).
    """
    # Load the .npy file
    data = np.load(file_path, allow_pickle=True)
    print("Loaded data type:", type(data))
    if isinstance(data, dict):
        print("Keys:", data.keys())
    elif isinstance(data, np.ndarray):
        print("Shape:", data.shape)
    else:
        print("Unexpected data type:", type(data))
    # Extract embeddings, labels, and indexes
    embeddings = data.get("embeddings")
    labels = data.get("labels")
    indexes = data.get("indexes")

    return embeddings, labels, indexes

if __name__ == "__main__":
    # Path to the saved data file
    file_path = "/home/atefeh/AG-MAE/experiments/shrec21/asl_right_1/stmae_embeddings_pd_2.npy"  # Update with the correct path

    # Load the data
    embeddings, labels, indexes = load_stored_data(file_path)

    # Print some information
    print("Loaded data:")
    print(f"- Embeddings shape: {embeddings.shape}")
    print(f"- Labels shape: {labels.shape}")
    print(f"- Number of indexes: {len(indexes)}")

