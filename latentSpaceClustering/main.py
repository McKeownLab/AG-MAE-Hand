import numpy as np
import pandas as pd
import pickle
import pickletools
from collections import Counter

NPY_ADDRESS = './stmae_v2.npy_943MB/stmae_embeddings_pd_4.npy'

with open(NPY_ADDRESS, 'rb') as f:
    data = np.load(NPY_ADDRESS, allow_pickle=True)  # or 'bytes'

print("Shape:", data.keys())  # Dimensions of the array
# print("Data type:", data.dtype)  # Data type of the elements
# print("Array contents:\n", data)  # Optional: Print the full array

# if isinstance(data, dict):
#     print("Keys:", data.keys())
#     # print("Values:", data.values())


keys = list(data.keys())
values = list(data.values())

embeddings = values[0]
labels = values[1]
indices = values[2]

print(type(embeddings))
print(type(labels))
print(type(indices))

print("for embeddings:")
print("shape:", embeddings.shape)
print("dimension:", embeddings.ndim)

print("for labels:")
print("shape:", embeddings.shape)
print("dimension:", embeddings.ndim)

print("for indexes:")
print(len(indices))

print("keys:", keys)

with open("stmae_v1.npy_5GB/indices.txt", "w") as f:
    for element in indices:
        f.write(f"{element}\n")

