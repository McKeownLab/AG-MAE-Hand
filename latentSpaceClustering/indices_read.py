from collections import Counter
file_path = "indices.txt"

with open(file_path, "r") as f:
    lines = f.read().splitlines()

index_counts = Counter(lines)

print(f"Total unique indices: {len(index_counts)}")
print("Counts for each index:")
total_frames = 0
for index, count in index_counts.items():
    print(f"{index}: {count}")
    total_frames += count

print(total_frames)