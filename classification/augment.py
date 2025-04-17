import numpy as np
import pandas as pd

def balance_and_augment(X, y, noise_std=0.01, random_state=42):
    np.random.seed(random_state)
    
    # Split by label
    true_mask = y == True
    false_mask = y == False

    X_true = X[true_mask.values]
    X_false = X[false_mask.values]

    y_true = y[true_mask]
    y_false = y[false_mask]

    num_true = len(X_true)
    num_false = len(X_false)

    print(f"True count: {num_true}, False count: {num_false}, Ratio: {num_true / num_false:.2f}")

    # How many times to augment each false sample
    augment_times = (num_true - num_false) // num_false
    extra_needed = (num_true - num_false) % num_false

    new_X = []
    new_y = {}

    # Augment each false sample
    for idx, (original_id, sample) in enumerate(zip(y_false.index, X_false)):
        for i in range(augment_times):
            aug_sample = sample + np.random.normal(0, noise_std, size=sample.shape)
            new_id = f"{original_id}_aug_{i+1}"
            new_X.append(aug_sample)
            new_y[new_id] = False

    # Add remaining extras if needed
    for i in range(extra_needed):
        original_id = y_false.index[i % num_false]
        sample = X_false[i % num_false]
        aug_sample = sample + np.random.normal(0, noise_std, size=sample.shape)
        new_id = f"{original_id}_aug_extra_{i+1}"
        new_X.append(aug_sample)
        new_y[new_id] = False

    # Combine everything
    X_aug = np.vstack([X, np.array(new_X)])
    y_combined = pd.concat([
        y,
        pd.Series(new_y, name="Prediction")
    ])

    return X_aug, y_combined


def report_dataset_balanced_stat(X, y):
    # Count the number of True and False in the new y
    label_counts = y.value_counts()
    print("X len:", len(X))
    print("Label distribution after augmentation:")
    print(label_counts)
