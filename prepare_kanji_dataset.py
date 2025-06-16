import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from collections import Counter
import tensorflow_datasets as tfds

#constants
NUM_KANJI_CLASSES = 10
OUTPUT_DIR = 'kanji_dataset_processed' 
VALIDATION_SPLIT = 0.1 
TEST_SPLIT = 0.1    

print("Downloading Kuzushiji-MNIST dataset using tensorflow_datasets...")
(ds_train, ds_test), ds_info = tfds.load(
    'kmnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
    data_dir='tfds_data' 
)

# Chuyển đổi Dataset sang numpy arrays
x_train_full_list = []
y_train_full_list = []
for image, label in tfds.as_numpy(ds_train):
    x_train_full_list.append(image)
    y_train_full_list.append(label)
x_train_full = np.array(x_train_full_list)
y_train_full = np.array(y_train_full_list)

x_test_raw_list = []
y_test_raw_list = []
for image, label in tfds.as_numpy(ds_test):
    x_test_raw_list.append(image)
    y_test_raw_list.append(label)
x_test_raw = np.array(x_test_raw_list)
y_test_raw = np.array(y_test_raw_list)

print(f"training data shape: {x_train_full.shape}, training labels shape: {y_train_full.shape}")
print(f"test data shape: {x_test_raw.shape}, test labels shape: {y_test_raw.shape}")
print(f"\nFiltering to {NUM_KANJI_CLASSES} Kanji classes...")

x_full = np.concatenate((x_train_full, x_test_raw), axis=0)
y_full = np.concatenate((y_train_full, y_test_raw), axis=0)

mask = y_full < NUM_KANJI_CLASSES
x_filtered = x_full[mask]
y_filtered = y_full[mask]

print(f"Filtered data shape: {x_filtered.shape}, Filtered labels shape: {y_filtered.shape}")
print("Class distribution after filtering:", Counter(y_filtered))

x_temp, x_test, y_temp, y_test = train_test_split(x_filtered, y_filtered, 
                                                    test_size=TEST_SPLIT, 
                                                    random_state=42, 
                                                    stratify=y_filtered) 

validation_size_relative_to_temp = VALIDATION_SPLIT / (1 - TEST_SPLIT)
x_train, x_val, y_train, y_val = train_test_split(x_temp, y_temp, 
                                                  test_size=validation_size_relative_to_temp, 
                                                  random_state=42, 
                                                  stratify=y_temp)

print(f"\nDataset Splits:")
print(f"Train samples: {len(x_train)}")
print(f"Validation samples: {len(x_val)}")
print(f"Test samples: {len(x_test)}")

os.makedirs(OUTPUT_DIR, exist_ok=True)
np.save(os.path.join(OUTPUT_DIR, 'x_train.npy'), x_train)
np.save(os.path.join(OUTPUT_DIR, 'y_train.npy'), y_train)
np.save(os.path.join(OUTPUT_DIR, 'x_val.npy'), x_val)
np.save(os.path.join(OUTPUT_DIR, 'y_val.npy'), y_val)
np.save(os.path.join(OUTPUT_DIR, 'x_test.npy'), x_test)
np.save(os.path.join(OUTPUT_DIR, 'y_test.npy'), y_test)

print(f"\Saved to '{OUTPUT_DIR}' directory.")

