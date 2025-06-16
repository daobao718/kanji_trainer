import numpy as np
import os
import tensorflow as tf 

#Constants
DATA_DIR = 'kanji_dataset_processed' 
NUM_KANJI_CLASSES = 10 

print("Starting data preprocessing...")

try:
    x_train = np.load(os.path.join(DATA_DIR, 'x_train.npy'))
    y_train = np.load(os.path.join(DATA_DIR, 'y_train.npy'))
    x_val = np.load(os.path.join(DATA_DIR, 'x_val.npy'))
    y_val = np.load(os.path.join(DATA_DIR, 'y_val.npy'))
    x_test = np.load(os.path.join(DATA_DIR, 'x_test.npy'))
    y_test = np.load(os.path.join(DATA_DIR, 'y_test.npy'))
    print("Dataset loaded successfully from .npy files.")
except FileNotFoundError:
    print(f"Error: Could not find dataset files in '{DATA_DIR}'.")
    print("Need to run 'prepare_kanji_dataset.py' first.")
    exit()

print(f"Original x_train shape: {x_train.shape}")
print(f"Original y_train shape: {y_train.shape}")

#Pixel
x_train = x_train.astype('float32') / 255.0
x_val = x_val.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
print("Pixel values set to [0, 1].")

#channels
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_val = x_val.reshape(x_val.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
print(f"Reshaped x_train shape: {x_train.shape} (added channel dimension).")

#One-Hot Encode 
y_train = tf.keras.utils.to_categorical(y_train, NUM_KANJI_CLASSES)
y_val = tf.keras.utils.to_categorical(y_val, NUM_KANJI_CLASSES)
y_test = tf.keras.utils.to_categorical(y_test, NUM_KANJI_CLASSES)
print(f"Labels one-hot encoded. y_train shape: {y_train.shape}")

print("\nData preprocessing complete.")


