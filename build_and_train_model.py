import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os

#preprrocessed data for model
#Constants
# DATA_DIR = 'kanji_dataset_processed'
# NUM_KANJI_CLASSES = 10

# print("Starting preprocessing for model building...")

# try:
#     x_train = np.load(os.path.join(DATA_DIR, 'x_train.npy'))
#     y_train = np.load(os.path.join(DATA_DIR, 'y_train.npy'))
#     x_val = np.load(os.path.join(DATA_DIR, 'x_val.npy'))
#     y_val = np.load(os.path.join(DATA_DIR, 'y_val.npy'))
#     x_test = np.load(os.path.join(DATA_DIR, 'x_test.npy'))
#     y_test = np.load(os.path.join(DATA_DIR, 'y_test.npy'))
#     print("Dataset loaded successfully ")
# except FileNotFoundError:
#     print(f"Error: Could not find dataset files in '{DATA_DIR}'.")
#     print("Need to run preproceds part first")
#     exit()

# #pixel
# x_train = x_train.astype('float32') / 255.0
# x_val = x_val.astype('float32') / 255.0
# x_test = x_test.astype('float32') / 255.0
# print("Pixel values normalized to [0, 1].")

# #Channel
# x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
# x_val = x_val.reshape(x_val.shape[0], 28, 28, 1)
# x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
# print(f"Reshaped x_train shape: {x_train.shape} (added channel dimension).")

# # One-Hot Encode
# y_train = tf.keras.utils.to_categorical(y_train, NUM_KANJI_CLASSES)
# y_val = tf.keras.utils.to_categorical(y_val, NUM_KANJI_CLASSES)
# y_test = tf.keras.utils.to_categorical(y_test, NUM_KANJI_CLASSES)
# print(f"Labels one-hot encoded. y_train shape: {y_train.shape}")
# print("Data preprocessing complete.\n")

#cnn model
print("Building CNN model...")

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),

    layers.Dense(128, activation='relu'),

    layers.Dense(NUM_KANJI_CLASSES, activation='softmax')
])
model.summary()
print("\nCNN model built successfully.")

#compling
print("\nCompiling model...")
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy']) 

print("Model compiled successfully.")

#tranning
print("\nStarting model training...")

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='best_kanji_model.keras', 
    monitor='val_accuracy', 
    mode='max',
    save_best_only=True,
    verbose=1 
)

early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    mode='max', 
    restore_best_weights=True, 
    verbose=1 
)

history = model.fit(
    x_train, y_train,       
    epochs=20,              
    batch_size=32,       
    validation_data=(x_val, y_val), 
    callbacks=[model_checkpoint_callback, early_stopping_callback],
    verbose=1               
)

print("\nModel training complete.")
print(f"Best validation accuracy achieved: {max(history.history['val_accuracy']):.4f}")

#Evaluation
print("\n--- Model Evaluation ---")

try:
    loaded_model = tf.keras.models.load_model('best_kanji_model.keras')
    print("Loaded best model from 'best_kanji_model.keras'")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure 'best_kanji_model.keras' exists.")
    exit()

print("\nEvaluating model on Test Set...")
test_loss, test_accuracy = loaded_model.evaluate(x_test, y_test, verbose=1)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

#Confusion Matrix
y_pred_probs = loaded_model.predict(x_test)
y_pred_labels = np.argmax(y_pred_probs, axis=1)
y_true_labels = np.argmax(y_test, axis=1)

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_true_labels, y_pred_labels)
print("Confusion Matrix:\n", cm)

# plt.figure(figsize=(10, 8))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
# plt.xlabel('Predicted Label')
# plt.ylabel('True Label')
# plt.title('Confusion Matrix for Kanji Recognition')
# plt.show()

print("\nModel evaluation complete.")
