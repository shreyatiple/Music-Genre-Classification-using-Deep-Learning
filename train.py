import numpy as np
import librosa
import os
import warnings
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

warnings.filterwarnings("ignore")

DATASET_PATH = "Data/genres_original"
genres = os.listdir(DATASET_PATH)

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, duration=30, res_type='kaiser_fast')
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        return np.mean(mfcc.T, axis=0)
    except:
        return None

X, y = [], []

print("📂 Processing dataset...")

for i, genre in enumerate(genres):
    genre_path = os.path.join(DATASET_PATH, genre)
    
    for file in os.listdir(genre_path):
        file_path = os.path.join(genre_path, file)
        features = extract_features(file_path)
        
        if features is not None:
            X.append(features)
            y.append(i)

X = np.array(X)
y = np.array(y)

if len(X) == 0:
    print("❌ No data loaded")
    exit()

print(f"✅ Total samples: {len(X)}")

y = keras.utils.to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = keras.Sequential([
    layers.Reshape((40, 1), input_shape=(40,)),
    layers.Conv1D(64, 3, activation='relu'),
    layers.MaxPooling1D(2),
    layers.Conv1D(128, 3, activation='relu'),
    layers.MaxPooling1D(2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(len(genres), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("🚀 Training model...")

model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=32,
    validation_data=(X_test, y_test)
)

print("💾 Saving model...")

tf.saved_model.save(model, "saved_model")

print("✅ Model saved in folder: saved_model")