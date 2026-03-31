import streamlit as st
import numpy as np
import librosa
import tensorflow as tf

# Load model (SavedModel format)
model = tf.saved_model.load("saved_model")
infer = model.signatures["serving_default"]

# Genres (IMPORTANT: same order as training)
genres = ['blues', 'classical', 'country', 'disco', 'hiphop',
          'jazz', 'metal', 'pop', 'reggae', 'rock']

# Feature extraction (same as training)
def extract_features(file):
    y, sr = librosa.load(file, duration=30, res_type='kaiser_fast')
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)

# UI
st.set_page_config(page_title="🎵 Music Genre Classifier", layout="centered")

st.title("🎵 Music Genre Classification")
st.write("Upload a music file (.wav) and get predicted genre")

uploaded_file = st.file_uploader("Upload Audio File", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file)

    st.write("🔍 Extracting features...")

    features = extract_features(uploaded_file)

    if features is not None:
        features = np.expand_dims(features, axis=0).astype(np.float32)

        st.write("🤖 Predicting...")

        # Predict using SavedModel
        prediction = infer(tf.constant(features))

        # Get output tensor
        output = list(prediction.values())[0].numpy()

        predicted_index = np.argmax(output)
        predicted_genre = genres[predicted_index]
        confidence = np.max(output)

        st.success(f"🎶 Predicted Genre: **{predicted_genre.upper()}**")
        st.info(f"Confidence: {confidence:.2f}")

    else:
        st.error("❌ Could not process audio file")