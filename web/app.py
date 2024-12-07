from flask import Flask, request, render_template
import tensorflow as tf
import librosa
import numpy as np

# Load the trained model
model = tf.keras.models.load_model("C:/DeepBeat/models/music_genre_ann.h5")

# Define the genres (ensure these match your label encoding)
GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop',
          'jazz', 'metal', 'pop', 'reggae', 'rock']

# Feature extraction function
import librosa
import numpy as np

def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=30, sr=22050)

    # Extract 13 MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs.T, axis=0)

    # Extract Chroma features
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma.T, axis=0)

    # Extract Spectral Contrast features
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    spectral_contrast_mean = np.mean(spectral_contrast.T, axis=0)

    # Combine features into a single array
    combined_features = np.hstack((mfccs_mean, chroma_mean, spectral_contrast_mean))

    # Ensure the resulting feature vector has exactly 57 elements
    if combined_features.shape[0] != 57:
        raise ValueError(f"Feature vector shape mismatch: Expected 57, got {combined_features.shape[0]}")

    return combined_features



# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files or request.files['file'].filename == '':
        return "No file uploaded. Please upload a valid .wav file.", 400

    file = request.files['file']
    if file:
        try:
            # Save the uploaded file locally
            file_path = "uploaded_audio.wav"
            file.save(file_path)
            
            # Extract features
            features = extract_features(file_path)
            print(f"Extracted features shape: {features.shape}")  # Debugging

            # Ensure features match model's input shape
            features = np.expand_dims(features, axis=0)
            print(f"Input to model shape: {features.shape}")  # Debugging
            
            # Predict the genre
            prediction = model.predict(features)
            predicted_genre = GENRES[np.argmax(prediction)]

            return f"Predicted Genre: {predicted_genre}"
        except Exception as e:
            return f"Error processing file: {str(e)}", 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
