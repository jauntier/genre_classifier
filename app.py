from flask import Flask, request, render_template
import numpy as np
import librosa
import pickle

# Load the trained model and label encoder
with open('genre_classifier_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('label_encoder.pkl', 'rb') as le_file:
    le = pickle.load(le_file)

app = Flask(__name__)

def extract_features_for_prediction(file_name, target_sr=22050):
    try:
        audio, sample_rate = librosa.load(file_name, sr=target_sr)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        return mfccs_scaled
    except Exception as e:
        print(f"Error encountered while parsing file: {file_name}. Exception: {e}")
        return None

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error="No file part")
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error="No selected file")
        if file:
            file_path = f"./{file.filename}"
            file.save(file_path)
            features = extract_features_for_prediction(file_path)
            if features is not None:
                features = np.expand_dims(features, axis=0)
                prediction = model.predict(features)
                predicted_genre = le.inverse_transform(prediction)
                return render_template('index.html', prediction=f"Predicted genre: {predicted_genre[0]}")
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
