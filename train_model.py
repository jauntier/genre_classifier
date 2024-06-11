import librosa
import numpy as np
import os
import pandas as pd
import soundfile as sf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle

def is_valid_audio(file_name):
    try:
        with sf.SoundFile(file_name) as f:
            return True
    except Exception as e:
        print(f"Invalid audio file: {file_name}. Exception: {e}")
        return False

def extract_features(file_name):
    try:
        audio, sample_rate = sf.read(file_name)
        audio = audio.astype(np.float32)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        return mfccs_scaled
    except Exception as e:
        print(f"Error encountered while parsing file: {file_name}. Exception: {e}")
        return None

dataset_path = './Data/genres_original'

if not os.path.exists(dataset_path):
    print(f"Dataset path '{dataset_path}' does not exist. Please check the path and try again.")
else:
    genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
    features_list = []

    for genre in genres:
        genre_path = os.path.join(dataset_path, genre)
        for file_name in os.listdir(genre_path):
            file_path = os.path.join(genre_path, file_name)
            if is_valid_audio(file_path):
                data = extract_features(file_path)
                if data is not None:
                    features_list.append([data, genre])
            else:
                print(f"Skipping missing file: {file_path}")

    features_df = pd.DataFrame(features_list, columns=['feature', 'label'])
    features_df.dropna(inplace=True)

    X = np.array(features_df['feature'].tolist())
    y = np.array(features_df['label'].tolist())

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    with open('label_encoder.pkl', 'wb') as le_file:
        pickle.dump(le, le_file)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    model = SVC(kernel='linear', probability=True)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy * 100:.2f}%")

    with open('genre_classifier_model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)