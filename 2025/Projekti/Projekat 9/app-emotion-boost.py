import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, res_type='kaiser_fast')
        
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        
        mel = librosa.feature.melspectrogram(y=y, sr=sr)
        mel_mean = np.mean(mel, axis=1)
        
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        contrast_mean = np.mean(contrast, axis=1)

        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
        tonnetz_mean = np.mean(tonnetz, axis=1)

        features = np.hstack((mfcc_mean, chroma_mean, mel_mean, contrast_mean, tonnetz_mean))
        return features
    except Exception as e:
        print(f"Error while loading {file_path}: {e}")
        return None

def emotion_from_filename(filename):
    emotion_code = int(filename.split("-")[2])
    emotions = {
        1: "neutral",
        2: "calm",
        3: "happy",
        4: "sad",
        5: "angry",
        6: "fearful",
        7: "disgust",
        8: "surprised"
    }
    return emotions.get(emotion_code)

def load_data(base_dir, max_actors=5):
    X, y = [], []
    actors = sorted([folder for folder in os.listdir(base_dir) if folder.startswith("Actor_")])
    
    for actor in actors:
        actor_dir = os.path.join(base_dir, actor)
        for file in os.listdir(actor_dir):
            if file.endswith(".wav"):
                file_path = os.path.join(actor_dir, file)
                label = emotion_from_filename(file)
                features = extract_features(file_path)
                if features is not None:
                    X.append(features)
                    y.append(label)
    
    return np.array(X), np.array(y)

if __name__ == "__main__":
    base_dir = r"C:\Users\ZIRA\Desktop\MASTER\RP\soundFiles"

    X, y = load_data(base_dir, max_actors=5)
    print(f"Loaded: {len(X)} samples.\n")

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\n Report")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
