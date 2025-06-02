import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = "C:/Users/ZIRA/Desktop/MASTER/RP/soundFiles" # Change with your path (Actor1,Actor2,Actor3 folders must be inside here)

emotion_dict = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

# Extract features from file (mfcc)
def extract_features(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        return mfccs_scaled
    except Exception as e:
        print(f"Error while reading {file_path}: {e}")
        return None

features = []
labels = []

# Loads every file, gets features out of it, adds data to 2 lists - feature (mfccs numbers, label - target label
for root, _, files in os.walk(DATA_PATH):
    for file in files:
        if file.endswith(".wav"):
            path = os.path.join(root, file)
            mfccs = extract_features(path)
            if mfccs is not None:
                emotion_code = file.split("-")[2]
                emotion = emotion_dict.get(emotion_code)
                if emotion is not None:
                    features.append(mfccs)
                    labels.append(emotion)

print(f"Loaded: {len(features)} samples.\n")

X = np.array(features)
y = np.array(labels)

# Split data to test/train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)


print("Report")
print(classification_report(y_test, y_pred))

cm_labels = sorted(set(y))  
cm = confusion_matrix(y_test, y_pred, labels=cm_labels)

plt.figure(figsize=(10, 6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=cm_labels, yticklabels=cm_labels, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()
