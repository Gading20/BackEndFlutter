import os
import librosa
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Fungsi untuk ekstraksi fitur
def extract_features(file_path, mfcc=True, chroma=True, mel=True):
    with tf.device('/cpu:0'):
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        features = []
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)
            features.append(mfccs)
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sample_rate).T, axis=0)
            features.append(chroma)
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sample_rate).T, axis=0)
            features.append(mel)
    return np.concatenate(features)

# Direktori dataset
data_Idgham_Bighunnah = "Dataset/Idgham_Bighunnah"
data_Ikhfa_Haqiqi = "Dataset/Ikhfa_Haqiqi"
data_Izhar_Halqi = "Dataset/Izhar_Halqi"

features = []
labels = []

# Proses ekstraksi fitur dan pelabelan
for file in os.listdir(data_Idgham_Bighunnah):
    file_path = os.path.join(data_Idgham_Bighunnah, file)
    feature = extract_features(file_path)
    features.append(feature)
    labels.append(0)

for file in os.listdir(data_Ikhfa_Haqiqi):
    file_path = os.path.join(data_Ikhfa_Haqiqi, file)
    feature = extract_features(file_path)
    features.append(feature)
    labels.append(1)

for file in os.listdir(data_Izhar_Halqi):
    file_path = os.path.join(data_Izhar_Halqi, file)
    feature = extract_features(file_path)
    features.append(feature)
    labels.append(2)

features = np.array(features)
labels = np.array(labels)

# Pembagian dataset menjadi training dan validation
X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)

# Pembangunan model
inputs = tf.keras.layers.Input(shape=(X_train.shape[1],))
reshape = tf.keras.layers.Reshape((X_train.shape[1], 1))(inputs)
conv1 = tf.keras.layers.Conv1D(64, 3, activation='relu')(reshape)
pool1 = tf.keras.layers.MaxPooling1D(2)(conv1)
conv2 = tf.keras.layers.Conv1D(128, 3, activation='relu')(pool1)
pool2 = tf.keras.layers.MaxPooling1D(2)(conv2)
conv3 = tf.keras.layers.Conv1D(256, 3, activation='relu')(pool2)
global_pool = tf.keras.layers.GlobalAveragePooling1D()(conv3)
dense1 = tf.keras.layers.Dense(64, activation='relu')(global_pool)
dropout = tf.keras.layers.Dropout(0.5)(dense1)
outputs = tf.keras.layers.Dense(3, activation='softmax')(dropout)

model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

# Kompilasi model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Pelatihan model
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=32)

# Evaluasi model
loss, accuracy = model.evaluate(X_val, y_val)
print("Validation Accuracy:", accuracy)

model.summary()

# Simpan model ke format H5
model.save('Model/model.h5')
print("Model saved to model.h5")
