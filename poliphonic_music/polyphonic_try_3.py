import os
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau

# Funkcja do ekstrakcji cech z pliku dźwiękowego
def extract_features(file_path):
    y, sr = librosa.load(file_path)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    mfccs = librosa.feature.mfcc(y=y, sr=sr)
    return np.vstack((chroma, mfccs)).astype(float)

folder_path = "piano_triads"
audio_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(".wav")]

labels = [os.path.splitext(os.path.basename(file))[0] for file in audio_files]
numbers = np.array([i for i, _ in enumerate(labels)])

# Ekstrakcja cech z plików dźwiękowych
X = np.array([extract_features(file) for file in audio_files])

X_train, X_test, y_train, y_test = train_test_split(X, numbers, test_size=0.3, random_state=42)


model = Sequential()
model.add(LSTM(units=130, input_shape=(X_train[0].shape), activation='tanh', recurrent_activation='sigmoid'))
model.add(Dense(340, activation='relu'))
model.add(Dropout(0.5))  # Dodanie warstwy Dropout w celu regularyzacji
model.add(Dense(len(labels), activation='softmax'))

optimizer = Adam(learning_rate=0.001, amsgrad=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=80, validation_data=(X_test, y_test), callbacks=[reduce_lr])
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))

# Plot training & validation accuracy values
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')

# Show the plots
plt.tight_layout()
plt.show()

example = np.array([extract_features("E:\Amron\music_processing\project_audio_to_midi\Audio_to_midi\piano_triads\Cs_min_3_0.wav")])
prediction = model.predict(example)
odp = np.argmax(prediction)

print(labels[odp])
print(model.summary())
