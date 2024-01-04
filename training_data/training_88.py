from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


folder_path = 'project_audio_to_midi/Audio_to_midi/notes_88'
training_data = []
training_labels = []

for folder_name in os.listdir(folder_path):
    folder_dir = os.path.join(folder_path, folder_name)
    if not os.path.isdir(folder_dir):
        continue
    note_label = folder_name
    for filename in os.listdir(folder_dir):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_dir, filename)
            with open(file_path, 'r') as file:
                reader = csv.reader(file)
                mel_spec = list(reader)
            mel_spec = np.array(mel_spec, dtype=np.float32)
            training_data.append(mel_spec)
            training_labels.append(note_label)

print(training_data[0].shape)

# Convert training_data to a NumPy array
training_data = np.array(training_data)

# Perform label encoding on training_labels
label_encoder = LabelEncoder()
training_labels = label_encoder.fit_transform(training_labels)
print(training_labels)
X_train, X_test, y_train, y_test = train_test_split(training_data, training_labels, test_size=0.3, random_state=42)

model = Sequential()

model.add(Flatten(input_shape=training_data[0].shape))
model.add(Dense(640, activation='relu'))
model.add(Dense(176, activation='relu'))
model.add(Dense(len(label_encoder.classes_), activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=40, validation_data=(X_test, y_test))

prediction = model.predict(np.expand_dims(X_train[0], axis=0))

predicted_label = np.argmax(prediction)

model.summary()

predicted_note = label_encoder.inverse_transform([predicted_label])[0]

print(f"Predicted Note: {predicted_note}")

model.save('project_audio_to_midi/Audio_to_midi/notes_88.h5')

