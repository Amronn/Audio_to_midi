from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, SimpleRNN
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


folder_path = 'Audio_to_midi/notes_88_mels'
x_data = []
y_data = []

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
                mels = list(reader)
            x_data.append(np.array(mels).T)
            y_data.append(int(note_label))

x_data = np.array(x_data).astype(float)
y_data = np.array(y_data)
# print(training_labels)
X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=42)

print(y_train[2])
print(X_train[0].shape)

model = Sequential()

model.add(SimpleRNN(units = 85, input_shape=(1,128)))
print(model.output_shape)
model.add(Dense(170, activation='relu'))
print(model.output_shape)
model.add(Dense(85, activation='softmax'))
print(model.output_shape)

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test))

prediction = model.predict(np.array([X_train[2]]))

# print(prediction)

model.save('Audio_to_midi/notes_88_mels.h5')

print(np.argmax(prediction))
print(np.max(prediction))