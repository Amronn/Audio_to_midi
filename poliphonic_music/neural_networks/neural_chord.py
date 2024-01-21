from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, SimpleRNN, LSTM, Dropout, Conv1D
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.optimizers import Adam

cqt_path = 'poliphonic_music/chords_cqt_1000'
labels_path = 'poliphonic_music/chord_data.csv'

times_read = []
chords_read = []

with open(labels_path, mode='r') as file:
    reader = csv.reader(file)
    header = next(reader)
    for row in reader:
        times_read.append(int(row[0]))
        chords_read.append([int(note) for note in row[1:]])
        
labels = []

for chord in chords_read:
    a = np.zeros(109)
    a[chord] = 1
    a = a[21:]
    labels.append(a)

labels = np.array(labels)

num_of_notes = [len(chord) for chord in chords_read]
cqt = []
for filename in os.listdir(cqt_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(cqt_path, filename)
        with open(file_path, 'r') as file:
            reader = csv.reader(file)
            chromas = list(reader)
            chromas = np.array(chromas).astype(float).T
        cqt.append(chromas)
cqt = np.array(cqt)
print(cqt[0])
print(cqt.shape)
num_of_notes = np.array(num_of_notes)

X_train, X_test, y_train, y_test = train_test_split(cqt, labels, test_size=0.3, random_state=15)

model = Sequential()

model.add(Dense(128, activation='relu',input_shape=(32,88)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(88, activation='sigmoid'))

optimizer = Adam(learning_rate=0.01)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=80, batch_size=256, validation_data=(X_test, y_test))

prediction = model.predict(np.array([X_test[3]]))

# print(y_test[2])

model.save('models/num_cqt.h5')

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.show()

# print(prediction[0])
tab = []
print(y_test[3])
for i in range(88):
    if y_test[3][i]==1:
        tab.append(i)
plt.plot(prediction[0])
plt.vlines(tab, ymin=0, ymax=1, color='red')
plt.show()