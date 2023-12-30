from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, SimpleRNN, LSTM, Dropout
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


folder_path = 'notes_88_cqt'
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
                chromas = list(reader)
            x_data.append(np.array(chromas).T)
            y_data.append(int(note_label)//12)
            # print(note_label)

x_data = np.array(x_data).astype(float)
y_data = np.array(y_data)

X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=42)

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

model.add(SimpleRNN(units = 64, input_shape=(1,85)))

model.add(Dense(128, activation='relu'))

model.add(Dense(8, activation='softmax'))


from keras.optimizers import Adam
optimizer = Adam(learning_rate=0.001, amsgrad=True)

from keras.callbacks import ReduceLROnPlateau
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=40, validation_data=(X_test, y_test), callbacks=[reduce_lr])

prediction = model.predict(np.array([X_train[2]]))
# print(prediction)
print(y_train[2])

model.save('octaves_cqt.h5')

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
print(np.argmax(prediction))
print(np.max(prediction))