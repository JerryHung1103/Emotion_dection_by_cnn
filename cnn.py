from audio_processing import Preprocessing
import keras
from keras.layers import Conv1D, Dropout, Dense, Flatten, MaxPooling1D
from keras.models import Sequential
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.callbacks import Callback, EarlyStopping


df = pd.read_csv('haha.csv')

class CustomAccuracyCallback(Callback):
    def __init__(self, target_accuracy):
        super(CustomAccuracyCallback, self).__init__()
        self.target_accuracy = target_accuracy

    def on_epoch_end(self, epoch, logs=None):
        if logs.get('accuracy') >= self.target_accuracy:
            self.model.stop_training = True

features_vectors = np.array(df.drop(['sentiment'], axis=1))
labels = np.array(df['sentiment'])

labels[labels == 'anger'] = 0
labels[labels == 'disgust'] = 1
labels[labels == 'fear'] = 2
labels[labels == 'happiness'] = 3

labels = to_categorical(labels, 4)

x_train, x_test, y_train, y_test = train_test_split(features_vectors, labels, test_size=0.2, random_state=8964)

model = keras.Sequential()
model.add(Conv1D(filters=32, kernel_size=3,  padding = 'same', activation='relu', input_shape=(39, 1)))
model.add(MaxPooling1D(pool_size=2))

model.add(Conv1D(filters=32*2, kernel_size=3, padding = 'same' , activation='relu'))
model.add(MaxPooling1D(pool_size=2))

model.add(Conv1D(filters=32*3, kernel_size=3, padding = 'same' , activation='relu'))
model.add(MaxPooling1D(pool_size=2))

model.add(Conv1D(filters=32*3, kernel_size=3, padding = 'same' , activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(keras.layers.Dense(128, activation='relu'))
# model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(4, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy', metrics=['accuracy'])
# early_stopping_callback = EarlyStopping(monitor='val_accuracy', patience=10, mode='max')
model.fit(x_train, y_train, batch_size=32, validation_data=(x_test, y_test), epochs=200)

loss, accuracy = model.evaluate(x_test, y_test)

print(f'Test loss: {loss}')
print(f'Test accuracy: {accuracy}')
if(input('save? yes or no') == 'yes'):
    model.save('cnn.h5')





