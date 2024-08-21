from keras.models import load_model
import librosa
from audio_processing import Preprocessing, EMOTION
import numpy as np
model = load_model('cnn.h5')
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
def predict(path):
    sample, _ = librosa.load(f'{path}.wav')
    pre = Preprocessing()
    feature = pre.features(sample)
    # this is 1D (38,) lol
    feature = feature.reshape(1, 38)
    prediction = model.predict(feature)
    print(EMOTION[np.argmax(prediction)])
while True:
    input_ = input('input the audio.wav you wanna know the emotion')
    if(input_ == ' '):
        break
    else:
        predict(input_)
