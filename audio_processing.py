import librosa
import librosa.feature
from pathlib import Path
import numpy as np
import pandas as pd
SAMPLING_RATE = 22050
FRAME_LENGTH = 2048
HOP_LENGTH = 1024
EMOTION = ['anger', 'disgust', 'fear', 'happiness']

class Preprocessing:
    def __init__(self, sr=SAMPLING_RATE, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH, emotions=EMOTION):
        self.sr = sr
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.emotions = emotions

    def features(self, sample):
        zcr = librosa.feature.zero_crossing_rate(y=sample, frame_length=self.frame_length,
                                                 hop_length=self.hop_length)

        rms = librosa.feature.rms(y=sample, frame_length=self.frame_length,
                                  hop_length=self.hop_length)

        rolloff_30 = librosa.feature.spectral_rolloff(y=sample, n_fft=self.frame_length,
                                                      hop_length=self.hop_length, roll_percent=0.3)

        rolloff_50 = librosa.feature.spectral_rolloff(y=sample, n_fft=self.frame_length,
                                                      hop_length=self.hop_length, roll_percent=0.5)

        rolloff_70 = librosa.feature.spectral_rolloff(y=sample, n_fft=self.frame_length,
                                                      hop_length=self.hop_length, roll_percent=0.7)

        rolloff_85 = librosa.feature.spectral_rolloff(y=sample, n_fft=self.frame_length,
                                                      hop_length=self.hop_length, roll_percent=0.85)

        mfcc = librosa.feature.mfcc(y=sample, n_fft=self.frame_length,
                                    hop_length=self.hop_length, n_mfcc=13)

        feature_vector = np.array([np.mean(zcr), np.std(zcr),
                                   np.mean(rms), np.std(rms),
                                   np.mean(rolloff_30), np.std(rolloff_30),
                                   np.mean(rolloff_50), np.std(rolloff_50),
                                   np.mean(rolloff_70), np.mean(rolloff_70),
                                   np.mean(rolloff_85), np.mean(rolloff_85),
                                   *np.mean(mfcc, axis=1), *np.std(mfcc, axis=1)])
        return feature_vector

    def form_df(self):
        result = np.array([])
        for emotion in self.emotions:
            path = Path(f'dataverse_files/{emotion}')
            for files in path.glob('*.wav'):
                y, _ = librosa.load(files)
                sample = np.append(self.features(sample=y), emotion)
                if (result.size == 0):
                    result = sample
                else:
                    result = np.vstack((result, sample))
        df = pd.DataFrame(result, columns=['zcr_mean', 'zcr_std', 'rms_mean', 'rms_std', 'rf30_mean',
                                               'rf30_std', 'rf50_mean', 'rf50_std', 'rf70_mean',
                                               'rf70_std', 'rf85_mean', 'rf85_std', 'mfcc1_mean',
                                               'mfcc2_mean', 'mfcc3_mean', 'mfcc4_mean', 'mfcc5_mean',
                                               'mfcc6_mean', 'mfcc7_mean', 'mfcc8_mean', 'mfcc9_mean',
                                               'mfcc10_mean', 'mfcc11_mean',
                                               'mfcc12_mean', 'mfcc13_mean',
                                               'mfcc1_std', 'mfcc2_std', 'mfcc3_std', 'mfcc4_std',
                                               'mfcc5_std', 'mfcc6_std', 'mfcc7_std', 'mfcc8_std',
                                               'mfcc9_std', 'mfcc10_std', 'mfcc11_std', 'mfcc12_std',
                                               'mfcc13_std', 'sentiment'
                                               ])

        df.to_csv('haha.csv')



pre = Preprocessing()
pre.form_df()


