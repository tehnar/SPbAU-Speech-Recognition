import librosa
import numpy as np
import pandas as pd
from scipy.io import wavfile as wav


class FeatureExtractor:
    def extract_features_from_file(self, wav_path):
        """
        Extracts features for classification ny frames for .wav file

        :param wav_path: string, path to .wav file
        :return: pandas.DataFrame with features of shape (n_chunks, n_features)
        """
        raise NotImplementedError("Should have implemented this")


class LibrosaExtractor(FeatureExtractor):
    def __init__(self, frame_sec=0.5):
        self.frame_sec = frame_sec

    def extract_features_from_file(self, wav_path):
        audio, rate = librosa.load(wav_path)
        return self.extract_features(rate, audio)

    def extract_features(self, rate, audio):
        audio = audio.astype(np.float64)
        length = audio.shape[0]
        frame_length = int(rate * self.frame_sec)
        frame_shift = frame_length // 4

        features = []
        for i in range(0, length - frame_length + 1, frame_shift):
            features.append(np.concatenate((
                np.mean(librosa.feature.mfcc(audio[i: i + frame_length], rate).T, axis=0),
                np.mean(np.log(librosa.feature.melspectrogram(audio[i: i + frame_length], rate).T), axis=0)
            )))

        return pd.DataFrame(np.array(features))
