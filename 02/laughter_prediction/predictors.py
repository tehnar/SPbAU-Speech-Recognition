import numpy as np

from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
import keras
from keras.models import Input, Model
from keras.layers import LSTM, Dense
from keras.optimizers import Adam


class Predictor:
    """
    Wrapper class used for loading serialized model and
    using it in classification task.
    Defines unified interface for all inherited predictors.
    """

    def predict(self, X):
        """
        Predict target values of X given a model

        :param X: numpy.ndarray, dtype=float, shape=[n_samples, n_features]
        :return: numpy.array predicted classes
        """
        raise NotImplementedError("Should have implemented this")

    def predict_proba(self, X):
        """
        Predict probabilities of target class

        :param X: numpy.ndarray, dtype=float, shape=[n_samples, n_features]
        :return: numpy.array target class probabilities
        """
        raise NotImplementedError("Should have implemented this")


class XgboostPredictor(Predictor):
    """Parametrized wrapper for xgboost-based predictors"""

    def __init__(self, model_path, threshold, scaler=None):
        self.threshold = threshold
        self.clf = joblib.load(model_path)
        self.scaler = scaler

    def _simple_smooth(self, data, n=50):
        dlen = len(data)

        def low_pass(data, i, n):
            if i < n // 2:
                return data[:i]
            if i >= dlen - n // 2 - 1:
                return data[i:]
            return data[i - n // 2: i + n - n // 2]

        sliced = np.array([low_pass(data, i, n) for i in range(dlen)])
        sumz = np.array([np.sum(x) for x in sliced])
        return sumz / n

    def predict(self, X):
        y_pred = self.clf.predict_proba(X)
        ypreds_bin = np.where(y_pred[:, 1] >= self.threshold, np.ones(len(y_pred)), np.zeros(len(y_pred)))
        return ypreds_bin

    def predict_proba(self, X):
        X_scaled = self.scaler.fit_transform(X) if self.scaler is not None else X
        not_smooth = self.clf.predict_proba(X_scaled)[:, 1]
        return self._simple_smooth(not_smooth)


class StrictLargeXgboostPredictor(XgboostPredictor):
    """
    Predictor trained on 3kk training examples, using PyAAExtractor
    for input features
    """
    def __init__(self, threshold=0.045985743):
        XgboostPredictor.__init__(self, model_path="models/XGBClassifier_3kk_pyAA10.pkl",
                                  threshold=threshold, scaler=StandardScaler())


class RnnPredictor(Predictor):
    def __init__(self, x_len, num_mfcc, num_fbank):
        self.num_mfcc = num_mfcc
        self.num_fbank = num_fbank

        in_mfcc = Input(shape=(x_len, num_mfcc))
        in_fbank = Input(shape=(x_len, num_fbank))

        lstm_mfcc = LSTM(128, return_sequences=True)(in_mfcc)
        lstm_fbank = LSTM(128, return_sequences=True)(in_fbank)

        out_mfcc = Dense(1, activation='sigmoid')(lstm_mfcc)

        lstm = keras.layers.concatenate([lstm_mfcc, lstm_fbank], axis=2)
        out_all = Dense(1, activation='sigmoid')(lstm)

        self.model = Model(inputs=[in_mfcc, in_fbank], outputs=[out_mfcc, out_all])
        self.model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

    def load(self, filename):
        self.model = keras.models.load_model(filename)

    def fit(self, X_train, y_train, batch_size, epochs):
        y_train = y_train.reshape(y_train.shape + (1,))
        self.model.fit(
            [X_train[:, :, :self.num_mfcc], X_train[:, :, self.num_mfcc:]],
            [y_train, y_train],
            batch_size=batch_size,
            epochs=epochs,
            verbose=1
        )

    def predict(self, X, threshold=0.5):
        return (np.array(self.predict_proba(X)) > threshold).astype(int)

    def predict_proba(self, X):
        return self.model.predict([X[:, :, :self.num_mfcc], X[:, :, self.num_mfcc:]])[1].squeeze()

    def save(self, filename):
        self.model.save(filename)