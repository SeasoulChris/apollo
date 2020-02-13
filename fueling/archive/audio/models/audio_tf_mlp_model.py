#!/usr/bin/env python
import argparse
import os
import glob
import time
import warnings
import sys

from absl import flags
import librosa
from keras import backend as K
from keras import optimizers
from keras.models import load_model
from keras.callbacks import EarlyStopping
from keras.layers import Dense
from keras.models import Sequential
import matplotlib.ticker as ticker
import matplotlib.style as ms
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import seed
import pandas as pd
from scipy.signal import butter, lfilter, hilbert
import seaborn as sns
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow import set_random_seed
from tqdm import tqdm

from fueling.audio.pyAudioAnalysis import audioFeatureExtraction


flags.DEFINE_string('root_dir', '/home/jinyun/cleaned_data/',
                    'The root dir containing data.')

sns.set_style('whitegrid')
if not sys.warnoptions:
    warnings.simplefilter('ignore')


def butter_bandpass_filter(data, lowcut=500, highcut=1500, fs=8000, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, data)
    return y


def melspectrogram(y, sr):
    # Let's make and display a mel-scaled power (energy-squared) spectrogram
    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)

    # Convert to log scale (dB). We'll use the peak power (max) as reference.
    log_S = librosa.power_to_db(S, ref=np.max)

    return log_S


def preprocess(y):
    y_filt = butter_bandpass_filter(y)
    analytic_signal = hilbert(y_filt)
    amplitude_envelope = np.abs(analytic_signal)
    return amplitude_envelope


def prepare_data(X_em, X_nonem, scale=True):
    X_em = np.array(X_em)
    X_nonem = np.array(X_nonem)

    X = np.vstack((X_em, X_nonem))
    Y = np.hstack((np.ones(len(X_em)), np.zeros(len(X_nonem))))

    scaler = StandardScaler()
    if scale:
        scaler.fit_transform(X)

    X, Y = shuffle(X, Y, random_state=7)

    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    return X, Y, scaler


def plot_model_history(model_history):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    # summarize history for accuracy
    axs[0].plot(range(1, len(model_history.history['acc'])+1),
                model_history.history['acc'])
    axs[0].plot(range(1, len(model_history.history['val_acc'])+1),
                model_history.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(
        1, len(model_history.history['acc'])+1), len(model_history.history['acc'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1, len(model_history.history['loss'])+1),
                model_history.history['loss'])
    axs[1].plot(range(1, len(model_history.history['val_loss'])+1),
                model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(
        1, len(model_history.history['loss'])+1), len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    plt.savefig('model_history.png')


def predict_op(y, sr,  scaler, model):
    y = preprocess(y)
    features_list = audioFeatureExtraction.stFeatureExtraction(
        y, sr, 0.10*sr, .05*sr)
    scaler.transform(features_list)
    count = 0
    N = 10
    th = 0.5

    prob_list = []
    class_list = []
    for i in range(N):
        p = model.predict(features_list[i].reshape(
            1, 105), batch_size=None, verbose=0)
        p = p.flatten()
        prob_list.append(p)
    prob = np.mean(prob_list)
    # print(prob)
    if prob > th:
        # print('Em')
        class_list.append(1)
    else:
        # print('Non-em')
        class_list.append(0)

    for i in range(N, len(features_list)):
        prob_list.pop(0)
        p = model.predict(features_list[i].reshape(
            1, 105), batch_size=None, verbose=0)
        p = p.flatten()
        prob_list.append(p)
        prob = np.mean(prob_list)
        # print(prob)
        if prob > th:
            # print('Em')
            class_list.append(1)
        else:
            # print('Non-em')
            class_list.append(0)
    if np.mean(class_list) > 0.5:
        return 1
    else:
        return 0


def predict_prob(y, scaler):
    y = preprocess(y)
    mfccs_list = audioFeatureExtraction.stFeatureExtraction(
        y, sr, 0.10*sr, .05*sr)
    scaler.transform(mfccs_list)
    count = 0
    N = 20
    th = 0.5

    model = load_model('model_3h.h5')

    prob_list = []
    class_list = []
    for i in range(N):
        p = model.predict(mfccs_list[i].reshape(
            1, 105), batch_size=None, verbose=0)
        p = p.flatten()
        prob_list.append(p)
    prob = np.mean(prob_list)
    # print(prob)
    if prob > th:
        # print('Em')
        class_list.append(1)
    else:
        # print('Non-em')
        class_list.append(0)

    for i in range(N, len(mfccs_list)):
        prob_list.pop(0)
        p = model.predict(mfccs_list[i].reshape(
            1, 105), batch_size=None, verbose=0)
        p = p.flatten()
        prob_list.append(p)
        prob = np.mean(prob_list)
        # print(prob)
        if prob > th:
            # print('Em')
            class_list.append(1)
        else:
            # print('Non-em')
            class_list.append(0)
    return class_list


if __name__ == '__main__':

    def main(argv):

        flags_dict = flags.FLAGS.flag_values_dict()
        root_dir = flags_dict['root_dir']

        train_path_em = os.path.join(
            root_dir, 'cleaned_data/train_balanced/Emergency/')
        train_path_nonem = os.path.join(
            root_dir, 'cleaned_data/train_balanced/nonEmergency/')
        test_path_em = os.path.join(
            root_dir, 'cleaned_data/eval_balanced/Emergency/')
        test_path_nonem = os.path.join(
            root_dir, 'cleaned_data/eval_balanced/nonEmergency/')

        train_em_files = glob.glob(os.path.join(train_path_em, '*.wav'))
        train_nonem_files = glob.glob(os.path.join(train_path_nonem, '*.wav'))

        print('Generating X_em')
        X_em = []
        for fn in tqdm(train_em_files):
            try:
                y, sr = librosa.load(fn, sr=8000)
            except:
                print('Failed to open file {}'.format(fn))
                continue
            y = preprocess(y)
            features = audioFeatureExtraction.stFeatureExtraction(
                y, sr, 0.10*sr, .05*sr)
            X_em.extend(features)

        print('Number of em data is {}'.format(len(X_em)))
        print('EM data dimension = {}'.format(len(X_em[0])))

        print('Generating X_nonem')
        count = 0
        X_nonem = []
        for fn in tqdm(train_nonem_files):
            try:
                y, sr = librosa.load(fn, sr=8000)
            except:
                print('Failed to open file {}'.format(fn))
                continue
            y = preprocess(y)
            features = audioFeatureExtraction.stFeatureExtraction(
                y, sr, 0.10*sr, .05*sr)
            X_nonem.extend(features)
            count += 1
            if count == 450:
                break

        print('Number of nonem data is {}'.format(len(X_nonem)))

        X_train, Y_train, scaler1 = prepare_data(X_em, X_nonem)

        f = open('scaler_values.txt', 'w+')
        mean_v = []
        std_v = []
        for i in range(len(scaler1.mean_)):
            mean_v.append(scaler1.mean_[i])
            std_v.append(scaler1.scale_[i])
        f.write('scaler mean:\n{}'.format(mean_v))
        f.write('\nscaler std:\n{}'.format(std_v))
        f.close()

        test_em_files = glob.glob(os.path.join(test_path_em, '*.wav'))
        test_nonem_files = glob.glob(os.path.join(test_path_nonem, '*.wav'))

        print('Generating X_test_em')
        X_test_em = []
        for fn in tqdm(test_em_files):
            try:
                y, sr = librosa.load(fn, sr=8000)
            except:
                print('Failed to open file {}'.format(fn))
                continue
            y = preprocess(y)
            features = audioFeatureExtraction.stFeatureExtraction(
                y, sr, 0.10*sr, .05*sr)
            X_test_em.extend(features)

        print('Generating X_test_nonem')
        X_test_nonem = []
        for fn in tqdm(test_nonem_files):
            try:
                y, sr = librosa.load(fn, sr=8000)
            except:
                print('Failed to open file {}'.format(fn))
                continue
            y = preprocess(y)
            features = audioFeatureExtraction.stFeatureExtraction(
                y, sr, 0.10*sr, .05*sr)
            X_test_nonem.extend(features)

        X_test, Y_test, _ = prepare_data(X_test_em, X_test_nonem, False)

        K.set_image_dim_ordering('th')

        # fix random seed for reproducibility
        np.random.seed(7)

        # Supress Tensorflow error logs
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        seed(1)
        set_random_seed(2)
        model = Sequential()
        model.add(Dense(105, input_dim=105, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(4, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        optm = optimizers.Adam(lr=0.005, beta_1=0.9, beta_2=0.999,
                               epsilon=None, decay=0.0, amsgrad=False)
        model.compile(loss='binary_crossentropy',
                      optimizer=optm, metrics=['accuracy'])
        earlystop = EarlyStopping(
            monitor='val_loss', min_delta=0.01, patience=20, verbose=0, mode='auto')
        callbacks_list = [earlystop]

        history = model.fit(X_train, Y_train, epochs=200, validation_data=(
            X_test, Y_test), batch_size=256, callbacks=callbacks_list)
        model.save('model_3h.h5')
        print('Saved model to disk!')

        plot_model_history(history)

        model = load_model('model_3h.h5')

        Y_pred = model.predict_classes(X_test)
        cm = confusion_matrix(Y_pred, Y_test)
        df_cm = pd.DataFrame(cm, index=['Non-EM', 'EM'],
                             columns=['Non-EM', 'EM'])
        plt.figure(figsize=(8, 6))
        sns.heatmap(df_cm, annot=True, cmap='YlGnBu')

        pred_test = np.array(model.predict_classes(X_test))
        pred_test = pred_test[:, 0]
        pred_test

        Y_test_np = np.array(Y_test)
        Y_test_em = Y_test_np[Y_test_np == 1]
        Y_test_nonem = Y_test_np[Y_test_np == 0]

        print('------ data level results -----')
        is_correct = pred_test == Y_test_np
        print('Overall ACC = {}'.format(np.mean(is_correct)))

        em_correct = is_correct[Y_test_np == 1]
        print('EM ACC = {}'.format(np.mean(em_correct)))

        nonem_correct = is_correct[Y_test_np == 0]
        print('Non-EM ACC = {}'.format(np.mean(nonem_correct)))

        em_tot = 0
        correct_em = 0
        op_list = []
        for test_file in tqdm(test_em_files):
            try:
                y, sr = librosa.load(fn, sr=8000)
            except:
                print('Failed to open file {}'.format(fn))
                continue
            classes = predict_op(y, sr, scaler1, model)
            if classes == 1:
                correct_em += 1
            em_tot += 1

        print('------ File level results ------')
        print('Correct EM count = {}'.format(correct_em))
        print('Total EM count = {}'.format(em_tot))

        nonem_tot = 0
        correct_nonem = 0
        op_list = []
        for test_file in tqdm(test_nonem_files):
            try:
                y, sr = librosa.load(fn, sr=8000)
            except:
                print('Failed to open file {}'.format(fn))
                continue
            classes = predict_op(y, sr, scaler1, model)
            if classes == 0:
                correct_nonem += 1
            nonem_tot += 1

        print('Correct Non-EM count = {}'.format(correct_nonem))
        print('Total Non-EM count = {}'.format(nonem_tot))

    from absl import app
    app.run(main)
