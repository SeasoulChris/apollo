#!/usr/bin/env python
# coding: utf-8

# In[1]:

import warnings
import sys
from keras import backend as K
import tensorflow as tf
from keras import optimizers
from keras.models import load_model
from keras.callbacks import EarlyStopping
from keras.layers import Dense
from keras.models import Sequential
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pyAudioAnalysis import audioFeatureExtraction
from pyAudioAnalysis import audioBasicIO
import seaborn as sns
import matplotlib.ticker as ticker
import matplotlib.style as ms
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, hilbert
from tensorflow import set_random_seed
import argparse
import os
import glob
import time

import numpy as np
import pandas as pd
# IPython.display for audio output
import IPython.display as ipd
# Librosa for audio
import librosa
import librosa.display
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib

from numpy.random import seed

sns.set_style("whitegrid")

if not sys.warnoptions:
    warnings.simplefilter("ignore")


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


def predict_op(y, scaler):
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
        # print("Em")
        class_list.append(1)
    else:
        # print("Non-em")
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
            # print("Em")
            class_list.append(1)
        else:
            # print("Non-em")
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
        # print("Em")
        class_list.append(1)
    else:
        # print("Non-em")
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
            # print("Em")
            class_list.append(1)
        else:
            # print("Non-em")
            class_list.append(0)
    return class_list


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_dir", help="parent dir to cleaned_data/ on your local machine",
        type=str)
    args = parser.parse_args()
    root_dir = args.root_dir

    train_path_em = os.path.join(
        root_dir, 'cleaned_data/train_balanced/Emergency/')
    train_path_nonem = os.path.join(
        root_dir, 'cleaned_data/train_balanced/nonEmergency/')
    test_path_em = os.path.join(
        root_dir, 'cleaned_data/eval_balanced/Emergency/')
    test_path_nonem = os.path.join(
        root_dir, 'cleaned_data/eval_balanced/nonEmergency/')

    fn = os.path.join(train_path_em, '101.wav')
    y, sr = librosa.load(fn, sr=8000)
    ipd.Audio(fn)

    librosa.display.waveplot(y=y, sr=sr)

    print(y.shape)

    y_filt = butter_bandpass_filter(y)
    print(y_filt.shape)

    librosa.display.waveplot(y=y_filt, sr=sr)

    log_S = melspectrogram(y, sr)
    # Make a new figure
    plt.figure(figsize=(12, 4))
    # Display the spectrogram on a mel scale
    # sample rate and hop length parameters are used to render the time axis
    librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')

    log_S = melspectrogram(y_filt, sr)
    # Make a new figure
    plt.figure(figsize=(12, 4))
    # Display the spectrogram on a mel scale
    # sample rate and hop length parameters are used to render the time axis
    librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')

    print(log_S.shape)

    analytic_signal = hilbert(y_filt)
    amplitude_envelope = np.abs(analytic_signal)

    t = np.arange(len(y[:8000])) / sr
    fig = plt.figure(figsize=(16, 5))
    ax0 = fig.add_subplot(111)
    # ax0.plot(y[:8000], label='signal')
    ax0.plot(analytic_signal[:8000], label='signal')
    ax0.plot(amplitude_envelope[:8000], label='envelope')
    ax0.legend()

    em_files = glob.glob(os.path.join(train_path_em, '*.wav'))
    nonem_files = glob.glob(os.path.join(train_path_nonem, '*.wav'))

    print("Generating X_em")
    X_em = []
    for fn in tqdm(em_files):
        y, sr = librosa.load(fn, sr=8000)
        y = preprocess(y)
        features = audioFeatureExtraction.stFeatureExtraction(
            y, sr, 0.10*sr, .05*sr)
        X_em.extend(features)

    print("Number of em data is {}".format(len(X_em)))
    print("EM data dimension = {}".format(len(X_em[0])))

    print("Generating X_nonem")
    count = 0
    X_nonem = []
    for fn in tqdm(nonem_files):
        y, sr = librosa.load(fn, sr=8000)
        y = preprocess(y)
        features = audioFeatureExtraction.stFeatureExtraction(
            y, sr, 0.10*sr, .05*sr)
        X_nonem.extend(features)
        count += 1
        if count == 120:
            break

    print("Number of nonem data is {}".format(len(X_nonem)))

    X_train, Y_train, scaler1 = prepare_data(X_em, X_nonem)

    f = open("scaler_values.txt", "w+")
    mean_v = []
    std_v = []
    for i in range(len(scaler1.mean_)):
        mean_v.append(scaler1.mean_[i])
        std_v.append(scaler1.scale_[i])
    f.write("scaler mean:\n{}".format(mean_v))
    f.write("\nscaler std:\n{}".format(std_v))
    f.close()

    test_em_files = glob.glob(os.path.join(test_path_em, '*.wav'))
    test_nonem_files = glob.glob(os.path.join(test_path_nonem, '*.wav'))

    print("Generating X_test_em")
    X_test_em = []
    for fn in tqdm(test_em_files):
        y, sr = librosa.load(fn, sr=8000)
        y = preprocess(y)
        features = audioFeatureExtraction.stFeatureExtraction(
            y, sr, 0.10*sr, .05*sr)
        X_test_em.extend(features)

    print("Generating X_test_nonem")
    X_test_nonem = []
    for fn in tqdm(test_nonem_files):
        y, sr = librosa.load(fn, sr=8000)
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
    model.add(Dense(16, activation='relu'))
    # model.add(Dense(128, activation='relu'))
    # model.add(Dense(256, activation='relu'))
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
    model.save("model_3h.h5")
    print("Saved model to disk!")

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

    print("------ data level results -----")
    is_correct = pred_test == Y_test_np
    print("Overall ACC = {}".format(np.sum(is_correct) / is_correct.shape[0]))

    em_correct = is_correct[Y_test_np == 1]
    print("EM ACC = {}".format(np.sum(em_correct) / em_correct.shape[0]))

    nonem_correct = is_correct[Y_test_np == 0]
    print("Non-EM ACC = {}".format(np.sum(nonem_correct) /
                                   nonem_correct.shape[0]))

    em_tot = 0
    correct_em = 0
    op_list = []
    for test_file in tqdm(test_em_files):
        y, sr = librosa.load(test_file, sr=8000)
        classes = predict_op(y, scaler1)
        if classes == 1:
            correct_em += 1
        em_tot += 1

    print("------ File level results ------")
    print("Correct EM count = {}".format(correct_em))
    print("Total EM count = {}".format(em_tot))

    nonem_tot = 0
    correct_nonem = 0
    op_list = []
    for test_file in tqdm(test_nonem_files):
        y, sr = librosa.load(test_file, sr=8000)
        classes = predict_op(y, scaler1)
        if classes == 0:
            correct_nonem += 1
        nonem_tot += 1

    print("Correct Non-EM count = {}".format(correct_nonem))
    print("Total Non-EM count = {}".format(nonem_tot))

    test_file = os.path.join(test_path_em, '101.wav')
    y, sr = librosa.load(test_file, sr=8000)
    ipd.Audio(test_file)

    classes = predict_prob(y, scaler1)

    plt.figure()
    plt.plot(classes, c='r', linewidth=3.0, alpha=0.5)
    plt.yticks([0, 1])
    plt.ylim([-0.1, 1.1])
    plt.xlabel("Samples")
    plt.ylabel("Em signal presence")
    plt.grid('on')
    plt.show()

    log_S = melspectrogram(y, sr)
    # Make a new figure
    plt.figure(figsize=(12, 4))
    # Display the spectrogram on a mel scale
    # sample rate and hop length parameters are used to render the time axis
    librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')
