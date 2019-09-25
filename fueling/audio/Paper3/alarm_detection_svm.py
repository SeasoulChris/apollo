#!/usr/bin/env python
# coding: utf-8

# In[1]:

import warnings
import sys

import numpy as np
import pandas as pd

from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils import shuffle

from pyAudioAnalysis import audioFeatureExtraction

import seaborn as sns
import matplotlib.ticker as ticker
import matplotlib.style as ms
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, hilbert
import argparse
import os
import glob
import time

import librosa
import librosa.display
from tqdm import tqdm

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


def predict_op(y, scaler, model, N=20):
    y = preprocess(y)
    feature_list = audioFeatureExtraction.stFeatureExtraction(
        y, sr, 0.10*sr, .05*sr)
    scaler.transform(feature_list)
    count = 0
    th = 0.5

    prob_list = []
    class_list = []
    for i in range(N):
        p = model.predict_proba(feature_list[i].reshape(1, 105))
        p = np.array(p)
        p = p[0, 1]
        prob_list.append(p)
    prob = np.mean(prob_list)
    if prob > th:
        # print("Em")
        class_list.append(1)
    else:
        # print("Non-em")
        class_list.append(0)

    for i in range(N, len(feature_list)):
        prob_list.pop(0)
        p = model.predict_proba(feature_list[i].reshape(1, 105))
        p = np.array(p)
        p = p[0, 1]
        prob_list.append(p)
        prob = np.mean(prob_list)
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


def predict_prob(y, scaler, model, N=20):
    y = preprocess(y)
    feature_list = audioFeatureExtraction.stFeatureExtraction(
        y, sr, 0.10*sr, .05*sr)
    scaler.transform(feature_list)
    count = 0
    th = 0.5

    prob_list = []
    class_list = []
    for i in range(N):
        p = model.predict_proba(feature_list[i].reshape(1, 105))
        # print(p)
        p = np.array(p)
        p = p[0, 1]
        print("p = {}".format(p))
        # p = p.flatten()
        prob_list.append(p)
    prob = np.mean(prob_list)
    print("prob = {}".format(prob))
    # print(prob)
    if prob > th:
        # print("Em")
        class_list.append(1)
    else:
        # print("Non-em")
        class_list.append(0)

    for i in range(N, len(feature_list)):
        prob_list.pop(0)
        p = model.predict_proba(feature_list[i].reshape(1, 105))
        p = np.array(p)
        p = p[0, 1]
        print("p = {}".format(p))
        # p = p.flatten()
        prob_list.append(p)
        prob = np.mean(prob_list)
        print("prob = {}".format(prob))
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

    # fix random seed for reproducibility
    np.random.seed(7)

    print("------------ Start training SVM model ---------------")
    model = SVC(gamma='auto', probability=True)
    model.fit(X_train, Y_train)

    print("------------ Finish training SVM model ---------------")

    Y_pred = model.predict(X_test)
    cm = confusion_matrix(Y_pred, Y_test)
    df_cm = pd.DataFrame(cm, index=['Non-EM', 'EM'],
                         columns=['Non-EM', 'EM'])
    plt.figure(figsize=(8, 6))
    sns.heatmap(df_cm, annot=True, cmap='YlGnBu')

    pred_test = np.array(model.predict(X_test))
    pred_test

    Y_test_np = np.array(Y_test)
    Y_test_em = Y_test_np[Y_test_np == 1]
    Y_test_nonem = Y_test_np[Y_test_np == 0]

    print("------ data level results -----")
    is_correct = pred_test == Y_test_np
    print("Overall ACC = {}".format(np.mean(is_correct)))

    em_correct = is_correct[Y_test_np == 1]
    print("EM ACC = {}".format(np.mean(em_correct)))

    nonem_correct = is_correct[Y_test_np == 0]
    print("Non-EM ACC = {}".format(np.mean(nonem_correct)))

    em_tot = 0
    correct_em = 0
    op_list = []
    for test_file in tqdm(test_em_files):
        y, sr = librosa.load(test_file, sr=8000)
        classes = predict_op(y, scaler1, model, 10)
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
        classes = predict_op(y, scaler1, model, 10)
        if classes == 0:
            correct_nonem += 1
        nonem_tot += 1

    print("Correct Non-EM count = {}".format(correct_nonem))
    print("Total Non-EM count = {}".format(nonem_tot))

    test_file = os.path.join(test_path_em, '101.wav')
    y, sr = librosa.load(test_file, sr=8000)

    classes = predict_prob(y, scaler1, model, 20)

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
