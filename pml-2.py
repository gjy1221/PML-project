import librosa
import os
import numpy as np
import pandas as pd
import skimage.util
import config
import pickle
import math
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, \
    f1_score, matthews_corrcoef, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import PolynomialFeatures

data_dir = 'humbugdb_neurips_2021'
dir_out = os.path.join('outputs')
rate = 8000
win_size = 30
step_size = 5
n_feat = 128
NFFT = 2048
n_hop = NFFT / 4
frame_duration = n_hop / rate  # Frame duration in ms

df = pd.read_csv('neurips_2021_zenodo_0_0_1.csv')

idx_test_A = np.logical_and(df['country'] == 'Tanzania', df['location_type'] == 'field')
idx_test_B = np.logical_and(df['country'] == 'UK', df['location_type'] == 'culture')
idx_train = np.logical_not(np.logical_or(idx_test_A, idx_test_B))
df_test_A = df[idx_test_A]
df_test_B = df[idx_test_B]


# Extract features from wave files with id corresponding to dataframe data_df.
def get_feat(data_df, df_dir, rates, min_duration, num_feat):
    X = []
    Y = []
    idx = 0
    for row in data_df.iterrows():
        idx += 1
        if idx % 100 == 0:
            print('Completed', idx, 'of', len(data_df))

        duration = row[1]['length']
        if duration > min_duration:
            filename = os.path.join(df_dir, str(row[1]['id']) + '.wav')
            length = librosa.get_duration(path=filename)

            if math.isclose(length, duration, rel_tol=0.01):
                signal, rates = librosa.load(filename, sr=rates)
                feat = librosa.feature.melspectrogram(y=signal, sr=rate, n_mels=num_feat)
                feat = librosa.power_to_db(feat, ref=np.max)

                feat = (feat - np.mean(feat)) / np.std(feat)
                X.append(feat)
                if row[1]['sound_type'] == 'mosquito':
                    Y.append(1)
                elif row[1]['sound_type']:  # Condition to check we are not adding empty (or unexpected) labels as 0
                    Y.append(0)
    return X, Y


# reshape the features
def reshape_feat(feats, labels, win_sizes, step_sizes):
    x = []
    y = []
    for idx, feat in enumerate(feats):
        if np.shape(feat)[1] >= win_sizes:
            feat_win = skimage.util.view_as_windows(feat.T, (win_sizes, np.shape(feat)[0]), step=step_sizes)
            label_win = np.full(len(feat_win), labels[idx])
            x.append(feat_win)
            y.append(label_win)
    X = np.vstack(x)
    Y = np.hstack(y)
    print(f"Shape of X after reshape: {np.array(X).shape}")
    print(f"Shape of Y after reshape: {np.array(Y).shape}")
    return X, Y


def get_train_test_from_df(train_df, test_A_df, test_B_df):
    pickle_name_train = 'feat_train_' + str(n_feat) + '_win_' + str(win_size) + '_step_' + str(
        step_size) + '.pickle'
    pickle_name_test = 'feat_test_' + str(n_feat) + '_win_' + str(win_size) + '_step_' + str(
        step_size) + '.pickle'

    if not os.path.isfile(os.path.join(dir_out, pickle_name_train)):
        print('Extracting training features:')
        X_train, Y_train = get_feat(data_df=train_df, df_dir=data_dir, rates=rate,
                                    min_duration=(win_size * frame_duration), num_feat=n_feat)
        X_train, Y_train = reshape_feat(feats=X_train, labels=Y_train, win_sizes=win_size, step_sizes=step_size)

        with open(os.path.join(dir_out, pickle_name_train), 'wb') as f:
            pickle.dump({'X_train': X_train, 'Y_train': Y_train}, f, protocol=4)
            print('Saved features to:', os.path.join(dir_out, pickle_name_train))
    else:
        print('Loading train features:', pickle_name_train)
        with open(os.path.join(dir_out, pickle_name_train), 'rb') as input_file:
            features = pickle.load(input_file)
            # print(features)
            X_train = features['X_train']
            Y_train = features['Y_train']

    if not os.path.isfile(os.path.join(dir_out, pickle_name_test)):
        print('Extracting testing features:')
        X_test_A, Y_test_A = get_feat(data_df=test_A_df, df_dir=data_dir, rates=rate,
                                      min_duration=(win_size * frame_duration), num_feat=n_feat)
        X_test_A, Y_test_A = reshape_feat(feats=X_test_A, labels=Y_test_A, win_sizes=win_size, step_sizes=step_size)

        X_test_B, Y_test_B = get_feat(data_df=test_B_df, df_dir=data_dir, rates=rate,
                                      min_duration=(win_size * frame_duration), num_feat=n_feat)
        X_test_B, Y_test_B = reshape_feat(feats=X_test_B, labels=Y_test_B, win_sizes=win_size, step_sizes=step_size)
        with open(os.path.join(dir_out, pickle_name_test), 'wb') as f:
            pickle.dump({'X_test_A': X_test_A, 'X_test_B': X_test_B, 'Y_test_A': Y_test_A, 'Y_test_B': Y_test_B}, f, protocol=4)
            print('Saved features to:', os.path.join(dir_out, pickle_name_test))
    else:
        print('Loading test features:', pickle_name_test)
        with open(os.path.join(dir_out, pickle_name_test), 'rb') as input_file:
            features = pickle.load(input_file)
            X_test_A = features['X_test_A']
            X_test_B = features['X_test_B']
            Y_test_A = features['Y_test_A']
            Y_test_B = features['Y_test_B']
    return X_train, Y_train, X_test_A, X_test_B, Y_test_A, Y_test_B


df_train = df[idx_train]

# Assertion to check that train does NOT appear in test:
assert len(np.where(pd.concat([df_train, df_test_A, df_test_B]).duplicated())[0]) == 0, (
    'Train dataframe contains overlap with Test A, Test B')

X_train, Y_train, X_test_A, X_test_B, Y_test_A, Y_test_B = get_train_test_from_df(df_train, df_test_A, df_test_B)

# Convert to numpy arrays
X_train = np.array(X_train)
y_train = np.array(Y_train)
X_test_A = np.array(X_test_A)
X_test_B = np.array(X_test_B)
y_test_A = np.array(Y_test_A)
y_test_B = np.array(Y_test_B)


# Check for NaN values in X_train
nan_indices = np.isnan(X_train)
if np.any(nan_indices):
    print("NaN values found in X_train. Handling NaN values...")
    # Option 1: Replace NaN with mean value
    from sklearn.impute import SimpleImputer

    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)

    # Option 2: Drop samples with NaN values
    # X_train = X_train[~np.any(nan_indices, axis=1)]


# # Reshape the feature arrays to 2D
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_A_flat = X_test_A.reshape(X_test_A.shape[0], -1)
X_test_B_flat = X_test_B.reshape(X_test_B.shape[0], -1)


# KNN Classifier
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train_flat, y_train)
# Predictions
y_pred = model.predict(X_test_B_flat)


# # Logistic Regression Classifier
# model = LogisticRegression(max_iter=1000)
# model.fit(X_train_flat, y_train)
# y_pred = model.predict(X_test_B_flat)


# Calculate evaluation metrics
accuracy = accuracy_score(y_test_B, y_pred)
precision = precision_score(y_test_B, y_pred)
recall = recall_score(y_test_B, y_pred)
f1 = f1_score(y_test_B, y_pred)
mcc = matthews_corrcoef(y_test_B, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("MCC:", mcc)

# Confusion Matrix
print(confusion_matrix(y_test_B, y_pred))

# ROC Curve
# y_scores = model.predict_proba(X_test_B_flat)[:, 1]  # probability estimates of the positive class
# fpr, tpr, thresholds = roc_curve(y_test_B, y_scores)
# roc_auc = roc_auc_score(y_test_B, y_scores)

# Accuracy: 0.7141541295670862
# Precision: 0.9387254901960784
# Recall: 0.21829581077229981
# F1-score: 0.35421965317919074
# MCC: 0.36469736169730954
# [[6212   50]
#  [2743  766]]
