import os
import pandas as pd
import numpy as np
import librosa
import skimage.util
import config
import pickle
import math
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, \
    f1_score, matthews_corrcoef, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('neurips_2021_zenodo_0_0_1.csv')

# To be kept: please do not edit the test set: these paths select test set A, test set B as described in the paper
idx_test_A = np.logical_and(df['country'] == 'Tanzania', df['location_type'] == 'field')
idx_test_B = np.logical_and(df['country'] == 'UK', df['location_type'] == 'culture')
idx_train = np.logical_not(np.logical_or(idx_test_A, idx_test_B))
df_test_A = df[idx_test_A]
df_test_B = df[idx_test_B]


# Extract features from wave files with id corresponding to dataframe data_df.
def get_feat(data_df, data_dir, rate, min_duration, n_feat):
    ''' Returns features extracted with Librosa. A list of features, with the number of items equal to the number of input recordings'''
    X = []
    y = []
    bugs = []
    idx = 0
    skipped_files = []
    for row_idx_series in data_df.iterrows():
        idx += 1
        if idx % 100 == 0:
            print('Completed', idx, 'of', len(data_df))
        row = row_idx_series[1]
        label_duration = row['length']
        if label_duration > min_duration:
            _, file_format = os.path.splitext(row['name'])
            filename = os.path.join(data_dir, str(row['id']) + file_format)
            length = librosa.get_duration(path=filename)
            #             assert math.isclose(length,label_duration, rel_tol=0.01), "File: %s label duration (%.4f) does not match audio length (%.4f)" % (row['path'], label_duration, length)

            if math.isclose(length, label_duration, rel_tol=0.01):
                signal, rate = librosa.load(filename, sr=rate)
                feat = librosa.feature.melspectrogram(y=signal, sr=rate, n_mels=n_feat)
                feat = librosa.power_to_db(feat, ref=np.max)
                if config.norm_per_sample:
                    feat = (feat - np.mean(feat)) / np.std(feat)
                X.append(feat)
                if row['sound_type'] == 'mosquito':
                    y.append(1)
                elif row['sound_type']:  # Condition to check we are not adding empty (or unexpected) labels as 0
                    y.append(0)
            else:
                print("File: %s label duration (%.4f) does not match audio length (%.4f)" % (
                    row['name'], label_duration, length))
                bugs.append([row['name'], label_duration, length])

        else:
            skipped_files.append([row['id'], row['name'], label_duration])
    return X, y, skipped_files, bugs


def reshape_feat(feats, labels, win_size, step_size):
    '''Reshaping features from get_feat to be compatible for classifiers expecting a 2D slice as input. Parameter `win_size` is
    given in number of feature windows (in librosa this is the hop length divided by the sample rate.)
    Can code to be a function of time and hop length instead in future.'''

    feats_windowed_array = []
    labels_windowed_array = []
    for idx, feat in enumerate(feats):
        if np.shape(feat)[1] < win_size:
            print('Length of recording shorter than supplied window size.')
            pass
        else:
            feats_windowed = skimage.util.view_as_windows(feat.T, (win_size, np.shape(feat)[0]), step=step_size)
            labels_windowed = np.full(len(feats_windowed), labels[idx])
            feats_windowed_array.append(feats_windowed)
            labels_windowed_array.append(labels_windowed)
    return np.vstack(feats_windowed_array), np.hstack(labels_windowed_array)


def get_train_test_from_df(df_train, df_test_A, df_test_B, debug=False):
    pickle_name_train = 'log_mel_feat_train_' + str(config.n_feat) + '_win_' + str(config.win_size) + '_step_' + str(
        config.step_size) + '_norm_' + str(config.norm_per_sample) + '.pickle'
    # step = window for test (no augmentation of test):
    pickle_name_test = 'log_mel_feat_test_' + str(config.n_feat) + '_win_' + str(config.win_size) + '_step_' + str(
        config.win_size) + '_norm_' + str(config.norm_per_sample) + '.pickle'

    if not os.path.isfile(os.path.join(config.dir_out_MED, pickle_name_train)):
        print('Extracting training features...')
        X_train, y_train, skipped_files_train, bugs_train = get_feat(data_df=df_train, data_dir=config.data_dir,
                                                                     rate=config.rate, min_duration=config.min_duration,
                                                                     n_feat=config.n_feat)
        X_train, y_train = reshape_feat(X_train, y_train, config.win_size, config.step_size)

        log_mel_feat_train = {'X_train': X_train, 'y_train': y_train, 'bugs_train': bugs_train}

        if debug:
            print('Bugs train', bugs_train)

        with open(os.path.join(config.dir_out_MED, pickle_name_train), 'wb') as f:
            pickle.dump(log_mel_feat_train, f, protocol=4)
            print('Saved features to:', os.path.join(config.dir_out_MED, pickle_name_train))

    else:
        print('Loading training features found at:', os.path.join(config.dir_out_MED, pickle_name_train))
        with open(os.path.join(config.dir_out_MED, pickle_name_train), 'rb') as input_file:
            log_mel_feat = pickle.load(input_file)
            X_train = log_mel_feat['X_train']
            y_train = log_mel_feat['y_train']

    if not os.path.isfile(os.path.join(config.dir_out_MED, pickle_name_test)):
        print('Extracting test features...')

        X_test_A, y_test_A, skipped_files_test_A, bugs_test_A = get_feat(data_df=df_test_A, data_dir=config.data_dir,
                                                                         rate=config.rate,
                                                                         min_duration=config.min_duration,
                                                                         n_feat=config.n_feat)
        X_test_B, y_test_B, skipped_files_test_B, bugs_test_B = get_feat(data_df=df_test_B, data_dir=config.data_dir,
                                                                         rate=config.rate,
                                                                         min_duration=config.min_duration,
                                                                         n_feat=config.n_feat)
        X_test_A, y_test_A = reshape_feat(X_test_A, y_test_A, config.win_size,
                                          config.win_size)  # Test should be strided with step = window.
        X_test_B, y_test_B = reshape_feat(X_test_B, y_test_B, config.win_size, config.win_size)

        log_mel_feat_test = {'X_test_A': X_test_A, 'X_test_B': X_test_B, 'y_test_A': y_test_A, 'y_test_B': y_test_B}

        if debug:
            print('Bugs test A', bugs_test_A)
            print('Bugs test B', bugs_test_B)

        with open(os.path.join(config.dir_out_MED, pickle_name_test), 'wb') as f:
            pickle.dump(log_mel_feat_test, f, protocol=4)
            print('Saved features to:', os.path.join(config.dir_out_MED, pickle_name_test))
    else:
        print('Loading test features found at:', os.path.join(config.dir_out_MED, pickle_name_test))
        with open(os.path.join(config.dir_out_MED, pickle_name_test), 'rb') as input_file:
            log_mel_feat = pickle.load(input_file)

            X_test_A = log_mel_feat['X_test_A']
            y_test_A = log_mel_feat['y_test_A']
            X_test_B = log_mel_feat['X_test_B']
            y_test_B = log_mel_feat['y_test_B']

    return X_train, y_train, X_test_A, y_test_A, X_test_B, y_test_B


df_train = df[idx_train]

# Modify by addition or sub-sampling of df_train here
# df_train ...

# Assertion to check that train does NOT appear in test:
assert len(np.where(pd.concat([df_train, df_test_A, df_test_B]).duplicated())[0]) == 0, (
    'Train dataframe contains overlap with Test A, Test B')

X_train, y_train, X_test_A, y_test_A, X_test_B, y_test_B = get_train_test_from_df(df_train, df_test_A, df_test_B,
                                                                                  debug=True)

# Convert to numpy arrays
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test_A = np.array(X_test_A)
X_test_B = np.array(X_test_B)
y_test_A = np.array(y_test_A)
y_test_B = np.array(y_test_B)

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

# 将X_train从四维数组展平成二维数组
n_samples, n_time_steps, n_features, n_channels = X_train.shape
X_train_flat = X_train.reshape(n_samples, n_time_steps * n_features * n_channels)

# 对测试数据集进行相同的展平处理
n_samples_A, n_time_steps_A, n_features_A, n_channels_A = X_test_A.shape
X_test_A_flat = X_test_A.reshape(n_samples_A, n_time_steps_A * n_features_A * n_channels_A)

n_samples_B, n_time_steps_B, n_features_B, n_channels_B = X_test_B.shape
X_test_B_flat = X_test_B.reshape(n_samples_B, n_time_steps_B * n_features_B * n_channels_B)

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
y_scores = model.predict_proba(X_test_B_flat)[:, 1]  # probability estimates of the positive class
fpr, tpr, thresholds = roc_curve(y_test_B, y_scores)
roc_auc = roc_auc_score(y_test_B, y_scores)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
