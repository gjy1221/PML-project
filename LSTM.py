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
    f1_score, matthews_corrcoef, roc_curve, roc_auc_score, precision_recall_curve, roc_curve, auc
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

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
            pickle.dump({'X_test_A': X_test_A, 'X_test_B': X_test_B, 'Y_test_A': Y_test_A, 'Y_test_B': Y_test_B}, f,
                        protocol=4)
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


# 检查是否有可用的GPU加速
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 调整特征数组的形状为3D以适应LSTM
X_train_lstm = np.reshape(X_train, (X_train.shape[0] * X_train.shape[1], X_train.shape[2], X_train.shape[3]))
X_test_B_lstm = np.reshape(X_test_B, (X_test_B.shape[0] * X_test_B.shape[1], X_test_B.shape[2], X_test_B.shape[3]))

print(np.shape(Y_train), np.shape(X_train_lstm))


X_train_tensor = torch.tensor(X_train_lstm, dtype=torch.float32)
y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)
X_test_B_tensor = torch.tensor(X_test_B_lstm, dtype=torch.float32)
y_test_B_tensor = torch.tensor(Y_test_B, dtype=torch.float32)

# load data by dataloader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_B_dataset = TensorDataset(X_test_B_tensor, y_test_B_tensor)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_B_loader = DataLoader(test_B_dataset, batch_size=batch_size, shuffle=False)


#
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # 只获取最后一个时间步的输出
        out = self.sigmoid(out)
        return out


# 初始化模型、损失函数和优化器
input_size = n_feat
hidden_size = 128
num_layers = 4
output_size = 1

model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch.view(-1, 1))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')

# 评估模型
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for X_batch, y_batch in test_B_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch)
        preds = (outputs > 0.5).float()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

# 计算评估指标
all_preds = np.array(all_preds).flatten()
all_labels = np.array(all_labels).flatten()
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
mcc = matthews_corrcoef(all_labels, all_preds)

# 计算PR曲线和AUC
precision_vals, recall_vals, _ = precision_recall_curve(all_labels, all_preds)
pr_auc = auc(recall_vals, precision_vals)

# 计算ROC曲线和AUC
fpr, tpr, _ = roc_curve(all_labels, all_preds)
roc_auc = auc(fpr, tpr)

# 计算TPR和TNR
tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
tpr = tp / (tp + fn)  # 真阳性率
tnr = tn / (tn + fp)  # 真阴性率

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("MCC:", mcc)
print("TPR (True Positive Rate):", tpr)
print("TNR (True Negative Rate):", tnr)
print("PR AUC:", pr_auc)
print("ROC AUC:", roc_auc)
print("Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))


# Accuracy: 0.7188619383891106
# Precision: 0.8587570621468926
# Recall: 0.2599031062980906
# F1-score: 0.39903740975716473
# MCC: 0.36368059886479226
# TPR (True Positive Rate): 0.2599031062980906
# TNR (True Negative Rate): 0.9760459916959437
# PR AUC: 0.6922233397746357
# ROC AUC: 0.6179745489970171
# Confusion Matrix:
# [[6112  150]
#  [2597  912]]
