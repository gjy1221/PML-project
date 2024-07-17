import librosa
import os
import numpy as np
import pandas as pd
import skimage.util
from sklearn.preprocessing import label_binarize

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
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.model_selection import train_test_split

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
                    Y.append(0)
                elif row[1]['sound_type'] == 'audio':
                    Y.append(1)
                elif row[1]['sound_type'] == 'background':
                    Y.append(2)

                # if row[1]['sound_type'] == 'mosquito':
                #     Y.append(1)
                # elif row[1]['sound_type']:
                #     Y.append(0)

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


# 检查是否有可用的GPU加速
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 调整特征数组的形状为3D以适应LSTM
X_train_lstm = np.reshape(X_train, (X_train.shape[0] * X_train.shape[1], X_train.shape[2], X_train.shape[3]))
X_test_B_lstm = np.reshape(X_test_B, (X_test_B.shape[0] * X_test_B.shape[1], X_test_B.shape[2], X_test_B.shape[3]))

print(np.shape(Y_train), np.shape(X_train_lstm))

# SPLIT VAL+TRAIN
X_train_lstm, X_val_lstm, y_train, y_val = train_test_split(X_train_lstm, y_train, test_size=0.2, random_state=42)


X_train_tensor = torch.tensor(X_train_lstm, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_B_tensor = torch.tensor(X_test_B_lstm, dtype=torch.float32)
y_test_B_tensor = torch.tensor(Y_test_B, dtype=torch.long)
X_val_tensor = torch.tensor(X_val_lstm, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)


# load data by dataloader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_B_dataset = TensorDataset(X_test_B_tensor, y_test_B_tensor)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_B_loader = DataLoader(test_B_dataset, batch_size=batch_size, shuffle=False)


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
hidden_size = 32
num_layers = 2
output_size = 3

model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
#
num_epochs = 10
best_test_loss = float('inf')
best_model_path = ''

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    test_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # 评估模型
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            test_loss += loss.item()

    print(f'Epoch [{epoch + 1}/{num_epochs}], trainingLoss: {running_loss / len(train_loader):.4f}, testLoss: {test_loss / len(val_loader):.4f}')

    # 检查是否是最佳模型
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        best_model_path = os.path.join('outputs', f'{test_loss / len(val_loader):.4f}.pth')
        torch.save(model.state_dict(), best_model_path)
        print(f'Saved Best Model with test loss: {best_test_loss / len(val_loader):.4f}')

# best_model_path = 'outputs/0.561932053006554.pth'

# 加载最佳模型
best_model = LSTMModel(input_size, hidden_size, num_layers, output_size)
best_model.load_state_dict(torch.load(best_model_path))
best_model.eval()

# 使用测试集进行模型评估
model.eval()
test_correct = 0
all_test_preds = []
all_test_labels = []
all_test_probs = []


with torch.no_grad():
    for inputs, labels in test_B_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)

        _, predicted = torch.max(outputs, 1)
        test_correct += (predicted == labels).sum().item()

        all_test_preds.extend(predicted.cpu().numpy())
        all_test_labels.extend(labels.cpu().numpy())
        all_test_probs.extend(F.softmax(outputs, dim=1).cpu().numpy())  # 使用softmax获得每个类别的概率

# 计算并显示评估指标
conf_mat = confusion_matrix(all_test_labels, all_test_preds)
print("Confusion Matrix:")
print(conf_mat)

accuracy = accuracy_score(all_test_labels, all_test_preds)
print(f"Accuracy: {accuracy:.4f}")

precision = precision_score(all_test_labels, all_test_preds, average=None)
recall = recall_score(all_test_labels, all_test_preds, average=None)
f1 = f1_score(all_test_labels, all_test_preds, average=None)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

mcc = matthews_corrcoef(all_test_labels, all_test_preds)
print(f"MCC: {mcc:.4f}")

# 计算并绘制多分类的ROC曲线和AUC
y_test_binarized = label_binarize(all_test_labels, classes=[0, 1, 2])
n_classes = y_test_binarized.shape[1]

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], np.array(all_test_probs)[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure()
colors = ['blue', 'red', 'green']
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='ROC curve of class {0} (area = {1:0.2f})'
                   ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic for Multi-class')
plt.legend(loc="lower right")
plt.show()

overall_roc_auc = roc_auc_score(y_test_binarized, np.array(all_test_probs), average="macro")
print(f"Overall ROC AUC Score: {overall_roc_auc:.4f}")


# 计算TPR, TNR, 和 PR
TPR = recall
TNR = []
for i in range(conf_mat.shape[0]):
    tn = conf_mat[i, i]  # True Negatives for class i
    fn_fp = conf_mat.sum(axis=1)[i] + conf_mat.sum(axis=0)[i] - 2 * tn  # False Negatives + False Positives for class i
    TNR.append(tn / (tn + fn_fp))

print("TPR:", TPR)
print("TNR:", TNR)

