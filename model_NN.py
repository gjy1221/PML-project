import os
import numpy as np
from joblib import load
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, \
    SimpleRNN, Input, AveragePooling1D, GlobalAveragePooling1D, Dropout
from tensorflow.keras.regularizers import l2, l1
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, \
    f1_score, matthews_corrcoef, roc_curve, roc_auc_score

# 加载数据
data_f_dir = os.path.join('data_f_dir')
X_train_resampled = load((os.path.join(data_f_dir, 'X_train_resampled.joblib')))
y_train_resampled = load((os.path.join(data_f_dir, 'y_train_resampled.joblib')))
X_train_flat = load((os.path.join(data_f_dir, 'X_train_flat.joblib')))
X_test_A_flat = load((os.path.join(data_f_dir, 'X_test_A_flat')))
X_test_B_flat = load((os.path.join(data_f_dir, 'X_test_B_flat')))
y_train_tensor = load((os.path.join(data_f_dir, 'y_train.joblib')))
X_train_tensor = load((os.path.join(data_f_dir, 'X_train.joblib')))
# y_train_tensor_resampled = load((os.path.join(data_f_dir, 'y_train_tensor_resampled.joblib')))
# X_train_tensor_resampled = load((os.path.join(data_f_dir, 'X_train_tensor_resampled.joblib')))
y_test_A = load((os.path.join(data_f_dir, 'y_test_A.joblib')))
y_test_B = load((os.path.join(data_f_dir, 'y_test_B.joblib')))
X_test_A = load((os.path.join(data_f_dir, 'X_test_A.joblib')))
X_test_B = load((os.path.join(data_f_dir, 'X_test_B.joblib')))

#输入数据
y_test = y_test_B
X_test = X_test_B
X_train = X_train_tensor
y_train = y_train_tensor

n_samples, n_no, n_w, n_h = X_train_tensor.shape
print(X_train_tensor.shape)
# # 定义dnn模型
#
# model = Sequential([
#     Flatten(input_shape=(X_train.shape[1], )),  # 假设 X_train_flat 是一个二维数组
#     Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
#     Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
#     Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
#     Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
#     Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
#     Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
#     Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
#     Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
#     Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
#     Dense(1, activation='sigmoid')  # 二分类问题，使用sigmoid激活函数
# ])

# # 编译模型
# model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# 定义cnn模型
# 设定输入形状
# 去掉多余特征
X_train = np.squeeze(X_train)
X_test = np.squeeze(X_test)
print(X_train.shape)
input_shape = (n_w, n_h)
# 创建模型
model = Sequential()

# 添加卷积层
model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape))
# model.add(AveragePooling1D(pool_size=2))  # 添加最大池化层
model.add(Dropout(0.5))

# 添加第二个卷积层
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(AveragePooling1D(pool_size=2))  # 添加平均池化层

# 添加第二个卷积层
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(AveragePooling1D(pool_size=2))

# 添加第三个卷积层
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(GlobalAveragePooling1D())  # 添加全局平均池化层

# 展平层
# GlobalAveragePooling1D 已经展平输出，所以不需要 Flatten 层

# 全连接层
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))  # 添加50%的Dropout

# 输出层
model.add(Dense(1, activation='sigmoid'))

# 编译模型
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])



# model = Sequential([
#     # 第一层卷积层
#     Conv2D(32, kernel_size=(1, 3), activation='relu', input_shape=input_shape, padding='same'),
#     MaxPooling2D(pool_size=(1, 2)),
#     # 第二层卷积层
#     Conv2D(64, kernel_size=(1, 3), activation='relu', padding='same'),
#
#     # 第三层卷积层
#     Conv2D(64, kernel_size=(1, 3), activation='relu', padding='same'),
#
#     # 第四层卷积层
#     Conv2D(64, kernel_size=(1, 3), activation='relu', padding='same'),
#
#     # 展平层
#     Flatten(),
#     Dense(128, activation='relu'),
#     Dense(1, activation='sigmoid')  # 假设是二分类问题
#
# ])
# # 编译模型
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# #
# # 构建RNN模型
# # 有些问题，RNN需要三维的张量
# X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[2], X_test.shape[3]))
# X_train = np.reshape(X_train_tensor, (X_train_tensor.shape[0], X_train_tensor.shape[2], X_train_tensor.shape[3]))
#
#
#
# model = Sequential([
#     SimpleRNN(units=256, input_shape=(n_features, n_channels)),  # 注意 input_shape 的设置
#     Dense(1, activation='sigmoid')  # 假设你有一个二分类任务
# ])
#
# # 编译模型
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 训练模型
model.fit(X_train, y_train, epochs=10, batch_size=16, validation_split=0.2)

# 评估模型
# loss, accuracy = model.evaluate(X_test, y_test)
# print(f"测试损失: {loss}, 测试准确率: {accuracy}")
# y_probs 是模型预测的正类概率，y_test 是真实的类别标签
y_probs = model.predict(X_test).ravel()
threshold = 0.5  # 使用0.5作为阈值
y_pred = (y_probs > threshold).astype(int)  # 将概率转换为类别标签

# Calculate evaluation metrics

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("MCC:", mcc)

# Confusion Matrix
print(confusion_matrix(y_test, y_pred))

# ROC Curve
y_scores = model.predict(X_test).ravel()  # probability estimates of the positive class
fpr, tpr, thresholds = roc_curve(y_test, y_scores)
roc_auc = roc_auc_score(y_test, y_scores)

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