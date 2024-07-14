from PML import X_train_flat, y_train, X_test_B_flat, y_test_B, X_test_A_flat, y_test_A,n_samples, \
    n_time_steps, n_features, n_channels, X_train, y_test_A, X_test_A, X_test_B
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Conv2D, MaxPooling2D, Flatten, Dense, Dropout

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, \
    f1_score, matthews_corrcoef, roc_curve, roc_auc_score

# 这里我们使用 X_train_flat 和 y_train 作为训练数据
# 使用 X_test_A_flat 和 y_test_A 作为测试数据
y_test = y_test_B
X_test = X_test_B

# # 定义模型
# model = Sequential([
#     Flatten(input_shape=(X_train_flat.shape[1],)),  # 假设 X_train_flat 是一个二维数组
#     Dense(128, activation='relu'),
#     Dense(64, activation='relu'),
#     Dense(32, activation='relu'),
#     Dense(32, activation='relu'),
#     Dense(1, activation='sigmoid')  # 二分类问题，使用sigmoid激活函数
# ])
#
# # 编译模型
# model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# 设定输入形状
input_shape = (n_time_steps, n_features, n_channels)

model = Sequential([
    # 第一层卷积层
    Conv2D(32, kernel_size=(1, 3), activation='relu', input_shape=input_shape),
    MaxPooling2D(pool_size=(1, 2)),
    # 第二层卷积层
    Conv2D(64, kernel_size=(1, 3), activation='relu'),
    MaxPooling2D(pool_size=(1, 2)),
    # 第三层卷积层
    Conv2D(128, kernel_size=(1, 3), activation='relu'),
    MaxPooling2D(pool_size=(1, 2)),
    # 展平层
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')  # 假设是二分类问题

])
# 编译模型
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 训练模型
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# 评估模型
loss, accuracy = model.evaluate(X_test, y_test)
print(f"测试损失: {loss}, 测试准确率: {accuracy}")
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