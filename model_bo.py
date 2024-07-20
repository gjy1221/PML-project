import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, \
    f1_score, matthews_corrcoef, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from joblib import load
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
# 加载数据
data_f_dir = os.path.join('data_f_dir')
X_train_resampled = load((os.path.join(data_f_dir, 'X_train_resampled.joblib')))
y_train_resampled = load((os.path.join(data_f_dir, 'y_train_resampled.joblib')))
X_train_flat = load((os.path.join(data_f_dir, 'X_train_flat.joblib')))
X_test_A_flat = load((os.path.join(data_f_dir, 'X_test_A_flat')))
X_test_B_flat = load((os.path.join(data_f_dir, 'X_test_B_flat')))
# y_train = load((os.path.join(data_f_dir, 'y_train.joblib'))
# X_train = load((os.path.join(data_f_dir, 'X_train.joblib'))
y_test_A = load((os.path.join(data_f_dir, 'y_test_A.joblib')))
y_test_B = load((os.path.join(data_f_dir, 'y_test_B.joblib')))

y_test = y_test_B
X_test = X_test_B_flat
X_train = X_train_resampled
y_train = y_train_resampled

# KNN Classifier
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)
# Predictions
y_pred = model.predict(X_test)


# # Logistic Regression Classifier
# model = LogisticRegression(max_iter=1000)
# model.fit(X_train_flat, y_train)
# y_pred = model.predict(X_test_B_flat)



# # SVM Classifier
# model = SVC(kernel='rbf')
# model.fit(X_train_flat, y_train)
# # Predictions
# y_pred = model.predict(X_test_B_flat)


# # RandomForest Classifier
# model = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=2024)
# # Predictions
# model.fit(X_train_flat, y_train)
# y_pred = model.predict(X_test_B_flat)


# # 初始化朴素贝叶斯分类器，这里使用高斯朴素贝叶斯
# model = GaussianNB()
# model.fit(X_train_flat, y_train)
# # Predictions
# y_pred = model.predict(X_test)

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
y_scores = model.predict_proba(X_test)[:, 1]  # probability estimates of the positive class
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