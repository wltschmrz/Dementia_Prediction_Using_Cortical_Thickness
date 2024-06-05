import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)

# 파일 불러오기 & ptid 순으로 정렬(df index는 유지)
data = pd.read_csv('corticalthickness.csv')
data = data.sort_values(by='ptid', ascending=True)

# Features & Target 생성
# data.columns = ['ptid', 'dx', 'age', 'sex', 'edu', 'napoe4', 'mmse', 'apos', 'icv', '{68}']
Features = list(data.columns)
del Features[5:7]; del Features[:2]
Target = 'mmse'

# Feature selection - 일단은 r_entorhinal 이외의 cortical thickness 제외함. filter method를 이용하도록 하자
del Features[5:]
Features.append('r_entorhinal')
Features.insert(3, 'napoe4_0')
Features.insert(4, 'napoe4_1')
print(Features, end='\n'*2)

dummy_napoe4 = pd.get_dummies(data['napoe4'], prefix='napoe4')
data = pd.concat([data, dummy_napoe4], axis=1)

# Test set & Train set 분할
test_percentage = int(len(data) * 0.2)
train = data.tail(len(data) - test_percentage)
test = data.head(test_percentage)

X_train = train[Features]
X_test = test[Features]

y_train = train[Target]
y_test = test[Target]

print(X_train)

from sklearn.metrics import mean_squared_error as mse
from sklearn.svm import SVR

models = []

for gamma in [1, 0.1, 0.01, 0.001, 0.0001]:
    models.append(SVR(kernel='rbf', gamma=gamma, C=1.0, epsilon=0.1).fit(X_train, y_train))

# for gamma in [1, 0.1, 0.01, 0.001, 0.0001]:
#     models.append(SVR(kernel='poly', gamma=gamma, coef0=0, C=1.0, epsilon=0.1, degree=2).fit(X_train, y_train))

for gamma in [1, 0.1, 0.01, 0.001, 0.0001]:
    models.append(SVR(kernel='sigmoid', gamma=gamma, coef0=0, C=1.0, epsilon=0.1).fit(X_train, y_train))

print('model_running: Done')

test_performance_measure = []

for model in models:
    predict = model.predict(X_test)
    test_performance_measure.append(mse(y_test, predict))

print('model_evaluating: Done')

best_support_vector_regression = models[np.argmin(test_performance_measure)]
names = ['%s_%s' %(model.kernel, model.gamma) for model in models]
colors = ['dodgerblue' if e != min(test_performance_measure) else 'r' for e in test_performance_measure]
plt.bar(range(len(models)), test_performance_measure, color=colors)
plt.ylim(7, 15)
plt.xticks(range(len(models)), labels=names, rotation=90)
plt.show()

