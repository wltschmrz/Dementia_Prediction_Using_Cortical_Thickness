import pandas as pd

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# 파일 불러오기 & ptid 순으로 정렬(df index는 유지)
data = pd.read_csv('corticalthickness.csv')
data = data.sort_values(by='ptid', ascending=True)

# Features & Target 생성
Features = list(data.columns)
del Features[6]
del Features[:2]
Target = 'mmse'

# Feature selection - 일단은 r_entorhinal 이외의 cortical thickness 제외함
del Features[6:]
Features.append('r_entorhinal')
print(Features, end='\n'*2)

# Test set & Train set 분할
test_percentage = int(len(data) * 0.2)
train = data.tail(len(data) - test_percentage)
test = data.head(test_percentage)

X_train = train[Features]
X_test = test[Features]

y_train = train[Target]
y_test = test[Target]



# #------------------------------------------
# # 산점도 행렬 그리기
# import seaborn as sns
# import matplotlib.pyplot as plt
#
# train_selected = train[Features]
#
# sns.pairplot(train_selected)
# plt.show()
# #------------------------------------------


# Linear Regression 진행
from scipy.optimize import linear_sum_assignment # 이거 어디에 쓰는거지
from sklearn.linear_model import LinearRegression

linear_regression = LinearRegression()
linear_regression.fit(X_train, y_train)

print('LinearRegression')
print(pd.DataFrame(dict(coef=linear_regression.coef_), index=Features).transpose())
print('intercept', linear_regression.intercept_, end='\n'*2)

# Huber regression
from sklearn.linear_model import HuberRegressor

huber_regressor = HuberRegressor(epsilon=1.35, max_iter=100, fit_intercept=True, alpha=0.0)
huber_regressor.fit(X_train, y_train)

print('Huber regression')
print(pd.DataFrame(dict(coef=huber_regressor.coef_), index=Features).transpose())
print('intercept', huber_regressor.intercept_, end='\n'*2)


# Linear support vector regression
from sklearn.svm import SVR

linear_svr_models = {}
for e in [0.1, 0.01, 0.001]:
    linear_svr = SVR(kernel='linear', C=10, epsilon=e)
    linear_svr.fit(X_train, y_train)

    linear_svr_models[e] = linear_svr
    print('Linear Support vector regression, epsilon=%s' % e)
    print(pd.DataFrame(dict(coef=linear_svr.coef_.ravel()), index=Features).transpose())
    print('intercept', linear_svr.intercept_.ravel(), end='\n'*2)

# ------------------------------------------------------------------------


# Evaluation metrics
from sklearn import metrics
import numpy as np
result = {}
for name, model in zip(['linear regression', 'huber regression', 'linear SVR, epsilon=0.1', 'linear SVR, epsilon=0.01', 'linear SVR, epsilon=0.001'],
                       [linear_regression, huber_regressor, linear_svr_models[0.1], linear_svr_models[0.01], linear_svr_models[0.001]]):
    predict = model.predict(X_train)
    MAE = metrics.mean_absolute_error(y_train, predict)
    MSE = metrics.mean_squared_error(y_train, predict)
    RMSE = np.sqrt(MSE)
    MAPE = metrics.mean_absolute_percentage_error(y_train, predict)
    R2 = metrics.r2_score(y_train, predict)
    result[name] = [MAE, MSE, RMSE, MAPE, R2]

print('Train set')
print(pd.DataFrame(result, index=['MAE', 'MSE', 'RMSE', 'MAPE', 'R2']).transpose())


print('\n\n')


result = {}
for name, model in zip(['linear regression', 'huber regression', 'linear SVR, epsilon=0.1', 'linear SVR, epsilon=0.01',
                        'linear SVR, epsilon=0.001'],
                       [linear_regression, huber_regressor, linear_svr_models[0.1], linear_svr_models[0.01],
                        linear_svr_models[0.001]]):
    predict = model.predict(X_test)
    MAE = metrics.mean_absolute_error(y_test, predict)
    MSE = metrics.mean_squared_error(y_test, predict)
    RMSE = np.sqrt(MSE)
    MAPE = metrics.mean_absolute_percentage_error(y_test, predict)
    R2 = metrics.r2_score(y_test, predict)
    result[name] = [MAE, MSE, RMSE, MAPE, R2]

print('Test set')
print(pd.DataFrame(result, index=['MAE', 'MSE', 'RMSE', 'MAPE', 'R2']).transpose())


# #------------------------------------------
# 산점도 행렬 그리기
# import seaborn as sns
# import matplotlib.pyplot as plt
#
# train_selected = train[Features]
#
# sns.pairplot(train_selected)
# plt.show()
# #------------------------------------------