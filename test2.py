import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 100)

# 데이터 불러오기
file_path = 'corticalthickness.csv'  # 파일 경로
data = pd.read_csv(file_path)

# 필요한 Features와 Target 선택
cortical_thickness = [col for col in data.columns if col.startswith('l_') or col.startswith('r_')]
selected_features = ['age', 'sex', 'edu', 'apos', 'icv'] + ['napoe4'] + cortical_thickness

X = data[selected_features] #df
y = data['mmse'] #df

# napoe4 더미코딩
X = pd.get_dummies(X, columns=['napoe4'], prefix='napoe4')
X = X.drop('napoe4_2', axis=1)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=18)

# Feature selection - Filter Method (SelectKBest)
selector = SelectKBest(score_func=f_regression, k=10)  # 68개의 cortical thickness values 중에서 10개의 특성만 선택
ct_X_train = X_train[cortical_thickness]
selector.fit_transform(ct_X_train, y_train)
mask = selector.get_support()  # 선택된 cortical thickness Features index
selected_ct_columns = ct_X_train.columns[mask]

X_train_selected = X_train[['age', 'sex', 'edu', 'apos', 'icv'] + ['napoe4_0', 'napoe4_1'] + selected_ct_columns.tolist()]
X_test_selected = X_test[['age', 'sex', 'edu', 'apos', 'icv'] + ['napoe4_0', 'napoe4_1'] + selected_ct_columns.tolist()]


# 모델 학습 및 평가 function define
def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_predict)
    mse = mean_squared_error(y_test, y_predict)
    rmse = mean_squared_error(y_test, y_predict, squared=False)
    mape = mean_absolute_percentage_error(y_test, y_predict)
    r2 = r2_score(y_test, y_predict)
    return mae, mse, rmse, mape, r2

# # 회귀 모델 생성
# models = {
#     'Regression tree': {},
#     'Kernelized support vector regression': {},
# }
#
# # 'Generalized addictive model': []
#
#
# # Regression tree
# for max_depth in [3, 5, 7, 9, 11]:
#     decision_tree_regressor = DecisionTreeRegressor(max_depth=max_depth,
#                                                     min_samples_split=2,
#                                                     min_samples_leaf=1,
#                                                     max_leaf_nodes=None,
#                                                     ccp_alpha=0.0,
#                                                     random_state=0)
#     models['Regression tree'][f'max_depth={max_depth}'] = decision_tree_regressor
#
# # Kernelized surport vector regression
# for kernel in ['rbf', 'sigmoid']:
#     for gamma in ['scale', 1, 0.1, 0.01, 0.001, 0.0001]:
#         models['Kernelized support vector regression'][f'{kernel}_{gamma}'] = SVR(kernel=kernel,
#                                                                                   gamma=gamma,
#                                                                                   C=1.0,
#                                                                                   epsilon=0.1)
# # for gamma in [1, 0.1, 0.01, 0.001, 0.0001]:
# #     models['Kernelized support vector regression'][f'poly_{gamma}'] = SVR(kernel='poly',
# #                                                                           gamma=gamma,
# #                                                                           C=1.0,
# #                                                                           epsilon=0.1,
# #                                                                           degree=2)
#
# Generalized addictive model
from pygam import LinearGAM
LinearGAM().gridsearch(X_train, y_train) # 모델의 최적의 매개변수 찾기
#
# # 각 모델에 대한 평가 수행
# for model_type, list in models.items():
#     print(f'{model_type}:')
#     result = {}
#     for parameters, model in list.items():
#         mae, mse, rmse, mape, r2 = evaluate_model(model, X_train_selected, X_test_selected, y_train, y_test)
#         result[parameters] = [mae, mse, rmse, mape, r2]
#     print(pd.DataFrame(result, index=['MAE', 'MSE', 'RMSE', 'MAPE', 'R2']).transpose())


# from sklearn.model_selection import GridSearchCV
#
# tree_param_grid = {
#     'max_depth': [3,4,5,6,7,8,9,10],
#     'min_samples_split': [3,4,5,6,7,8,9,10]
# }
#
# tree = DecisionTreeRegressor()
# tree_grid_search = GridSearchCV(tree, tree_param_grid, cv=5, scoring='neg_mean_squared_error')
# tree_grid_search.fit(X_train, y_train)
# print("Best parameters for Decision Tree:", tree_grid_search.best_params_)
#
#
#
#
# svr_param_grid = {
#     'kernel': ['rbf', 'sigmoid'],
#     'gamma': [0.001,0.01,0.1, 1, 10],
#     'C': [0.1, 1, 10]
# }
#
# svr = SVR()
# svr_grid_search = GridSearchCV(svr, svr_param_grid, cv=5, scoring='neg_mean_squared_error')
# svr_grid_search.fit(X_train, y_train)
#
# # 최적의 파라미터 출력
# print("Best parameters for SVR:", svr_grid_search.best_params_)

