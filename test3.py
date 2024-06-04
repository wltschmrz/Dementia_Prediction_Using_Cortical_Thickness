import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 400)

# 데이터 불러오기
file_path = 'corticalthickness.csv'  # 파일 경로
data = pd.read_csv(file_path)

# 필요한 Features와 Target 선택
cortical_thickness = [col for col in data.columns if col.startswith('l_') or col.startswith('r_')]
selected_features = ['age', 'sex', 'edu', 'apos', 'icv'] + ['napoe4'] + cortical_thickness

X = data[selected_features]
y = data['mmse']

# napoe4 더미코딩
X = pd.get_dummies(X, columns=['napoe4'], prefix='napoe4')
X = X.drop('napoe4_2', axis=1)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)

# Feature selection - Filter Method (SelectKBest)
# (사용 안할 경우 X_train_selected 대신, 기존 X_train 이용. ex: Regression Tree)
selector = SelectKBest(score_func=f_regression, k=7)   # 68개의 cortical thickness values 중에서 10개의 특성만 선택
ct_X_train = X_train[cortical_thickness]
selector.fit_transform(ct_X_train, y_train)
mask = selector.get_support()                           # 선택된 cortical thickness Features' index
selected_ct_columns = ct_X_train.columns[mask]

X_train_selected = X_train[['age', 'sex', 'edu', 'apos', 'icv'] + ['napoe4_0', 'napoe4_1'] + selected_ct_columns.tolist()]
X_test_selected = X_test[['age', 'sex', 'edu', 'apos', 'icv'] + ['napoe4_0', 'napoe4_1'] + selected_ct_columns.tolist()]



# 모델 학습 및 평가 function
def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_predict)
    mse = mean_squared_error(y_test, y_predict)
    rmse = mean_squared_error(y_test, y_predict, squared=False)
    mape = mean_absolute_percentage_error(y_test, y_predict)
    r2 = r2_score(y_test, y_predict)
    return mae, mse, rmse, mape, r2

# 그래프 작도 function
def plt_drawing(df, model, para):
    colors = ['dodgerblue' if e != min(df['MSE']) else 'r' for e in df['MSE']]
    plt.bar(df.index, df['MSE'], color=colors)
    names = []; k=0
    for i in range(len(df['MSE'])):
        if df['MSE'][i] == min(df['MSE']):
            if k == 0:
                names.append(para[i])
                k = k+1
            else: names.append('')
        else: names.append('')
    plt.xticks(range(len(df.index)), labels=names)
    plt.xlabel(model)
    plt.ylabel('MSE')
    plt.ylim(5,9.)
    plt.title(model)
    plt.show()



# # 회귀 모델 틀 생성
# models = {
#     'Regression tree': {},
#     'Kernelized support vector regression': {},
# }
#
# # 'Generalized addictive model': []
#
#
# # Regression tree
# for max_depth in [3, 7, 11]:
#     for min_samples_split in [3, 7, 11]:
#         decision_tree_regressor = DecisionTreeRegressor(max_depth=max_depth,
#                                                         min_samples_split=min_samples_split,
#                                                         min_samples_leaf=1,
#                                                         max_leaf_nodes=None,
#                                                         ccp_alpha=0.0,
#                                                         random_state=0)
#         models['Regression tree'][f'max_depth:{max_depth}/min_samples_split:{min_samples_split}'] = decision_tree_regressor
#
# # Kernelized surport vector regression
# for kernel in ['rbf', 'sigmoid']:
#     for gamma in [1, 0.1, 0.01]:
#         for c in [100, 1]:
#             models['Kernelized support vector regression'][f'kernel:{kernel}/gamma:{gamma}/C:{c}'] = SVR(kernel=kernel,
#                                                                                                          gamma=gamma,
#                                                                                                          C=c,
#                                                                                                          epsilon=0.1)
# Running Time이 너무 길어서 잠시 제외
'''
for gamma in [1, 0.1, 0.01]:
    models['Kernelized support vector regression'][f'poly_{gamma}'] = SVR(kernel='poly',
                                                                          gamma=gamma,
                                                                          C=10,
                                                                          epsilon=0.1,
                                                                          degree=2)
'''



# Generalized addictive model
from pygam import GAM, s, te
# best_model = GAM(s(0, n_splines=200) + te(3,1) + s(2), distribution='normal',link='identity').gridsearch(X_train, y_train) # 모델의 최적의 매개변수 찾기
# best_params = best_model.best_params_
# print(best_params)

print(len(X_train_selected.columns.tolist()))

print(X_train_selected.columns.tolist())

# for i in
terms = s(0) + s(1) + s(2) + s(3) + s(4) \
        + te(5,6) \
        + s(7) + s(8) + s(9) + s(10) + s(11) + s(12) + s(13)
model = GAM(terms=terms)
a,b,c,d,e = evaluate_model(model, X_train_selected, X_test_selected, y_train, y_test)
print(a,b,c,d,e)

# # 각 모델에 대한 평가 수행 & 그래프 작도에 필요한 정보 정리
# drawing_info = {}
# for model_type, model_list in models.items():
#     print(f'{model_type}:') # str
#     result = {}
#     for parameters, model in model_list.items():
#         mae, mse, rmse, mape, r2 = evaluate_model(model,
#                                                   # 모델에 따라 다른 Feature selection 수행
#                                                   X_train if model_type == 'Regression tree' else X_train_selected,
#                                                   X_test if model_type == 'Regression tree' else X_test_selected,
#                                                   y_train,
#                                                   y_test)
#         result[parameters] = [mae, mse, rmse, mape, r2]
#     result_data = pd.DataFrame(result, index=['MAE', 'MSE', 'RMSE', 'MAPE', 'R2']).transpose() # df
#     print(result_data, end='\n'*2)  # result 표 출력
#
#     parameter = list(result.keys()) # list
#     drawing_info[model_type] = [result_data, model_type, parameter]
#
# # 그래프 작도
# for vv in drawing_info.values():
#     plt_drawing(vv[0], vv[1], vv[2])