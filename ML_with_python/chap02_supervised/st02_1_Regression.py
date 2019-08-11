'''
Linear : mse를 최소화하는 w와 b를 찾음
			- 매개변수가 없어, 모델의 복잡도 제어X
Ridge : 선형회귀에 규제(L1)를 걸어, 모델의 과대적합을 방지
			- L2 : 가중치들의 제곱합을 최소화
Rasso : 릿지회귀와 비슷 , 특성선택이 자동으로 이루어짐(L1)
			- L1 : 가중치들의 절대값의 합을 최소화
ElasticNet : ridge + lasso 
'''
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# dataset
X, y = load_boston(True)

####### Linear Regression #######
from sklearn.linear_model import LinearRegression
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
lr = LinearRegression().fit(X_train, y_train)
print('----Linear Regression----')
print('훈련 세트 score : ',lr.score(X_train, y_train))  # 0.95
print('테스트 세트 score : ',lr.score(X_test, y_test))   # 0.60

train_pred = lr.predict(X_train)
test_pred = lr.predict(X_test)

print('MSE of train set: ', mean_squared_error(y_train, train_pred))
print('MSE of test set: ', mean_squared_error(y_test, test_pred))
####### Ridge Regression ####### 
# -> 과대적합이 되지 않도록 모델을 강제로 제한(= Regularization)
from sklearn.linear_model import Ridge
ridge = Ridge().fit(X_train, y_train)
print('----Ridge Regression----')
print('훈련 세트 score : ',ridge.score(X_train, y_train))   # 0.89
print('테스트 세트 score : ',ridge.score(X_test, y_test))    # 0.75

# alpha값 조정 -> alpha값을 높이면 계수를 0에 가깝게함 -> 최적의 alpha값을 찾아야 함
ridge10 = Ridge(alpha=10).fit(X_train, y_train)     # alpha값이 10일 때
print('훈련 세트 score : ',ridge10.score(X_train, y_train))     # 0.79
print('테스트 세트 score : ',ridge10.score(X_test, y_test))      # 0.64 

ridge01 = Ridge(alpha=0.1).fit(X_train, y_train)    # alpha값이 1일 때
print('훈련 세트 score : ',ridge01.score(X_train, y_train))     # 0.93
print('테스트 세트 score : ',ridge01.score(X_test, y_test))      # 0.77

####### Lasso Regression #######
from sklearn.linear_model import Lasso
lasso = Lasso().fit(X_train, y_train)
print('----Lasso Regression----')
print('훈련 세트 score : ',lasso.score(X_train, y_train))   # 0.29 -> 과소적합
print('테스트 세트 score : ',lasso.score(X_test, y_test))    # 0.20
print('사용한 특성의 수 : ',np.sum(lasso.coef_ != 0))          # 4 -> 105개의 특성 중 4개만 사용

# 과소적합을 줄이기 위해 alpha값(규제)을 줄임 
lasso001 = Lasso(alpha=0.01, max_iter=100000).fit(X_train, y_train)
print('훈련 세트 score : ',lasso001.score(X_train, y_train))   # 0.90
print('테스트 세트 score : ',lasso001.score(X_test, y_test))    # 0.77
print('사용한 특성의 수 : ',np.sum(lasso001.coef_ != 0))          # 33

####### ElasticNet #######
from sklearn.linear_model import ElasticNet
elastic = ElasticNet(alpha=0.001, max_iter=10000000).fit(X_train, y_train)
print('train score :',elastic.score(X_train, y_train))
print('test score :',elastic.score(X_test, y_test))