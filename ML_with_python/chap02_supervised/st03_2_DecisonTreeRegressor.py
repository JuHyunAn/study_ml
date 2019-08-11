'''
Created on 2017. 8. 15.

@author: jaehyeong

DecisionTreeRegressor와 LinearRegression의 비교
'''
import mglearn, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ram_prices = pd.read_csv(os.path.join(mglearn.datasets.DATA_PATH, "ram_price.csv"))
'''
plt.semilogy(ram_prices.date, ram_prices.price)
plt.xlabel('year')
plt.ylabel('price($/Mbyte_')
plt.show()
'''

from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
# 2000년 이전을 train data, 2000년 이후를 test data
data_train = ram_prices[ram_prices.date < 2000]
data_test = ram_prices[ram_prices.date >= 2000]

# 가격 예측을 위해 날짜 특성만 이용
X_train = data_train.date[:,np.newaxis]
# 데이터와 타깃의 관계를 간단히 하기 위해 로그 스케일로 변환
y_train = np.log(data_train.price)

tree = DecisionTreeRegressor().fit(X_train, y_train)
linear = LinearRegression().fit(X_train, y_train)

# 예측은 전체 기간에 대해 수행
X_all = ram_prices.date[:,np.newaxis]

pred_tree = tree.predict(X_all)
pred_lr = linear.predict(X_all)

# 예측한 값의 로그 스케일을 돌려준다
price_tree = np.exp(pred_tree)
price_lr = np.exp(pred_lr)

plt.semilogy(data_train.date, data_train.price, label='Train Data')
plt.semilogy(data_test.date, data_test.price, label="Test Data")
plt.semilogy(ram_prices.date, price_tree, label='Tree Prediction')
plt.semilogy(ram_prices.date, price_lr, label='Linear Prediction')
plt.legend()
plt.show()
'''
-> 트리 모델은 훈련데이터를 완벽하게 예측하지만, 새로운 데이터를 예측하는 능력은 떨어짐(즉, 훈련데이터에 과대적합한다)
이러한 문제를 해결하기 위한 앙상블(ensemble)기법으로 랜덤포레스트, 그래디언트 부스팅이 사용된다.
'''