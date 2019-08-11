'''
Created on 2017. 8. 6.

@author: jaehyeong
'''
import mglearn
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split

# dataset
X, y = mglearn.datasets.make_wave(n_samples=40)

# train & test
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

# k = 3, 객체생성
reg = KNeighborsRegressor(n_neighbors=3)
#  학습
reg.fit(X_train, y_train)

# test셋에 대한 예측
print('test set 예측 : ',reg.predict(X_test))
# score( R^2 : 결정계수 )
print('test set R^2 : ', reg.score(X_test, y_test))     # 0.834417244625
