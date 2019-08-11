'''
Created on 2017. 8. 6.

@author: jaehyeong
'''

import numpy as np
from sklearn.datasets import load_iris
iris_dataset = load_iris()

print("iris_dataset의 키 : \n{}" .format(iris_dataset.keys()))    # dict_keys(['data', 'target', 'target_names', 'DESCR', 'feature_names'])
print("타킷의 이름 : ", iris_dataset['target_names'])    # ['setosa' 'versicolor' 'virginica']

print('특성의 이름 : ', iris_dataset['feature_names'])   # ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
print('data의 크기 : ', iris_dataset['data'].shape)    # (150, 4)
print('target의 크기 : ', iris_dataset['target'].shape)    # (150,)

# train set(75%) & test set(25%)
# x : data, y : label
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'],
                                                    random_state = 0)
print('X_train의 크기 : ',X_train.shape)   # (112, 4)
print('y_train의 크기 : ',y_train.shape)    # (112,)

print('X_test의 크기 : ', X_test.shape)    # (38, 4)
print('y_test의 크기 : ',y_test.shape)     # (38,))

# K-Nearest Neighbors 
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1) 
# 훈련 데이터셋으로 모델 생성
knn.fit(X_train, y_train)
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                     metric_params=None, n_jobs=1, n_neighbors=1, p=2, weights='uniform')

# 예측하려는 새로운 값
X_new = np.array([[5, 2.9, 1, 0.2]])

# Predict
prediction = knn.predict(X_new)
print('예측 : ', prediction)      # [0]
print('예측한 target의 이름 : ', iris_dataset['target_names'][prediction])    # ['setosa']

# Model 평가(정확도 계산)
y_pred = knn.predict(X_test)    # test데이터셋을 예측
print('test 셋에 대한 예측 값 : ', y_pred)

print('test 셋의 정확도 : ',np.mean(y_pred == y_test))   # 0.973684210526
print('test 셋의 정확도 : ', knn.score(X_test, y_test))  # 0.973684210526