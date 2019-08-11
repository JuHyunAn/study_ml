'''
Created on 2017. 8. 24.

@author: jaehyeong
'''
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)

# SVC
svc = SVC()
svc.fit(X_train, y_train)

print('train set score : ', svc.score(X_train, y_train))    # 1.0
print('test set score : ',svc.score(X_test, y_test))        # 0.629
'''
svm은 매개변수 설정과 데이터 스케일에 매우 민감. 특히, 입력  특성의 범위가 비슷해야 함
각 특성의 최솟값과 최댓값에대한 로그스케일 시각화 
'''
# 로그스케일 시각화
plt.boxplot(X_train, manage_xticks=False)
plt.yscale('symlog')
plt.xlabel('feature list')
plt.ylabel('feature size')
plt.show()
'''
-> 유방암 데이터셋의 특성은 자릿수 자체가 완전히 달라, 영향을 많이 줌.
이를 위한 해결법으로, 특성 값의 범위가 비슷해도록 조정(0 ~ 1사이로) -> MinMaxScaler()적용
'''
from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# scale된 데이터를 통해 다시 학습
svc = SVC()
svc.fit(X_train_scaled, y_train)

print('train set score : ', svc.score(X_train_scaled, y_train)) # 0.948 -> 과소적합!
print('test set score : ', svc.score(X_test_scaled, y_test))    # 0.951

# C 혹은 gamma 값을 증가시켜 모델을 복잡하게
svc = SVC(C=1000)
svc.fit(X_train_scaled, y_train)

print('train set score : ', svc.score(X_train_scaled, y_train)) # 0.988
print('test set score : ', svc.score(X_test_scaled, y_test))    # 0.972
