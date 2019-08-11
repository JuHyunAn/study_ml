'''
Created on 2017. 8. 25.

@author: jaehyeong

MLP의 매개변수 조정 방법
-> 먼저 충분히 훈련 데이터가 학습될 수 있도록 과대적합 된 큰 모델을 만든 후,
신경망의 구조를 줄이거나 규제 강화를 위해 alpha값을 증가시켜 일반화 성능을 향상
'''
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)

from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(random_state=42)
mlp.fit(X_train, y_train)
print('train set score : ', mlp.score(X_train, y_train))    # 0.906
print('test set score : ', mlp.score(X_test, y_test))       # 0.881

# 신경망도 모든 입력 특성을 평균은 0, 분산은 1인 표준정규분포로 변형하는 것이 좋음.
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

mlp = MLPClassifier(max_iter=1000, random_state=0)     # 최대 반복 횟수에 대한 경고 나오면 -> 반복 횟수(max_iter)를 늘려줘야 함

mlp.fit(X_train_scaled, y_train)
print('train set score : ',mlp.score(X_train_scaled,  y_train)) # 0.992 -> 과대적합!
print('test set score : ',mlp.score(X_test_scaled, y_test))     # 0.972

# 규제(alpha)를 높여 모델의 복잡도를 낮춤
mlp = MLPClassifier(max_iter=1000, alpha=1, random_state=42)     # alpha의 기본값은 0.0001
mlp.fit(X_train_scaled, y_train)
print('train set score : ',mlp.score(X_train_scaled,  y_train)) # 0.988
print('test set score : ',mlp.score(X_test_scaled, y_test))     # 0.979

# 신경망의 첫 번째 가중치 히트맵 시긱화 
import matplotlib.pyplot as plt
plt.figure(figsize=(20,5))
plt.imshow(mlp.coefs_[0], interpolation='none', cmap='viridis') # coefs_[0]: 입력과 은닉층 사이의 가중치, coefs_[1]: 은닉층과 출력 사이의 가중치
plt.yticks(range(30), cancer.feature_names)
plt.xlabel('hidden units')
plt.ylabel('input features')
plt.colorbar(); plt.show()
