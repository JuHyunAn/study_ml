'''
Created on 2017. 8. 15.

@author: jaehyeong
'''
'''
RandomForest,
-> 각 트리는 비교적 예측을 잘 하지만 데이터의 일부에 과대적합한다는 것에 기초를 두고,
서로 다른 방향으로 과대적합된 트리를 많이 만들면 그 결과를 평균냄으로써 과대적합된 양을 줄인다.
'''
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)

# randomforest 객체 생성
forest = RandomForestClassifier(n_estimators=100,   # 100개의 트리(더 많은 트리를 평균할 수록 과대적합을 줄여주기 때문에, 값이 클 수 록 좋음
                                random_state=0,     # randomforest는 이름 그대로 랜덤하므로, random_state 값을 고정하는 것이 좋음 
                                n_jobs=-1)          # 사용할 코어의 수, -1은 컴퓨터의 모든 코어를 사용
forest.fit(X_train, y_train)

print('train set score : ',forest.score(X_train, y_train))  # 1.0
print('test set score : ',forest.score(X_test, y_test))     # 0.972


# 특성 중요도 파악
#from method.definition import plot_feature_importances_cancer
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
import numpy as np
cancer = load_breast_cancer()

def plot_feature_importances_cancer(model):
    n_features = cancer.data.shape[1]
    print('n_features\n',n_features)
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), cancer.feature_names)
    plt.xlabel('feature importance')
    plt.ylabel('features')
    plt.ylim(-1, n_features)
    plt.show()

plot_feature_importances_cancer(forest)
