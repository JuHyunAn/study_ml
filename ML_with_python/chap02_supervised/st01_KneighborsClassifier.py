'''
Created on 2017. 8. 6.

@author: jaehyeong
'''
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors.classification import KNeighborsClassifier
import matplotlib.pyplot as plt
import mglearn

cancer = load_breast_cancer()
#print(cancer.data)            # X : data
#print(cancer.target_names)    # y : label
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target,
                                                    stratify = cancer.target, random_state=66)

training_accuracy = []
test_accuracy = []

# 1 ~ 10까지 n_neighbors를 적용
neighbors_setting = range(1,11)

for n_neighbors in neighbors_setting:
    # 모델 생성
    clf = KNeighborsClassifier(n_neighbors= n_neighbors)
    clf.fit(X_train, y_train)
    # train 세트 정확도 저장
    training_accuracy.append(clf.score(X_train, y_train))
    # 일반화 정확도 저장
    test_accuracy.append(clf.score(X_test, y_test))

plt.plot(neighbors_setting, training_accuracy, label='traing accuracy')
plt.plot(neighbors_setting, test_accuracy, label='test accuracy')
plt.ylabel('Accuracy')
plt.xlabel('n_neighbors')
plt.legend()
plt.show()
