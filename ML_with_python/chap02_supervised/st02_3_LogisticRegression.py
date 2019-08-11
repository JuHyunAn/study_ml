'''
Created on 2017. 8. 8.

@author: User

## Logistic Regression ##
'''
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# dataset
cancer = load_breast_cancer()

# split train, test data
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)

# model adaption
# C = 1(default)
logreg = LogisticRegression().fit(X_train, y_train)     # training
print("훈련 세트 score : ", logreg.score(X_train, y_train))  # 0.955
print("테스트 세트 score : ", logreg.score(X_test, y_test))  # 0.958

# C = 100
logreg100 = LogisticRegression(C=100).fit(X_train, y_train)
print("훈련 세트 score : ", logreg100.score(X_train, y_train))  # 0.972
print("테스트 세트 score : ", logreg100.score(X_test, y_test))   # 0.965
 
# C = 0.1
logreg001 = LogisticRegression(C=0.01).fit(X_train, y_train)
print("훈련 세트 score : ", logreg001.score(X_train, y_train))  # 0.934
print("테스트 세트 score : ", logreg001.score(X_test, y_test))   # 0.930


# visualization
plt.plot(logreg.coef_.T, 'o', label="C=1")
plt.plot(logreg100.coef_.T, '^', label="C=100")
plt.plot(logreg001.coef_.T, 'v', label="C=0.01")
plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)
plt.hlines(0, 0, cancer.data.shape[1])
plt.ylim(-5,5)
plt.xlabel("feature")
plt.ylabel("w size")
plt.legend()
plt.show()
