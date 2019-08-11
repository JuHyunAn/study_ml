'''
Created on 2017. 8. 15.

@author: jaehyeong
'''
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)

#GradientBoostingClassifier
gbc = GradientBoostingClassifier(random_state=0)
gbc.fit(X_train, y_train)

print('train set score : ', gbc.score(X_train, y_train))    # 1.0 -> 과대적합
print('test set score : ', gbc.score(X_test, y_test))       # 0.958

# 과대적합을 막기 위해 
# 최대 깊이(max_depth)를 줄여 가전 가지치기 / 학습률(learning_rate) 낮추기
gbc1 = GradientBoostingClassifier(random_state=0, max_depth=1)
gbc1.fit(X_train, y_train)
gbc2 = GradientBoostingClassifier(random_state=0, learning_rate=0.01)
gbc2.fit(X_train, y_train)

print('-- max_depth=1 --')
print('train set score : ', gbc1.score(X_train, y_train))    # 0.990
print('test set score : ', gbc1.score(X_test, y_test))       # 0.972
print('-- learning_rate=0.01 --')
print('train set score : ', gbc2.score(X_train, y_train))    # 0.988
print('test set score : ', gbc2.score(X_test, y_test))       # 0.965

# 특성 중요도 시각화
from method.definition import plot_feature_importances_cancer
plot_feature_importances_cancer(gbc1)