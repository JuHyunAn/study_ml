'''
Created on 2017. 8. 15.

@author: jaehyeong
'''
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


# dataset
cancer = load_breast_cancer()
# split train, test set
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, 
                                                    stratify=cancer.target, random_state=42)
# Decision tree 객체 생성
tree = DecisionTreeClassifier(random_state=0)
# train model
tree.fit(X_train, y_train)
# score
print('trainset score : ',tree.score(X_train, y_train)) # 1.0  -> 과대적합!
print('testset score : ', tree.score(X_test, y_test))   # 0.937

# 결정 트리의 깊이 제한하여 과대적합 방지하기
# 훈련세트의 정확도는 떨어지나 테스트 세트의 성능을 개선시킬 수 있음
tree = DecisionTreeClassifier(max_depth=4, random_state=0)  # 질문길이를 4개로 제한
tree.fit(X_train, y_train)
print('trainset score : ',tree.score(X_train, y_train)) # 0.988
print('testset score : ', tree.score(X_test, y_test))   # 0.951

# 결정트리 시각화
#Graphviz2.38깔고 해당폴더의 dot.exe를 시스템 변수에 추가
from sklearn.tree import export_graphviz
import graphviz

export_graphviz(tree, out_file="tree.dot", class_names=["악성","양성"],     # .dot파일로 저장
                feature_names=cancer.feature_names, impurity=False, filled=True)    # filled=True : 색으로 노드 클래스 구분

with open("tree.dot", encoding='utf-8') as f:
    dot_graph = f.read()
dot = graphviz.Source(dot_graph,encoding='utf-8')
dot.format='svg'
dot.render(filename='tree', view=True)


# 특성 중요도 파악(각 특성이 얼마나 중요한지)
# 0~1사이의 값, 0 : 전혀 사용되지 않음, 1 : 완벽하게 타겟 클래스 예측
print("특성 중요도 : ",tree.feature_importances_)
# 특성 중요도 시각화
from method.definition import plot_feature_importances_cancer
plot_feature_importances_cancer(tree)
