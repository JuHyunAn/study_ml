from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
'''
# 표준적인 방법
pipe_long = Pipeline([('scaler',MinMaxScaler()),('svm',SVC(C=100))])
# 간소화된 방법
pipe_short = make_pipeline(MinMaxScaler(), SVC(C=100))
print('Pipeline process:\n', pipe_short.steps)

# 단계 속성에 접근하기
pipe = make_pipeline(MinMaxScaler(), PCA(n_components=2))
pipe.fit(cancer.data)
# pca 단계의 2개 주성분을 추출
components = pipe.named_steps["pca"].components_
print("components.shape :",components.shape)	# components.shape : (2, 30)
'''

# make pipeline
pipe = make_pipeline(StandardScaler(), LogisticRegression())
param_grid = {'logisticregression__C':[0.01, 0.1, 1, 10, 100]}

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target)

# grid search
grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(X_train, y_train)

print('최상의 모델:\n ',grid.best_estimator_)
print('\n로지스틱 회귀 단계: \n',grid.best_estimator_.named_steps['logisticregression'])
