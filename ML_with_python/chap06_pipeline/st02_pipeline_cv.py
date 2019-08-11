from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Pipeline
pipe = Pipeline([("scaler", MinMaxScaler()),("svm",SVC())])
pipe.fit(X_train, y_train)
print('test set score : ', pipe.score(X_test, y_test))

# Grid Search에 pipeline 적용
param_grid = {'svm__C' : [0.001, 0.01, 0.1, 1, 10, 100],
			  'svm__gamma' : [0.001, 0.01, 0.1, 1, 10, 100]}

grid = GridSearchCV(pipe, param_grid=param_grid, cv=5)
grid.fit(X_train, y_train)
print('best cv score : ',grid.best_score_)
print('test set score : ',grid.score(X_test, y_test))
print('best parameters : ',grid.best_params_)
