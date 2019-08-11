
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

X = cancer.data
y = cancer.target

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
sc.fit(X)
X_sc = sc.transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_sc, y, random_state=0)

from sklearn.svm import SVC
svm = SVC()
svm.fit(X_train, y_train)

print('test set score :',svm.score(X_test, y_test))


from sklearn.model_selection import GridSearchCV
param_grid = {'C':[0.001, 0.01, 0.1, 1, 10, 100],
			  'gamma':[0.001, 0.01, 0.1, 1, 10, 100]}

grid = GridSearchCV(SVC(),param_grid=param_grid, cv=5)
grid.fit(X_train, y_train)
print('best cv score :',grid.best_score_)
print('test set score :',grid.score(X_test, y_test))
print('best parameters :',grid.best_params_)
