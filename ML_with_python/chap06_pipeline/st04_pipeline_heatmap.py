import mglearn
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, GridSearchCV

boston = load_boston()
X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target,
														random_state=0)
# make pipeline
pipe = make_pipeline(StandardScaler(),
					 PolynomialFeatures(),
					 Ridge())

param_grid = {'polynomialfeatures__degree':[1,2,3],
			  'ridge__alpha':[0.001, 0.01, 0.1, 1, 10, 100]}

# grid search
grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(X_train, y_train)

# cv_score heatmap
mglearn.tools.heatmap(grid.cv_results_['mean_test_score'].reshape(3,-1),
						xlabel='ridge__alpha', ylabel='polynomialfeatures__degree',
						xticklabels=param_grid['ridge__alpha'],
						yticklabels=param_grid['polynomialfeatures__degree'],
						vmin=0)
plt.show()

print('최적의 매개변수 :',grid.best_params_)
print('test set score :',grid.score(X_test, y_test))