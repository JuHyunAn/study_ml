'''
Created on 2017. 8. 8.

@author: User

## Logistic regression, LinearSVC ##
'''
import mglearn
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

# datasets
X, y = mglearn.datasets.make_forge()

fig, axes = plt.subplots(1,2, figsize=(10,3))    # 1row 2colum plot

# model
for model, ax in zip([LinearSVC(), LogisticRegression()], axes):
    clf = model.fit(X, y)
    mglearn.plots.plot_2d_separator(clf, X, fill=False, eps=0.5,
                                    ax=ax, alpha=.7)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
    ax.set_title("{}".format(clf.__class__.__name__))
    ax.set_xlabel("feature 0")
    ax.set_ylabel("feature 1")
axes[0].legend()
plt.show()

# 규제의 강도를 결정하는 매개변수 : C
# C값이 높아지면, 규제가 감소
# 즉, 높은 C값을 지정하면 훈련세트에 가능한 최대로 맞추려하고, 낮은 C값은 w가 0에 가까워지도록 함
mglearn.plots.plot_linear_svc_regularization()
plt.show()