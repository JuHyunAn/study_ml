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
