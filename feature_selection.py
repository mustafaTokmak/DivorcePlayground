from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm

import matplotlib.pyplot as plt
from sklearn import  metrics

from sklearn import feature_selection


data = pd.read_csv("divorce.csv",dtype=int)


results = data['Class']
answers = data.drop(columns=['Class'],axis=1)#.to_numpy()

corr_matrix = data.corr().abs()
print(corr_matrix['Class'].sort_values(ascending=False).head(10))



initial_feature_names = list(answers.columns)

pca = PCA(n_components=3)
pca_fit = pca.fit(answers)
principal_components = pca_fit.transform(answers)
answers = pd.DataFrame(data = principal_components)
print(answers.shape)

n_pcs= pca_fit.components_.shape[0]
print(n_pcs)


most_important = [np.abs(pca_fit.components_[i]).argmax() for i in range(n_pcs)]
print(most_important)
print((initial_feature_names))
most_important_names = [ ]

for i in range(n_pcs):
    most_important_names.append(initial_feature_names[most_important[i]])
print(most_important_names)

"""
pca = PCA(n_components=2)
pca.fit(answers)
print(pca.score(answers))
print(pca.explained_variance_ratio_)
"""