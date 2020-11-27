import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm

import matplotlib.pyplot as plt
from sklearn import  metrics
from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.neighbors import NearestCentroid

from sklearn import tree

from sklearn.linear_model import SGDClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn import linear_model

from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import SelectFromModel


data = pd.read_csv("divorce.csv",dtype=int)


results = data['Class']
answers = data.drop(columns=['Class'],axis=1)#.to_numpy()
#answers = answers[['Atr2','Atr6','Atr11','Atr18','Atr26','Atr40']] # paper features
#answers = answers[['Atr17','Atr16','Atr19','Atr26','Atr11']]
#answers = answers[['Atr40','Atr17','Atr19','Atr18','Atr11','Atr9']] # paper features
answers = answers[['Atr17']]

initial_feature_names = list(answers.columns)
"""
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
#print(most_important_names)

#answers = answers[['Atr18','Atr19']]
#answers = answers[['Atr2']]

feature_one_negatives = []
feature_one_positives = []
"""
result = {}
for index,row in answers.iterrows():
    r = (int(results.loc[index]))
    a = (int(answers.loc[index]))
    if not r in result:
        result[r] = {}
    if not a in result[r]:
        result[r][a] = 0
    result[r][a] += 1
    
print(result)
"""

X_train, X_test, y_train, y_test = train_test_split(answers, results, test_size=0.2)
clf = svm.SVC()
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
clf = GridSearchCV(clf, tuned_parameters)

##new models
#clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
#clf = AdaBoostClassifier(n_estimators=100)
#clf = NearestCentroid()

clf.fit(X_train, y_train)
"""
model = SelectFromModel(clf, prefit=False)
answers_new = model.transform(answers)
print(answers_new.shape)

"""

"""
score = clf.decision_function(X_test)
fpr, tpr, thresholds = metrics.roc_curve(y_test, score,drop_intermediate=False)
plt.plot(fpr,tpr)
"""


print(clf.best_params_)




y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)


print(accuracy)
scores = cross_val_score(clf, answers, results, cv=17)
print(scores)
print("Accuracy: %0.7f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
plt.show()


