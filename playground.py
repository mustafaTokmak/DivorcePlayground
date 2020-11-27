import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm
from statistics import mean 


data = pd.read_csv("divorce.csv",dtype=int)


results = data['Class']
answers = data.drop(columns=['Class'],axis=1)#.to_numpy()
#answers = answers[['Atr2','Atr6','Atr11','Atr18','Atr26','Atr40']]
print(answers.columns)

impact = []
for i in answers.columns:#range(1,55):
    a = answers[i].to_numpy().reshape(-1,1)
    #answers = answers[['Atr2']]


    

    X_train, X_test, y_train, y_test = train_test_split(a, results, test_size=0.4)
    clf = svm.SVC(kernel='linear', C=1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)
    if accuracy == 1.0:
        print("BEST FEATURE: " + str(i))
    scores = cross_val_score(clf, a, results, cv=10,scoring='f1_macro')
    m = str(mean(scores))
    impact.append( m +" : "+str(i[:10]))
impact.sort()
print(impact)