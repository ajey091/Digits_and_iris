from sklearn import datasets
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
import time

iris = datasets.load_iris()
digits = datasets.load_digits()

clf = svm.SVC(gamma=0.001, C=100.,kernel='rbf')
print ("SVM classifier : ")

from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()
loo.get_n_splits(digits.data)

# digits = iris
X = digits.data
X = StandardScaler().fit_transform(X)
y = digits.target


# Digits dataset: LeaveOneOut
start = time.clock()
count = 0
for train_index, test_index in loo.split(X):
    X_train = digits.data[train_index]
    y_train = digits.target[train_index]
    clf.fit(X_train,y_train)
    val = clf.predict(digits.data[test_index])
    if val == digits.target[test_index]:
        count += 1

accuracy = count/len(digits.data)*100
print ("accuracy (LeaveOneOut) = ",accuracy)
print ("Time taken for LeaveOneOut = " ,time.clock() - start)

# Digits dataset: KFold (n=10)

start = time.clock()
count = 0
from sklearn.model_selection import KFold

kf = KFold(n_splits=10, shuffle=True)

for train_index, test_index in kf.split(X):
    X_train = digits.data[train_index]
    y_train = digits.target[train_index]
    clf.fit(X_train, y_train)
    val = clf.predict(digits.data[test_index])
    # print(val, digits.target[test_index])
    for i in range(len(val)):
        if val[i] == digits.target[test_index[i]]:
            count += 1

accuracy = count / len(digits.data) * 100
print("accuracy (KFold(n=10)) = ", accuracy)
print ("Time taken for Kfold(n=10) = " ,time.clock() - start)


# Digits dataset: KFold (n=50)

start = time.clock()
count = 0
from sklearn.model_selection import KFold

kf = KFold(n_splits=50, shuffle=True)

for train_index, test_index in kf.split(X):
    X_train = digits.data[train_index]
    y_train = digits.target[train_index]
    clf.fit(X_train, y_train)
    val = clf.predict(digits.data[test_index])
    # print(val, digits.target[test_index])
    for i in range(len(val)):
        if val[i] == digits.target[test_index[i]]:
            count += 1

accuracy = count / len(digits.data) * 100
print("accuracy (KFold(n=50)) = ", accuracy)
print ("Time taken for Kfold(n=50) = " ,time.clock() - start)


# Iris dataset: LeaveOneOut

X = iris.data
X = StandardScaler().fit_transform(X)
y = iris.target


start = time.clock()
count = 0
for train_index, test_index in loo.split(X):
    X_train = iris.data[train_index]
    y_train = iris.target[train_index]
    clf.fit(X_train,y_train)
    val = clf.predict(iris.data[test_index])
    if val == iris.target[test_index]:
        count += 1

accuracy = count/len(iris.data)*100
print ("accuracy (LeaveOneOut) = ",accuracy)
print ("Time taken for LeaveOneOut = " ,time.clock() - start)

# Iris dataset: KFold (n=10)

start = time.clock()
count = 0
from sklearn.model_selection import KFold

kf = KFold(n_splits=10, shuffle=True)

for train_index, test_index in kf.split(X):
    X_train = iris.data[train_index]
    y_train = iris.target[train_index]
    clf.fit(X_train, y_train)
    val = clf.predict(iris.data[test_index])
    # print(val, iris.target[test_index])
    for i in range(len(val)):
        if val[i] == iris.target[test_index[i]]:
            count += 1

accuracy = count / len(iris.data) * 100
print("accuracy (KFold(n=10)) = ", accuracy)
print ("Time taken for Kfold(n=10) = " ,time.clock() - start)

# Iris dataset: KFold (n=50)

start = time.clock()
count = 0
from sklearn.model_selection import KFold

kf = KFold(n_splits=50, shuffle=True)

for train_index, test_index in kf.split(X):
    X_train = iris.data[train_index]
    y_train = iris.target[train_index]
    clf.fit(X_train, y_train)
    val = clf.predict(iris.data[test_index])
    # print(val, iris.target[test_index])
    for i in range(len(val)):
        if val[i] == iris.target[test_index[i]]:
            count += 1

accuracy = count / len(iris.data) * 100
print("accuracy (KFold(n=50)) = ", accuracy)
print ("Time taken for Kfold(n=50) = " ,time.clock() - start)



############################################################################################################################
clf =  KNeighborsClassifier(4)
print ("KNN classifier : ")

from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()
loo.get_n_splits(digits.data)

# digits = iris
X = digits.data
X = StandardScaler().fit_transform(X)
y = digits.target


# Digits dataset: LeaveOneOut
start = time.clock()
count = 0
for train_index, test_index in loo.split(X):
    X_train = digits.data[train_index]
    y_train = digits.target[train_index]
    clf.fit(X_train,y_train)
    val = clf.predict(digits.data[test_index])
    if val == digits.target[test_index]:
        count += 1

accuracy = count/len(digits.data)*100
print ("accuracy (LeaveOneOut) = ",accuracy)
print ("Time taken for LeaveOneOut = " ,time.clock() - start)

# Digits dataset: KFold (n=10)

start = time.clock()
count = 0
from sklearn.model_selection import KFold

kf = KFold(n_splits=10, shuffle=True)

for train_index, test_index in kf.split(X):
    X_train = digits.data[train_index]
    y_train = digits.target[train_index]
    clf.fit(X_train, y_train)
    val = clf.predict(digits.data[test_index])
    # print(val, digits.target[test_index])
    for i in range(len(val)):
        if val[i] == digits.target[test_index[i]]:
            count += 1

accuracy = count / len(digits.data) * 100
print("accuracy (KFold(n=10)) = ", accuracy)
print ("Time taken for Kfold(n=10) = " ,time.clock() - start)


# Digits dataset: KFold (n=50)

start = time.clock()
count = 0
from sklearn.model_selection import KFold

kf = KFold(n_splits=50, shuffle=True)

for train_index, test_index in kf.split(X):
    X_train = digits.data[train_index]
    y_train = digits.target[train_index]
    clf.fit(X_train, y_train)
    val = clf.predict(digits.data[test_index])
    # print(val, digits.target[test_index])
    for i in range(len(val)):
        if val[i] == digits.target[test_index[i]]:
            count += 1

accuracy = count / len(digits.data) * 100
print("accuracy (KFold(n=50)) = ", accuracy)
print ("Time taken for Kfold(n=50) = " ,time.clock() - start)


# Iris dataset: LeaveOneOut

X = iris.data
X = StandardScaler().fit_transform(X)
y = iris.target


start = time.clock()
count = 0
for train_index, test_index in loo.split(X):
    X_train = iris.data[train_index]
    y_train = iris.target[train_index]
    clf.fit(X_train,y_train)
    val = clf.predict(iris.data[test_index])
    if val == iris.target[test_index]:
        count += 1

accuracy = count/len(iris.data)*100
print ("accuracy (LeaveOneOut) = ",accuracy)
print ("Time taken for LeaveOneOut = " ,time.clock() - start)

# Iris dataset: KFold (n=10)

start = time.clock()
count = 0
from sklearn.model_selection import KFold

kf = KFold(n_splits=10, shuffle=True)

for train_index, test_index in kf.split(X):
    X_train = iris.data[train_index]
    y_train = iris.target[train_index]
    clf.fit(X_train, y_train)
    val = clf.predict(iris.data[test_index])
    # print(val, iris.target[test_index])
    for i in range(len(val)):
        if val[i] == iris.target[test_index[i]]:
            count += 1

accuracy = count / len(iris.data) * 100
print("accuracy (KFold(n=10)) = ", accuracy)
print ("Time taken for Kfold(n=10) = " ,time.clock() - start)

# Iris dataset: KFold (n=50)

start = time.clock()
count = 0
from sklearn.model_selection import KFold

kf = KFold(n_splits=50, shuffle=True)

for train_index, test_index in kf.split(X):
    X_train = iris.data[train_index]
    y_train = iris.target[train_index]
    clf.fit(X_train, y_train)
    val = clf.predict(iris.data[test_index])
    # print(val, iris.target[test_index])
    for i in range(len(val)):
        if val[i] == iris.target[test_index[i]]:
            count += 1

accuracy = count / len(iris.data) * 100
print("accuracy (KFold(n=50)) = ", accuracy)
print ("Time taken for Kfold(n=50) = " ,time.clock() - start)


############################################################################################################################

clf =  MLPClassifier(alpha=1,solver='lbfgs')
print ("MLP Classifier : ")
from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()
loo.get_n_splits(digits.data)

# digits = iris
X = digits.data
X = StandardScaler().fit_transform(X)
y = digits.target


# Digits dataset: LeaveOneOut
start = time.clock()
count = 0
for train_index, test_index in loo.split(X):
    X_train = digits.data[train_index]
    y_train = digits.target[train_index]
    clf.fit(X_train,y_train)
    val = clf.predict(digits.data[test_index])
    if val == digits.target[test_index]:
        count += 1

accuracy = count/len(digits.data)*100
print ("accuracy (LeaveOneOut) = ",accuracy)
print ("Time taken for LeaveOneOut = " ,time.clock() - start)

# Digits dataset: KFold (n=10)

start = time.clock()
count = 0
from sklearn.model_selection import KFold

kf = KFold(n_splits=10, shuffle=True)

for train_index, test_index in kf.split(X):
    X_train = digits.data[train_index]
    y_train = digits.target[train_index]
    clf.fit(X_train, y_train)
    val = clf.predict(digits.data[test_index])
    # print(val, digits.target[test_index])
    for i in range(len(val)):
        if val[i] == digits.target[test_index[i]]:
            count += 1

accuracy = count / len(digits.data) * 100
print("accuracy (KFold(n=10)) = ", accuracy)
print ("Time taken for Kfold(n=10) = " ,time.clock() - start)


# Digits dataset: KFold (n=50)

start = time.clock()
count = 0
from sklearn.model_selection import KFold

kf = KFold(n_splits=50, shuffle=True)

for train_index, test_index in kf.split(X):
    X_train = digits.data[train_index]
    y_train = digits.target[train_index]
    clf.fit(X_train, y_train)
    val = clf.predict(digits.data[test_index])
    # print(val, digits.target[test_index])
    for i in range(len(val)):
        if val[i] == digits.target[test_index[i]]:
            count += 1

accuracy = count / len(digits.data) * 100
print("accuracy (KFold(n=50)) = ", accuracy)
print ("Time taken for Kfold(n=50) = " ,time.clock() - start)


# Iris dataset: LeaveOneOut

X = iris.data
X = StandardScaler().fit_transform(X)
y = iris.target


start = time.clock()
count = 0
for train_index, test_index in loo.split(X):
    X_train = iris.data[train_index]
    y_train = iris.target[train_index]
    clf.fit(X_train,y_train)
    val = clf.predict(iris.data[test_index])
    if val == iris.target[test_index]:
        count += 1

accuracy = count/len(iris.data)*100
print ("accuracy (LeaveOneOut) = ",accuracy)
print ("Time taken for LeaveOneOut = " ,time.clock() - start)

# Iris dataset: KFold (n=10)

start = time.clock()
count = 0
from sklearn.model_selection import KFold

kf = KFold(n_splits=10, shuffle=True)

for train_index, test_index in kf.split(X):
    X_train = iris.data[train_index]
    y_train = iris.target[train_index]
    clf.fit(X_train, y_train)
    val = clf.predict(iris.data[test_index])
    # print(val, iris.target[test_index])
    for i in range(len(val)):
        if val[i] == iris.target[test_index[i]]:
            count += 1

accuracy = count / len(iris.data) * 100
print("accuracy (KFold(n=10)) = ", accuracy)
print ("Time taken for Kfold(n=10) = " ,time.clock() - start)

# Iris dataset: KFold (n=50)

start = time.clock()
count = 0
from sklearn.model_selection import KFold

kf = KFold(n_splits=50, shuffle=True)

for train_index, test_index in kf.split(X):
    X_train = iris.data[train_index]
    y_train = iris.target[train_index]
    clf.fit(X_train, y_train)
    val = clf.predict(iris.data[test_index])
    # print(val, iris.target[test_index])
    for i in range(len(val)):
        if val[i] == iris.target[test_index[i]]:
            count += 1

accuracy = count / len(iris.data) * 100
print("accuracy (KFold(n=50)) = ", accuracy)
print ("Time taken for Kfold(n=50) = " ,time.clock() - start)


