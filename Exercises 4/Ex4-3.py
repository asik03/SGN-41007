# -*- coding: utf-8 -*-
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as plt

digits = load_digits()

print(digits.keys())

plt.gray() # DIsplay image as greyscale
plt.imshow(digits.images[0])
plt.show()
print(digits.target[0])

X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target)

print(y_train)
scores = []

# Nearest neighbor
neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(X_train, y_train)
neigh_score = accuracy_score(y_test, neigh.predict(X_test))
scores.append(neigh_score)

# LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
lda_score = accuracy_score(y_test, lda.predict(X_test))
scores.append(lda_score)

# SVM
svm = SVC(gamma='auto')
svm.fit(X_train, y_train)
svm_score = accuracy_score(y_test, svm.predict(X_test))
scores.append(svm_score)

# Logistic Regrasion
lmr = LogisticRegression()
lmr.fit(X_train, y_train)
lmr_score = accuracy_score(y_test, lmr.predict(X_test))
scores.append(lmr_score)

for i in scores:
    print(i)
# Values are different every testing, not using a random state
