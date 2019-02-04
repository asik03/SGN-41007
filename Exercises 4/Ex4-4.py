# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.image as mpimg
from math import floor, ceil
import os

# Sklearn imports
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import glob

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier


import matplotlib.pyplot as plt

def bilinear(image, r, c):
    minr = floor(r)
    minc = floor(c)
    maxr = ceil(r)
    maxc = ceil(c)

    dr = r-minr
    dc = c-minc

    top = (1-dc)*image[minr,minc] + dc*image[minr,maxc]
    bot = (1-dc)*image[maxr,minc] + dc*image[maxr,maxc]

    return (1-dr)*top+dr*bot

def local_binary_pattern(image, P=8, R=1):
    rr = - R * np.sin(2*np.pi*np.arange(P, dtype=np.double) / P)
    cc = R * np.cos(2*np.pi*np.arange(P, dtype=np.double) / P)
    rp = np.round(rr, 5)
    cp = np.round(cc, 5)
    
    rows = image.shape[0]
    cols = image.shape[1]

    output = np.zeros((rows, cols))

    for r in range(R,rows-R):
        for c in range(R,cols-R):
            lbp = 0
            for i in range(P):
                if bilinear(image, r+rp[i], c+cp[i]) - image[r,c] >= 0:
                    lbp += 1<<i
                            
            output[r,c] = lbp

    return output

def load_data(folder):
    """ 
    Load all images from subdirectories of
    'folder'. The subdirectory name indicates
    the class.
    """
    
    X = []          # Images go here
    y = []          # Class labels go here
    classes = []    # All class names go here
    
    subdirectories = glob.glob(folder + "/*")
    
    # Loop over all folders
    for d in subdirectories:
        
        # Find all files from this folder
        files = glob.glob(d + os.sep + "*.jpg")
        
        # Load all files
        for name in files:
            
            # Load image and parse class name
            img = plt.imread(name)
            class_name = name.split(os.sep)[-2]

            # Convert class names to integer indices:
            if class_name not in classes:
                classes.append(class_name)
            
            class_idx = classes.index(class_name)
            
            X.append(img)
            y.append(class_idx)
    
    # Convert python lists to contiguous numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    return X, y

def extract_lbp_features(X, P = 8, R = 5):
    """
    Extract LBP features from all input samples.
    - R is radius parameter
    - P is the number of angles for LBP
    """
    
    F = [] # Features are stored here
    
    N = X.shape[0]
    for k in range(N):
        
        print("Processing image {}/{}".format(k+1, N))
        
        image = X[k, ...]
        lbp = local_binary_pattern(image, P, R)
        hist = np.histogram(lbp, bins=range(257))[0]
        F.append(hist)

    return np.array(F)

# Test our loader
 
X, y = load_data('./GTSRB_subset')
F = extract_lbp_features(X)
print("X shape: " + str(X.shape))
print("F shape: " + str(F.shape))
images = []
labels = []
features = []
scores = []

X = X.reshape(X.shape[0], X.shape[1]*X.shape[2])   



X_train, X_test, y_train, y_test = train_test_split(X, y)


## Training with different classifiers
# Nearest neighbor
neigh = KNeighborsClassifier(n_neighbors=3)
scores_neigh = cross_val_score(neigh, X, y, cv=5)
print("Neigh score:")
print(scores_neigh)
neigh.fit(X_train, y_train)
neigh_score = accuracy_score(y_test, neigh.predict(X_test))
scores.append(neigh_score)

# LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()
scores_lda = cross_val_score(lda, X, y, cv=5)
print("Lda score: ") 
print(scores_lda)
lda.fit(X_train, y_train)
lda_score = accuracy_score(y_test, lda.predict(X_test))
scores.append(lda_score)

# SVM
svm = SVC(gamma='auto')
scores_svm = cross_val_score(svm, X, y, cv=5)
print("Svm score:")
print( scores_svm)
svm.fit(X_train, y_train)
svm_score = accuracy_score(y_test, svm.predict(X_test))
scores.append(svm_score)

# Logistic Regrasion
lmr = LogisticRegression()
scores_lmr = cross_val_score(lmr, X, y, cv=5)
print("Lmr score:")
print(scores_lmr)
lmr.fit(X_train, y_train)
lmr_score = accuracy_score(y_test, lmr.predict(X_test))
scores.append(lmr_score)


# --------------Ex 5-----------
forest = RandomForestClassifier(n_estimators=100, max_depth=2)
scores_lmr = cross_val_score(forest, X, y, cv=5)
print("Random Forest Score:")
print(scores_lmr)


extra = ExtraTreesClassifier()
scores_extra = cross_val_score(extra, X, y, cv=5)
print (scores_extra)

ada = AdaBoostClassifier()
scores_ada = cross_val_score(ada, X, y, cv=5)
print (scores_ada)

boost = GradientBoostingClassifier()
scores_boost = cross_val_score(boost, X, y, cv=5)
print(scores_boost)

