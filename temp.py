# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import matplotlib.pyplot as plt
from sklearn import datasets, svm


digits = datasets.load_digits()

def show_image(image, label):
    plt.figure(1, figsize=(3, 3))
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title(label)
    plt.show()
show_image(digits.images[5], digits.target[5])


X = digits.data
y = digits.dataset.target
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.33, random_state=42)

classifier = svm.SVC(gamma=0.001)

classifier.fit(X_train, y_train)

classifier.score(X_test, y_test)


predicted = classifier.predict(y_test[:20])
print(predicted)
print(testing_target[:20])