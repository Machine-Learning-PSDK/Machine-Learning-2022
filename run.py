import sklearn
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import numpy as np


inputs = pd.read_csv("inputs.txt", sep=" ", header=None)
labels = pd.read_csv("labels.txt", sep=" ", header=None)
labels = np.ravel(labels)

X_train , X_test , y_train ,y_test = train_test_split(inputs, labels ,test_size = 0.2 , random_state= 1) #20% test data , and 80% training data
x_train , x_val , Y_train, Y_val = train_test_split(X_train, y_train ,test_size = 0.25 , random_state= 1) #split 20% validation data 80% training data and remain with 60% training data

clf = MLPClassifier(solver='adam',activation = 'relu' ,alpha=1e-5, hidden_layer_sizes=(600, 600), random_state=1)

clf.fit(x_train, Y_train)
validation =  clf.predict(x_val)

print('validation Accuracy : %.3f'%clf.score(x_val, Y_val))
print('Training Accuracy : %.3f'%clf.score(x_train, Y_train))

testing =clf.predict(X_test)
print('Test Accuracy : %.3f'%clf.score(X_test, y_test))
print('Training Accuracy : %.3f'%clf.score(x_train, Y_train))