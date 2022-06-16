# Importing the libraries
from turtle import shape
import pandas as pd
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import math

# Params: List<String>
# Returns: List<Float>, List<Float>
# Function: Convert list of scientific notation inputs to list of floats
def convertLineToDataPointAndMemoryList(line):
    zeroes_list = []  # Used to store initialized memory list as array of zeroes
    # Convert each element to list of floats from scientific notation
    for j in range(len(line)):
        exponent = float(line[j][-2:])
        num = float(line[j][0:19])
        val = num * 10 ** (exponent)  # Converts scientific notation to float
        val_list.append(val)  # append value onto list of datapoint elements
        zeroes_list.append(0.0)  # Just to store size

    return [  # Returns line converted to float list, and a zero array of the same size
        val_list,
        zeroes_list,
    ]


# Params: (List<<List<float>>>, <float>)
# Returns: List<float>
# Function: Return list representing redundant data point indexes as zeroes.
def createMemoryList(dataset_input, memory_list):
    # Iterate through each data point's elements
    for i in range(len(dataset_input)):
        for j in range(len(dataset_input[i])):
            # Don't update element in memory list if it is equal to zero
            if dataset_input[i][j] != 0:
                memory_list[j] = dataset_input[i][j]

    return memory_list


# Params: List<List<FLoat>>, List<Float>
# Returns: List<List<FLoat>>
# Function: Remove corresponding redundant indeces from each datapoint
def removeRedundantData(dataset_input, memory_list):
    # Declare relevant Dataset
    relevant_data = []
    # Iterate through dataset
    for i in range(len(dataset_input)):
        relevant_data_point = []
        for j in range(len(dataset_input[i])):
            # If the memory list for datapoints at this index is 0, append it to the relevant datapoint list
            if memory_list[j] != 0:
                relevant_data_point.append(dataset_input[i][j])
        relevant_data.append(relevant_data_point)
    return relevant_data


# def trackTrendPerFeature(relevant_data, labels, index):
#     plot_list = []    
#     # We have 1917 relevant features.
#     # How does it trend across 2000 datapoints
#     # Track trend for particular feature (in this case feature[0])
#     # If feature isn't zero, track it's index in the dataset and record the feature 
    
#     for i in range(len(relevant_data)):
#         feature = relevant_data[i][index]
        
#         if (feature != 0):
#             val_to_append =(i ,feature)
#             plot_list = np.append( plot_list, val_to_append )
            
#             print("Data point at index", i ,"belongs to class", labels[i])
#             print(plot_list)
        
#     index_list = []
#     value_list = []
#     for i in range(len(plot_list)):
#         if (i%2 != 0 ):
#             value_list.append(plot_list[i])
#         else: index_list.append(plot_list[i])
    
    
#     return [index_list, value_list]

    
    


# Functions Ends

# region Pipeline Starts

# Declarations.
dataset_input = []
dataset_class = []
count = 0  # count number of rows we've ran through

# Open up the data and clean it

labels = pd.read_csv("labels.txt", sep=" ", header=None)
with open("inputs.txt") as file:

    memory_list = []  # Just to store the memory list, indexes to keep
    for lines in file:  # Iterate through row by row
        line = lines.split()  # Remove spaces
        val_list = []
        # Convert line to array and initialize memory list for single data point
        converted = convertLineToDataPointAndMemoryList(line)
        # Array of zeroes and non-zeroes representing redundancy
        memory_list = converted[1]  # Initializes length of mem list to match data point
        val_list = converted[0]
        dataset_input.append(val_list)  # append each word as element in datapoint
        # print("Line:", count, "of 2000")
        count += 1
    # Create list of redundant data
    memory_list = createMemoryList(dataset_input, memory_list)

    # Remove the redundant rows from memory list.

    relevant_data = removeRedundantData(dataset_input, memory_list)
    #TODO: Iterate through index 0, and find trend for relevant_data[i][0]
    
    # data_to_plot = trackTrendPerFeature(relevant_data, labels, 2)
    # plt.scatter(data_to_plot[0], data_to_plot[1] )
    # plt.ylabel("Feature 77 value")
    # plt.xlabel("Index in dataset")
    # plt.show()
    

# Relevant data is cleaned data, full scope



from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
inputs = sc.fit_transform(relevant_data)


# from sklearn.preprocessing import OneHotEncoder
# ohe = OneHotEncoder()
# labels = ohe.fit_transform(labels).toarray()

x_Train , X_Test , Y_train ,Y_test = train_test_split(inputs, labels ,test_size = 0.2 , random_state= 1 ,shuffle=True)
# Split the whole 
X_train , X_val , Y_train, Y_val = train_test_split(x_Train, Y_train ,test_size = 0.25 , random_state= 1 ,shuffle=True)

print(Y_val.shape)

model = Sequential()
# Hardcoded, 
# TODO: Justify why we use these numbers (For the pdf)
# Dense( int= next hidden layer dimensions, activation function, input dimensions )
# TODO: Test different types and record values
# TODO: Justify choice of layer type "Dense" has alternatives

relevant_dimensions= len(relevant_data[0])
model.add(Dense(relevant_dimensions//2, activation='relu', input_dim=relevant_dimensions, kernel_initializer='he_uniform')) # This defines the dimensions of the input dimension and 1st hidden layer
model.add(Dropout(0.5)) # We added drop out as half to reduce the underfitting.
model.add(Dense(relevant_dimensions//19, activation='relu', kernel_initializer='he_uniform'))
model.add(Dropout(0.5)) # We added drop out as half to reduce the underfitting.
model.add(Dense(10, activation='softmax'))

# TODO: Justify why we picked these specific optimizer, loss and metric parameters
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, epsilon=0.004, amsgrad=True), 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

model.fit(X_train, Y_train,  epochs=90,  batch_size=25)

# model.save_weights('model_weights.h5')

y_pred = model.predict(X_val)
#Converting predictions to label
pred = list()
for i in range(len(y_pred)):
    pred.append(np.argmax(y_pred[i]))
#Converting one hot encoded test label to label
test = list()
for i in range(len(Y_val)):
    test.append(np.argmax(Y_val[i]))
    
from sklearn.metrics import accuracy_score
a = accuracy_score(pred,test)
print('[Validation] Accuracy is:', a*100)

y_pred = model.predict(X_Test)
#Converting predictions to label
pred = list()
for i in range(len(y_pred)):
    pred.append(np.argmax(y_pred[i]))
#Converting one hot encoded test label to label
test = list()
for i in range(len(Y_test)):
    test.append(np.argmax(Y_test[i]))
    
from sklearn.metrics import accuracy_score
b = accuracy_score(pred,test)
print('[Testing] Accuracy is:', b*100)    

