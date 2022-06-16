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


# Functions Ends

# region Pipeline Starts

# Declarations.
dataset_input = []
dataset_class = []
count = 0  # count number of rows we've ran through

# Open up the data and clean it
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

# Relevant data is cleaned data, full scope

# inputs = pd.read_csv("inputs.txt", sep=" ", header=None)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
inputs = sc.fit_transform(relevant_data)

labels = pd.read_csv("labels.txt", sep=" ", header=None)

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
labels = ohe.fit_transform(labels).toarray()

x_Train , X_Test , Y_train ,Y_test = train_test_split(inputs, labels ,test_size = 0.2 , random_state= 1 ,shuffle=True)
# Split the whole 
X_train , X_val , Y_train, Y_val = train_test_split(x_Train, Y_train ,test_size = 0.25 , random_state= 1 ,shuffle=True)

model = Sequential()
# Hardcoded, 
# TODO: Justify why we use these numbers (For the pdf)
# Dense( int= next hidden layer dimensions, activation function, input dimensions )
# TODO: Test different types and record values
# TODO: Justify choice of layer type "Dense" has alternatives



# relevant_dimensions= len(relevant_data[0])
# model.add(Dense(558, activation='relu', input_dim=relevant_dimensions)) # This defines the dimensions of the input dimension and 1st hidden layer
# # model.add(Dense(relevant_dimensions/2, activation='relu'))
# # model.add(Dense(relevant_dimensions/4, activation='relu'))
# # model.add(Dense(relevant_dimensions/8, activation='relu'))
# model.add(Dense(10, activation='softmax'))  # This defines output layer dimensions

relevant_dimensions= len(relevant_data[0])
model.add(Dense(relevant_dimensions//8, activation='relu', input_dim=relevant_dimensions)) # This defines the dimensions of the input dimension and 1st hidden layer
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
# model.add(Dense(relevant_dimensions/2, activation='relu', input_dim=relevant_dimensions)) # This defines the dimensions of the input dimension and 1st hidden layer
# model.add(Dense(relevant_dimensions/4, activation='relu'))
# model.add(Dense(10, activation='softmax'))  # This defines output layer dimensions
# model.load_weights('my_model_weights.h5')
# Compile the model



# TODO: Justify why we picked these specific optimizer, loss and metric parameters
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, epsilon=0.004, amsgrad=True), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

model.fit(X_train, Y_train,  epochs=95,  batch_size=10)

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

