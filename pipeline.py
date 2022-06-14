from posixpath import relpath
from venv import create
import sklearn
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import numpy as np

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
# Pipeline Starts

# Declarations.
dataset_input = []
dataset_class = []
count = 0  # count number of rows we've ran through

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

# Pull Request template, What was your task, how do you know it is working? Show on terminal
# print("Please print a test of your code when committing, as follows")
# print("TODO: Removing redundant data from data set")
# print("Input: Full dataset;", "Expected output: Full Data > Relevant Data ")
# print(
#     "Full dataset length:",
#     len(dataset_input[0]),
#     "; Relevant dataset length:",
#     len(relevant_data[0]),
# )

labels = pd.read_csv("labels.txt", sep=" ", header=None)
labels = np.ravel(labels)

X_train , X_test , Y_train ,y_test = train_test_split(relevant_data, labels ,test_size = 0.2 , random_state= 1) #20% test data , and 80% training data
x_train , x_val , y_train, y_val = train_test_split(X_train, Y_train ,test_size = 0.25 , random_state= 1) #split 20% validation data 80% training data and remain with 60% training data

clf = MLPClassifier(solver='adam',activation = 'logistic' ,alpha=0.00001, hidden_layer_sizes=(570,), random_state=1, max_iter=300)

clf.fit(x_train, y_train)
# validation =  clf.predict(np.reshape(x_val[0], (1, -1)))
# print(str(validation))

# print(np.reshape(x_val[0], (1, -1)))

print(clf.coefs_[0])

# print('Validation Accuracy : %.3f'%clf.score(x_val, y_val))
# print('Training Accuracy : %.3f'%clf.score(x_train, y_train))

# print(len(relevant_data))
# print(len(labels))

# print("Memory list length:", len(memory_list))
# print("Relevant datapoint length:", len(relevant_data[1000]))
# print("Length of full dataset:", len(dataset_input))
# print("Length of datapoint:", len(dataset_input[0]))
# print("Datapoint row 0 zolumn zero:", dataset_input[0][0])

# Pipeline Ends
