# Importing the libraries
from cProfile import label
from os import remove, stat
from tempfile import TemporaryDirectory
from turtle import shape
from matplotlib.cbook import flatten
import pandas as pd
import numpy as np
import scipy
from tensorflow import keras
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import statistics
import math
from scipy import stats
import functools
import operator
import pandas as pd    


INDEXES_TO_REMOVE = np.empty(shape=(1))
CORRELATION_TREND = []


def readLabels(file_location):
    labels = pd.read_csv(file_location, sep=" ", header=None)
    labels_list = []

    for i in labels:
        labels_list.append(labels[i])
    return labels_list[0]

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

def findOutlierIndexes(data_to_plot):
    index_list = data_to_plot[0]
    values_list = data_to_plot[1]
    classes_list = data_to_plot[2]
    indexes_to_remove = []
    
    if(len(values_list)>=2):
        outlier_criteria = statistics.mean(values_list)+statistics.stdev(values_list)*3
        for i in range(len(index_list)):
            if (abs(values_list[i]) > outlier_criteria):
                # print("Adding index",index_list[i], "to indexes to remove")
                indexes_to_remove.append(int(index_list[i]))
    else: 
        print("Only than one occurance: ", len(index_list))
        return indexes_to_remove.append(int(index_list[0]))
    return indexes_to_remove

        
# Params: List<List<FLoat>>, List<Int>, Int
# Returns: List< List<Float>, List<FLoat>, List<Float> >
# Function: Return the indexes where the class doesn't equal zero
def trackTrendForFeature(relevant_data, labels_list, feature_index):
    plot_list = np.array([])
    # We have 1917 relevant features.
    # How does it trend across 2000 datapoints
    # Track trend for particular feature (in this case feature[0])
    # If feature isn't zero, track it's index in the dataset and record the feature 
    
    for i in range(len(relevant_data)):
        feature = relevant_data[i][feature_index]
        # record datapoint index and feature value in plot_list
        if (feature != 0):
            val_to_append =[i ,feature, labels_list[i]]
            # print("Adding datapoint number:", i, "Feature index:", index, " with Class label:", labels_list[i])
            plot_list = np.append( plot_list, val_to_append)
    index_list = []
    value_list = []
    class_list = []
    # get indexes and values
    counter_3 = 0
    for i in range(len(plot_list)):
        if (counter_3 == 0 ):
            index_list.append(plot_list[i]) 
            counter_3 += 1
            continue
        elif (counter_3 == 1):
            value_list.append(plot_list[i])
            counter_3 += 1
            continue
        else:
            class_list.append(plot_list[i])
            counter_3 = 0
            continue
    
    return [index_list, value_list, class_list]

#  Returns the number of times a class has shown up in the data set, while a specified feature was non zero
def countClassIfFeatureNonZero(data_to_plot, feature_index):
    data_to_plot = trackTrendForFeature(relevant_data, labels_list, feature_index) # [feature_index_list, value_list, label_list]
    classes_list = data_to_plot[2]
    values_list = data_to_plot[1]
    index_list = data_to_plot[0]
    class_range = np.zeros(shape=(10))
    
    # Count classes and store in onehot form
    for i in range(len(classes_list)):
        class_range[int(classes_list[i])] += 1
    
    
    # Track the ratio of a feature's CLASS_mean/class_mode
    class_correlations = (statistics.mean(classes_list)+1)/(statistics.mode(classes_list)+1)
    print("Feature", feature_index, " is ", class_correlations, "correlated with class", statistics.mode(classes_list))
    
    CORRELATION_TREND.append([feature_index, class_correlations, statistics.mode(classes_list)])
    
    indexes_to_pop = findOutlierIndexes(data_to_plot)
    indexes_to_pop = np.array(indexes_to_pop).flatten()   
    global INDEXES_TO_REMOVE 
    INDEXES_TO_REMOVE = np.append(INDEXES_TO_REMOVE, indexes_to_pop)
    print("Number of outliers found", len(INDEXES_TO_REMOVE))
        
    return class_range


# Functions Ends
    
    

# region Pipeline Starts
# region Pipeline
# Declarations.
dataset_input = []
dataset_class = []
count = 0  # count number of rows we've ran through

# Open up the data and clean it

labels_list = readLabels("labels.txt")

with open("inputs.txt") as file:

    memory_list = []  # Just to store the memory list, indexes to keep
    for lines in file:  # Iterate through row by row
        line = lines.split()  # Remove spaces
        val_list = []
        # Convert line to array and initialize memory list for single data point
        converted = convertLineToDataPointAndMemoryList(line)
        # Array of zeroes and non-zeroes representing redundancy)
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



for i in range(len(relevant_data[0])):
    data_to_plot = trackTrendForFeature(relevant_data, labels_list, i) # [index_list, value_list, label_list]
    classCountForFeatureX = countClassIfFeatureNonZero(data_to_plot, i)
    
    ## Flattens the multidimensional INDEXES_TO_REMOVE
    
    tempToRemove = np.array(INDEXES_TO_REMOVE)
    flatToRemove = tempToRemove.flatten()
    
    
    # print(classCountForFeatureX, i)
## Remove the bad data from the dataset
# Removes duplicates from INDEXES to REMOVE
INDEXES_TO_REMOVE = list(dict.fromkeys(INDEXES_TO_REMOVE))
print("Number of INDEXES_TO_REMOVE:",len(INDEXES_TO_REMOVE))

for i, e in reversed(list(enumerate(relevant_data))):
    for j in range(len(INDEXES_TO_REMOVE)):
        if INDEXES_TO_REMOVE[j] == i:
            print("Popping datapoint at index", i)
            
            relevant_data.pop(i)
            labels_list.pop(i)

print("Relevant data should be less than 2000:",len(relevant_data))
print("Indexes we removed:", len(INDEXES_TO_REMOVE))
print("Labels should be equal to the number above:",len(labels_list))

relevant_dataFrame = pd.DataFrame(relevant_data)
labels_list_dataFrame = pd.DataFrame(relevant_data)

relevant_dataFrame.to_csv('labels_NO_OUTLIERS.csv', index=False)
labels_list_dataFrame.to_csv('inputs_NO_OUTLIERS.csv', index=False)
    
plt.scatter(data_to_plot[0], data_to_plot[1] )
plt.ylabel("Feature X's value")
plt.xlabel("Index in dataset")
    
# endregion 