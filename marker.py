# Importing the libraries
from sys import stdin, stdout
import numpy as np
import tensorflow as tf


# Pre processing
#__________________________________________________
def removeRedundantFeatures(dataset_input, memory_list):
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
    print("Shape of relevant data", len(relevant_data), len(relevant_data[0]) )
    return relevant_data

memory_list = []
with open("memorylist.txt", "r") as f:
    for line in f:
            memory_list.append(float(line.strip()))

CLASSES = []
#___________________________________ preprocessing


# Loads our saved model from folder
model = tf.keras.models.load_model('saved_model.h5')

# Loads in test data from stdin as requested. but it's not working. How do we communicate with the marker?
test_data= np.loadtxt(stdin, max_rows=100)

# Preprocessing function
test_data_clean = removeRedundantFeatures(test_data, memory_list)

# Make predictions, store in list
CLASSES.append(model.predict(test_data_clean))


# Write output to communicate with the marker as concatenated string
answer = ''.join(map(str, CLASSES))
stdout.write(str(answer))