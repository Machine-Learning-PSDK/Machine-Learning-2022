#

dataset_input = []
dataset_class = []
count = 0  # count number of rows we've ran through

# 0.000000000000000000e+00
# str(scan first 20 chars)
# int(lines[23,24])
def convertLineToDataPointAndMemoryList(line):
    count_list = []
    for j in range(len(line)):
        exponent = float(line[j][-2:])
        num = float(line[j][0:19])
        val = num * 10 ** (exponent)
        val_list.append(val)
        count_list.append(0.0)
    return [val_list, count_list]


def createMemoryList(dataset_input):
    for i in range(len(dataset_input)):
        for j in range(len(dataset_input[i])):
            if dataset_input[i][j] != 0:
                memory_list[j] = dataset_input[i][j]
    return memory_list


with open("inputs.txt") as f:

    memory_list = []
    for lines in f:
        line = lines.split()
        val_list = []
        converted = convertLineToDataPointAndMemoryList(
            line
        )  # Convert line to array and initialize memory list
        memory_list = converted[1]
        val_list = converted[0]
        dataset_input.append(val_list)  # append each word as element in datapoint
        # print("Line:", count, "of 2000")
        count += 1

    memory_list = createMemoryList(dataset_input)  # Create list of redundant data

print("Memory list", memory_list)
print("Length of full dataset:", len(dataset_input))
print("Length of datapoint:", len(dataset_input[0]))
print("Datapoint column:", dataset_input[0][0])
