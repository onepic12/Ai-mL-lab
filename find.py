import pandas as pd
import numpy as np

# Read the data from the CSV file
data = pd.read_csv("ws.csv")
print(data, "\n")

# Make an array of all the attributes
d = np.array(data)[:, :-1]
print("The attributes are:", d)

# Separate the target that has positive and negative examples
target = np.array(data)[:, -1]
print("The target is:", target)

# Training function to implement the Find-S algorithm
def train(c, t):
    for i, val in enumerate(t):
        if val == "Yes":
            specific_hypothesis = c[i].copy()
            break
    for i, val in enumerate(c):
        if t[i] == "Yes":
            for x in range(len(specific_hypothesis)):
                if val[x] != specific_hypothesis[x]:
                    specific_hypothesis[x] = '?'
                else:
                    pass
    return specific_hypothesis

# Obtain the final hypothesis
print("The final hypothesis is:", train(d, target))
