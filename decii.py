import pandas as pd
import math
import numpy as np

# Load dataset
data = pd.read_csv("3.csv")

# Print column names and the first few rows to verify data and column names
print("Columns in the dataset:", data.columns)
print(data.head())  # Print the first few rows to inspect the data

# Prompt the user to enter the correct column name if 'answer' is not found
if 'answer' not in data.columns:
    correct_column = input("The 'answer' column is not found. Please enter the correct column name for the target variable: ")
    if correct_column in data.columns:
        data.rename(columns={correct_column: 'answer'}, inplace=True)  # Rename the column to 'answer'
    else:
        raise KeyError(f"The column '{correct_column}' is also not found in the dataset. Please check the file and column names.")

features = [feat for feat in data.columns if feat != "answer"]

# ... (rest of the code remains the same)

# Node class definition
class Node:
    def _init_(self):
        self.children = []
        self.value = ""
        self.isLeaf = False
        self.pred = ""

# Function to calculate entropy
def entropy(examples):
    pos = 0.0
    neg = 0.0
    for _, row in examples.iterrows():
        if row["answer"] == "yes":
            pos += 1
        else:
            neg += 1
    if pos == 0.0 or neg == 0.0:
        return 0.0
    else:
        p = pos / (pos + neg)
        n = neg / (pos + neg)
        return -(p * math.log(p, 2) + n * math.log(n, 2))

# Function to calculate information gain
def info_gain(examples, attr):
    uniq = np.unique(examples[attr])
    gain = entropy(examples)
    for u in uniq:
        subdata = examples[examples[attr] == u]
        sub_e = entropy(subdata)
        gain -= (float(len(subdata)) / float(len(examples))) * sub_e
    return gain

# ID3 algorithm implementation
def ID3(examples, attrs):
    root = Node()
    max_gain = -1  # Initialize to a very small value to ensure any gain will be greater
    max_feat = None  # Initialize to None for cases where no gain is found

    for feature in attrs:
        gain = info_gain(examples, feature)
        if gain > max_gain:
            max_gain = gain
            max_feat = feature

    if max_feat is None:
        return None  # No split can be performed

    root.value = max_feat
    uniq = np.unique(examples[max_feat])

    for u in uniq:
        subdata = examples[examples[max_feat] == u]
        if entropy(subdata) == 0.0:
            newNode = Node()
            newNode.isLeaf = True
            newNode.value = u
            newNode.pred = np.unique(subdata["answer"])[0]
            root.children.append(newNode)
        else:
            dummyNode = Node()
            dummyNode.value = u
            new_attrs = [attr for attr in attrs if attr != max_feat]
            child = ID3(subdata, new_attrs)
            if child:  # Only add the child if it's not None
                dummyNode.children.append(child)
            root.children.append(dummyNode)

    return root

# Function to print the decision tree
def printTree(root: Node, depth=0):
    if root is None:
        return

    for i in range(depth):
        print("\t", end="")
    print(root.value, end="")
    if root.isLeaf:
        print(" -> ", root.pred)
    else:
        print()
    for child in root.children:
        printTree(child, depth + 1)

# Build and print the decision tree
root = ID3(data, features)
printTree(root)
