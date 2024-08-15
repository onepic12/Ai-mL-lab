import numpy as np 
import pandas as pd

# Read the data from the CSV file
data = pd.read_csv("2.csv")
concepts = np.array(data.iloc[:, 0:-1])
print("\nInstances are:\n", concepts)
target = np.array(data.iloc[:, -1])
print("\nTarget Values are: ", target)

# Learning function to implement the Candidate Elimination algorithm
def learn(concepts, target): 
    specific_h = concepts[0].copy()
    print("\nInitialization of specific_h and general_h")
    print("\nSpecific Boundary: ", specific_h)
    general_h = [["?" for i in range(len(specific_h))] for i in range(len(specific_h))]
    print("\nGeneral Boundary: ", general_h)  

    for i, h in enumerate(concepts):
        print("\nInstance", i+1 , "is ", h)
        if target[i] == "yes":
            print("Instance is Positive")
            for x in range(len(specific_h)): 
                if h[x] != specific_h[x]:                    
                    specific_h[x] = '?'                     
                    general_h[x][x] = '?'

        if target[i] == "no":            
            print("Instance is Negative")
            for x in range(len(specific_h)): 
                if h[x] != specific_h[x]:                    
                    general_h[x][x] = specific_h[x]                
                else:                    
                    general_h[x][x] = '?'        

        print("Specific Boundary after ", i+1, " Instance is ", specific_h)         
        print("General Boundary after ", i+1, " Instance is ", general_h)
        print("\n")

    indices = [i for i, val in enumerate(general_h) if val == ['?', '?', '?', '?', '?', '?']]    
    for i in indices:   
        general_h.remove(['?', '?', '?', '?', '?', '?']) 
    return specific_h, general_h 

s_final, g_final = learn(concepts, target)

print("Final Specific_h: ", s_final, sep="\n")
print("Final General_h: ", g_final, sep="\n")
