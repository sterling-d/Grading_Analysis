# Import required libraries. 

import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

# Loading in Data 

# Since our data is seperated by semicolons we need to do sep=";"

data = pd.read_csv("/Users/sterlingdavis/Desktop/tensorEnv/student/student-mat.csv", sep=";")

# View our DataFrame 

print(data.head())

# This will trim our data so that we have access to 6 specific attributes. 

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
