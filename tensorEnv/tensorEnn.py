# Import required libraries. 

import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

# Loading in data 

# Since our data is seperated by semicolons we need to do sep=";"

data = pd.read_csv("/Users/sterlingdavis/Desktop/tensorEnv/student/student-mat.csv", sep=";")

# View our DataFrame 

# This will trim our data so that we have access to 6 specific attributes. 

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]


# We want to predict which final grade each student will receive. In order to do this, we will create 2 arrays: 
# 1 containing features and 1 containing labels  

predict = "G3"

X = np.array(data.drop([predict], 1)) # Features
y = np.array(data[predict]) # Label

# Now we will split our data into testing + training sets. 90% of our data will be used to train and 10% to test. 


x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

# Define model that we will be using. 

linear = linear_model.LinearRegression()

# Train and score our model utilizing arrays

linear.fit(x_train, y_train)
acc = linear.score(x_test, y_test) # acc (this is our accuracy score)

# This will display the constants used to generate our line for linear regression:

print('Coefficient: \n', linear.coef_) # slope value
print('Intercept: \n', linear.intercept_) # intercept

# Predicting grade scores on specific students 

predictions = linear.predict(x_test) # Gets a list of all predictions

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])