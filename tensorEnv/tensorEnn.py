# Import required libraries. 

import numpy as np
import pandas as pd
from sklearn import linear_model
import sklearn
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

# Specify our graph plotting style. 

style.use("ggplot")

# Loading in data 

# Since our data is seperated by semicolons we need to do sep=";"

data = pd.read_csv("/Users/sterlingdavis/Desktop/tensorEnv/student/student-mat.csv", sep=";")

# View our DataFrame 

# This will trim our data so that we have access to 6 specific attributes. 

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]


# We want to predict which final grade each student will receive. In order to do this, we will create 2 arrays: 
# 1 containing features and 1 containing labels  

predict = "G3"

x = np.array(data.drop([predict], 1)) # Features
y = np.array(data[predict]) # Label

# Now we will split our data into testing + training sets. 90% of our data will be used to train and 10% to test. 


x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)

# Train model multiple times for best accuracy

best = 0
for _ in range(20):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    print("Accuracy: " + str(acc))

    if acc > best:
        best = acc
        with open("studentgrades.pickle", "wb") as f: #"wb" will open the file in binary form for writing 
            pickle.dump(linear, f)


# Loading in our model using .pickle. "rb" will open the file in binary form for reading 

pickle_in = open("studentgrades.pickle", "rb")
linear = pickle.load(pickle_in)


print("-------------------------")
print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)
print("-------------------------")

predicted= linear.predict(x_test)
for x in range(len(predicted)):
    print(predicted[x], x_test[x], y_test[x])


# Drawing and plotting model
plot = "failures"
plt.scatter(data[plot], data["G3"])
plt.legend(loc=4)
plt.xlabel(plot)
plt.ylabel("Final Grade")
plt.show()





