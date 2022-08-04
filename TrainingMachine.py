import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score

"""**Data Gathering**"""

# loading CSV data through pandas
parkinsons_data = pd.read_csv('C:/Users/soft_/Desktop/parkinsons.csv')

# showing CSV
parkinsons_data.head()

# No of rows and columns
#parkinsons_data.shape()

# info about DATA set
parkinsons_data.info()

# checking missing values
parkinsons_data.isnull().sum()

# getting statistical measures about data set
parkinsons_data.describe()

# distribution of target variable
parkinsons_data['status'].value_counts()

"""1 -> Parkison 
0 -> Healthy 
"""

# grouping data based on target variable (status)
parkinsons_data.groupby('status').mean()

"""Data pre-processing

separating the features and target
"""

X = parkinsons_data.drop(columns=['name', 'status'], axis=1)
Y = parkinsons_data['status']
print(X)

print(Y)

""" Spliting into Train and Test data"""

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

print(X.shape, X_train.shape, X_test.shape)

""" Data standarization """

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)

print(X_train)

"""Model Trainig

Support Vector Model
"""

model = svm.SVC(kernel='linear')

# trainig SVM model
model.fit(X_train, Y_train)

""" Model Evaluation

Accuracy Score
"""

# Accuracy score on train data
X_train_predicition = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_predicition)

print('Accuracy score of training data : ', training_data_accuracy)

# Accuracy score on test data
X_test_predicition = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_predicition)

print('Accuracy score of test data : ', test_data_accuracy)

"""Building a Predictive System

"""

input_data = (
    162.56800, 198.34600, 77.63000, 0.00502, 0.00003, 0.00280, 0.00253, 0.00841, 0.01791, 0.16800, 0.00793, 0.01057,
    0.01799, 0.02380, 0.01170, 25.67800, 0.427785, 0.723797, -6.635729, 0.209866, 1.957961, 0.135242)

# changing input data to numpy array
input_data_as_numpy_arr = np.asarray(input_data)

# reshape arr
input_data_reshaped = input_data_as_numpy_arr.reshape(1, -1)

# standarize data
std_data = scaler.transform(input_data_reshaped)

prediction = model.predict(std_data)
print(prediction)

if prediction[0] == 0:
    print("Patient do not have parkinsons")

else:
    print("Patient has Parkinsons")
