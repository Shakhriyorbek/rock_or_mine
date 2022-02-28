# %%
# Importing dictionaries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Data collection and Data Processing

# loading the dataset to a pandas dataset
sonar_data = pd.read_csv('sonar_data.csv', header=None)
sonar_data.head()


# number of rows and columns

sonar_data.shape


sonar_data.describe()  # describe --> statistical measures of the data


sonar_data[60].value_counts()


# M --> Mine
# R --> Rock

# %%
sonar_data.groupby(60).mean()


# separating data and labels
X = sonar_data.drop(columns=60, axis=1)
Y = sonar_data[60]


print(X)
print(Y)


# Training and Test data

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.1, stratify=Y, random_state=1)


print(X.shape, X_train.shape, X_test.shape)


# Model training --> Logistic Regression


model = LogisticRegression()

# %%
# training the Logistic Regression with training data
model.fit(X_train, Y_train)


print(X_train)
print(Y_train)


# Model Evaluation


# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)


print(training_data_accuracy)


X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)


print(test_data_accuracy)


# Making a predictive System
input_data = (0.0414, 0.0436, 0.0447, 0.0844, 0.0419, 0.1215, 0.2002, 0.1516, 0.0818, 0.1975, 0.2309, 0.3025, 0.3938, 0.5050, 0.5872, 0.6610, 0.7417, 0.8006, 0.8456, 0.7939, 0.8804, 0.8384, 0.7852, 0.8479, 0.7434, 0.6433, 0.5514, 0.3519, 0.3168,
              0.3346, 0.2056, 0.1032, 0.3168, 0.4040, 0.4282, 0.4538, 0.3704, 0.3741, 0.3839, 0.3494, 0.4380, 0.4265, 0.2854, 0.2808, 0.2395, 0.0369, 0.0805, 0.0541, 0.0177, 0.0065, 0.0222, 0.0045, 0.0136, 0.0113, 0.0053, 0.0165, 0.0141, 0.0077, 0.0246, 0.0198)


# changing the input data to a np array
input_data_as_np = np.asarray(input_data)

# reshape the np array as we are predicting for one instance
input_data_reshaped = input_data_as_np.reshape(1, -1)

prediction = model.predict(input_data_reshaped)
print(prediction)


if(prediction[0] == 'R'):
    print('The object is rock')
else:
    print('The object is mine. Be careful!')
