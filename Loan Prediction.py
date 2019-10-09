## Load Important Libraries 

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

## Load The dataset with the help of pandas
## data set is  given in repository


data=pd.read_csv('train_ctrUa4K.csv')
print(data.head(5))

## Name of Columns we can check through

print("Columns\n\n")
print(data.columns)

# now use label encoder

encode=LabelEncoder()
data.Loan_Status=encode.fit_transform(data.Loan_Status)
data.Property_Area=encode.fit_transform(data.Property_Area)

## Now we will drop rows which contain empty fields with help of pandas

data.dropna(how='any',inplace=True)

## Now we will bifurgate the data into test and train percentage in this I have used 80% data to train my model and 20% data to test my model.

X_train, X_test, y_train, y_test=train_test_split(data[['Property_Area']],data['Loan_Status'],test_size=0.2,random_state=0)

## Now we will just check the shape of data we got 

print("Shape of training data ",X_train.shape)
print("Shape of testing data ",X_test.shape)
print("Shape of testing data ",y_train.shape)
print("Shape of testing data ",y_test.shape)

## Now we will make a refrence objectfor the Logistic regression Class

model=LogisticRegression()

## Now lets fir our model

model.fit(X_train,y_train)

## Now we will predict the value of X_test

predict=model.predict(X_test)

## Now we will prinit the predicted value

print("the predicted value on test data : ",predict)


## Now lets check the accuracy score 

print('Accuracy score\n\n')
print(accuracy_score(y_test,predict))

## My accuracy score is 63.54166666666666%


