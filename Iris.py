# import important libraries

import pandas as pd
from sklearn import datasets,linear_model
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


# load Iris dataset with the help of pandas


data=pd.read_csv("Iris.csv")
print(data.head())

#display  column names

print('Column Names \\n\\n')
print(data.columns)

# encode the class name with the help of label encoding


encode=LabelEncoder()
data['class:']=encode.fit_transform(data['class:'])
print(data.head(10))


#train test and split the data ,random state we use for same data otherwise it will always use different data when we run the code

train,test=train_test_split(data,test_size=0.2)
print('shape of training data ',train.shape)
print('shape of test data ',test.shape)

train_x=train.drop(columns=['class:'],axis=1)
train_y=train['class:']

test_x=test.drop(columns=['class:'],axis=1)
test_y=test['class:']

#now we will make an object of Logistic Regression class

model=LogisticRegression()

#Now we will fit our model with train_x and train_y

model.fit(train_x,train_y)

#Now we will predict the model with help of test_x

predict=model.predict(test_x)

print(predict)


print('Predicted Values on Test Data',encode.inverse_transform(predict))

#By accuracy_score we can check themodel accuracy

print(accuracy_score(test_y,predict))


