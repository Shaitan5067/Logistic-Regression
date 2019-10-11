## First we will import important libraries 

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
%matplotlib inline

## Now we will load our dataset in a variable

digits=load_digits()

## Now we will check the  shape of our data

print('Image data Shape',digits.data.shape)
print('Label data Shape',digits.target.shape)

## Now we will print the first five data of the dataset

plt.figure(figsize=(20,4))
for index,(image,label) in enumerate(zip(digits.data[0:5],digits.target[0:5])):
    plt.subplot(1,5,index+1)
    plt.imshow(np.reshape(image,(8,8)),cmap=plt.cm.gray)
    plt.title('Training%i\n'%label,fontsize=20)
    
## Now we will train test and split the data

X_train, X_test, y_train, y_test=train_test_split(digits.data,digits.target,test_size=0.3,random_state=0)

## Now we just want to see the shape of aur train an test variable

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


## Create a object for Logistic Regression Class

model=LogisticRegression()

## Now we will train our data 

model.fit(X_train,y_train)

## Now lets predict on the X_test data 

predictions=model.predict(X_test)

print(predictions)  

## Now we will check the accuracy score 

score=model.score(X_test,y_test)

## In this my score is 95% means my model is good model and predicting well
