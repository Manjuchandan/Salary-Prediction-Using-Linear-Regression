#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Importing libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[4]:


# Loading of data

dataset = pd.read_csv('salary_data.csv')

X = dataset.iloc[:, :-1].values

y = dataset.iloc[:,1].values

#dataset: the table contains all values in our csv file

#X: the first column which contains Years Experience array

#y: the last column which contains Salary array


# In[5]:


# Split data into training and testing


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)


#test_size=1/3: we will split our dataset (30 observations) into 2 parts (training set, test set) and the ratio of test set compare to dataset is 1/3 (10 observations will be put into the test set. You can put it 1/2 to get 50% or 0.5, they are the same. We should not let the test set too big; if it’s too big, we will lack of data to train. Normally, we should pick around 5% to 30%.

#train_size: if we use the test_size already, the rest of data will automatically be assigned to train_size.

#random_state: this is the seed for the random number generator. We can put an instance of the RandomState class as well. If we leave it blank or 0, the RandomState instance used by np.random will be used instead.

#We already have the train set and test set, now we have to build the Regression Model:


# In[6]:


#Fit Simple Linear Regression to Training Data

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train, y_train)

#regressor = LinearRegression(): our training model which will implement the Linear Regression.

#regressor.fit: in this line, we pass the X_train which contains value of Year Experience and y_train which contains values of particular Salary to form up the model. This is the training process.


# In[7]:


#Make Prediction

y_pred = regressor.predict(X_test)


# In[8]:


# Visualize training set results

import matplotlib.pyplot as plt

# plot the actual data points of training set
plt.scatter(X_train, y_train, color = 'red')

# plot the regression line
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


# In[9]:


# Visualize test set results

import matplotlib.pyplot as plt

# plot the actual data points of test set

plt.scatter(X_test, y_test, color = 'red')

# plot the regression line (same as above)
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


# In[10]:


# Make new prediction

new_salary_pred = regressor.predict([[10]])
print('The predicted salary of a person with 15 years experience is ',new_salary_pred)


# In[ ]:


# THANK YOU

