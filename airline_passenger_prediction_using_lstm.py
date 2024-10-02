#!/usr/bin/env python
# coding: utf-8

# # AIRLINE PASSENGER (TIME SERIES) PREDICTION USING LSTM
# 
# We will use Airline Passenger dataset for this project. This dataset provides monthly totals of a US airline passengers from 1949 to 1960. You can download the dataset from Kaggle link below: 
# https://www.kaggle.com/chirag19/air-passengers
# 
# We will use LSTM deep learning model for this project. The Long Short-Term Memory network, or LSTM network, is a recurrent neural network that is trained using Backpropagation through time and overcomes the vanishing gradient problem. LSTM can be used to create large recurrent networks that in turn can be used to address difficult sequence problems in machine learning and achieve state-of-the-art results. Instead of neurons, LSTM networks have memory blocks that are connected through layers.
# 
# #### Aim of the project:
# 
# Given the number of passengers (in units of thousands) for last two months, what is the number of passengers next month? In order to solve this problem we will build a LSTM model and train this model with our train data which is first 100 months in our dataset. After the LSTM model training finishes and learn the pattern in time series train data, we will ask it the above question  question and get the answer from it.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense,LSTM  # I use keras over Tensorflow 2, so I don't need extra Keras libraries.
from tensorflow.keras.models import Sequential  # Tensorflow includes Keras libraries after 2nd version of Tensorflow
from sklearn.preprocessing import MinMaxScaler  # If you have Tensorflow 1, please upgrade your library using:
                                                # conda install tensorflow
                                                # This command will upgrade your Tensorflow module to the latest version.


# In[ ]:


# You can downlad the dataset from Kaggle link below:
# https://www.kaggle.com/chirag19/air-passengers

data = pd.read_csv('AirPassengers.csv')
data.head()


# In[ ]:


data.rename(columns={'#Passengers':'passengers'},inplace=True)
# Since this is a time series, we need only second column.. So data now contains only passenger count...
data = data['passengers']


# In[ ]:


type(data)


# In[ ]:


data


# My data fromat is Series, but I need 2D array for MinMaxScaler() and my other methods to work. So I will change to numpy array and reshape it.

# In[ ]:


data=np.array(data).reshape(-1,1)


# In[ ]:


# ok, now we have 2D numpy array...
type(data)


# In[ ]:


# Lets plot our data:
plt.plot(data)
plt.show()


# ### Scaling..
# 
# LSTM is sensitive to the scale of the input data. So we will rescale the data to the range of 0-to-1, also called normalizing. 

# In[ ]:


scaler = MinMaxScaler()
data = scaler.fit_transform(data)


# ### Train, Test split

# In[ ]:


len(data)


# I have 144 data. I will use 100 of it as train set and 44 as test set..

# In[ ]:


train = data[0:100,:]
test = data[100:,:]


# We will now define a function to prepare the train and test datasets for modeling. The function takes two arguments: the dataset, which is a NumPy array that we want to convert into a dataset, and the steps, which is the number of previous time steps to use as input variables to predict the next time period.
# 
# 

# In[ ]:


def get_data(data, steps):      
    dataX = []
    dataY = []
    for i in range(len(data)-steps-1):
        a = data[i:(i+steps), 0]
        dataX.append(a)
        dataY.append(data[i+steps, 0])
    return np.array(dataX), np.array(dataY)


# So using this "get_data" function I will prepare a dataset for modeling... Then I give this new prepared datset to my model for training...

# In[ ]:


steps = 2


# #### Now I'm making my datasets for both training and testing..

# Important: You must have numpy version 1.19 in your Anaconda environment for LSTM work. If you have a error like "NotImplementedError: Cannot convert a symbolic Tensor (lstm/strided_slice:0) to a numpy array." you must change your numpy version to 1.19 using this commnad:
# 
# conda install numpy=1.19

# In[ ]:


X_train, y_train = get_data(train, steps)
X_test, y_test = get_data(test, steps)



# In[ ]:


# Im reshaping my sets for using in LSTM model..
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))


# In[ ]:


# I will use a Sequential model with 2 hidden layers
# Instead of neurons, LSTM networks have memory blocks that are connected through layers.
# The default sigmoid activation function is used for the LSTM blocks. 

model = Sequential()
model.add(LSTM(128, input_shape = (1, steps)))  # This is my first hidden layer with 128 memory blocks
model.add(Dense(64))                                  # This is my second hidden layer with 64 memory blocks
model.add(Dense(1))   # This is my output layer
model.compile(loss = 'mean_squared_error', optimizer = 'adam')


# In[ ]:


model.summary()


# ### Now it's time to train our model...

# In[ ]:


model.fit(X_train, y_train, epochs=25, batch_size=1)


# ### Let's make prediction..

# In[ ]:


y_pred = model.predict(X_test)


# We should rescale the prediction results, because our model gives us scaled predictions..

# In[ ]:


y_pred = scaler.inverse_transform(y_pred)
y_test = y_test.reshape(-1, 1)
y_test = scaler.inverse_transform(y_test)


# ### Now plot the test set results... Remember our test set contains last 44 data in original dataset..

# In[ ]:


# plot real number of passengers and predictions...
plt.plot(y_test, label = 'real number of passengers')
plt.plot(y_pred, label = 'predicted number of passengers')
plt.ylabel('Months')
plt.ylabel('Number of passengers')
plt.legend()
plt.show()


# In[ ]:





# In[ ]:




