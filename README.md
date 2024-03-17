# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Neural networks consist of simple input/output units called neurons (inspired by human brain neurons). These input/output units are interconnected and each connection has a weight associated with it. Neural networks are flexible and can be used for both classification and regression. In this article, we will see how neural networks can be applied to regression problems.

Regression helps establish a relationship between a dependent variable and one or more independent variables. Regression models work well only when the regression equation is a good fit for the data. Most regression models will not fit the data perfectly. Although neural networks are complex and computationally expensive, they are flexible and can dynamically pick the best type of regression, and if that is not enough, hidden layers can be added to improve prediction.

First import the libraries that we will use Import the dataset and check the types of the columns Now build your training and test set from the dataset Here we are making the neural network 2 hidden layers with 1 output and input layer and an activation layer as relu and with their nodes in them. Now we will fit our dataset and then predict the value.

## Neural Network Model

![Screenshot 2024-02-21 140031](https://github.com/etjabajasphin/basic-nn-model/assets/151705853/b468976c-c7e3-49ff-b5a3-f2a066b3cfe7)

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: KARUNIYA M
### Register Number: 212223240068
```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from google.colab import auth
import gspread
from google.auth import default
auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)
worksheet = gc.open('clinical').sheet1
data = worksheet.get_all_values()
dataset1 = pd.DataFrame(data[1:], columns=data[0])
dataset1 = dataset1.astype({'actual':'float'})
dataset1 = dataset1.astype({'predicted':'float'})
dataset1.head()
X = dataset1[['actual']].values
y = dataset1[['predicted']].values
X
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.33,random_state = 33)
Scaler = MinMaxScaler()
Scaler.fit(X_train)
X_train1 = Scaler.transform(X_train)
ai_brain = Sequential([
    Dense(6,activation = 'relu'),
    Dense(6,activation = 'relu'),
    Dense(1)
])
ai_brain.compile(optimizer = 'rmsprop', loss = 'mse')
ai_brain.fit(X_train1,y_train,epochs = 4000)
loss_df = pd.DataFrame(ai_brain.history.history)
loss_df.plot()
X_test1 = Scaler.transform(X_test)
ai_brain.evaluate(X_test1,y_test)
X_n1 = [[30]]
X_n1_1 = Scaler.transform(X_n1)
ai_brain.predict(X_n1_1)

```
## Dataset Information
![Screenshot 2024-02-21 141344](https://github.com/etjabajasphin/basic-nn-model/assets/151705853/6ec9fd30-2cdb-452a-b4c5-c6d5989ed869)

## OUTPUT

### Training Loss Vs Iteration Plot
![Screenshot 2024-02-21 141255](https://github.com/etjabajasphin/basic-nn-model/assets/151705853/82360829-4326-412e-b549-3eeda78c092b)

### Test Data Root Mean Squared Error
![Screenshot 2024-02-21 142334](https://github.com/etjabajasphin/basic-nn-model/assets/151705853/0068d0ed-c280-4cbf-ada0-e5ddf03e44a7)


### New Sample Data Prediction
![Screenshot 2024-02-21 142347](https://github.com/etjabajasphin/basic-nn-model/assets/151705853/29d51e6e-8282-4df1-a222-6a2dc0909294)



## RESULT

A neural network regression model for the given dataset has been developed Sucessfully.
