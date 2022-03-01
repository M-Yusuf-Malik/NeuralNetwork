import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from NeuralNetwork import model, predict

# import warnings
# warnings.filterwarnings("ignore")  # suppress warnings


# Step 1: Getting the data and cleaning it
# 1
data = pd.read_csv('nba.csv')
x = data

x = x.drop(columns='Name')
x = x.drop(columns='TARGET_5Yrs')

y = data['TARGET_5Yrs']

# Step 2: Splitting the data into x, y train and test which are (list or np array)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=2)


# Step 2.2: Standardizing the dataset (Optional but should do it)
sc = StandardScaler()
sc.fit(x_train)
x_train = sc.transform(x_train)
x_test = sc.transform(x_test)


# Step 3: Shaping the arrays into the specific required shape i.e., (num_features, examples)
x_train = x_train.T
y_train = np.array(y_train).reshape(y_train.shape[0], 1).T
x_test = x_test.T
y_test = np.array(y_test).reshape(y_test.shape[0], 1).T

# Step 4: Setting the hyper parameters
num_features = x_train.shape[0]
dims = [num_features, 5, 1]

learning_rate = 0.0007
iterations = 200
lambd = 0  # For the regularization
mini_batch_size = 64

# Step 5: Giving the required output and training the model
paras, cache, cost = model(x_train, y_train, dims,
                           learning_rate, iterations, lambd, mini_batch_size)

L = len(dims) - 1

# Step 6: Getting the accuracy of training and testing to analyze the model
print("\nTraining accuracy is: ", predict(
    x_train, y_train, paras, cache, L), "\n")
print("Testing accuracy is: ", predict(x_test, y_test, paras, cache, L))

# Step 6.2: Plotting the cost function to catch any abnormal behaviour
plt.plot(range(iterations), cost)
plt.show()

# Viola - Ended the Neural Network :)
