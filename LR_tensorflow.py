import numpy as np
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression as LR
from utils import load_data, create_train_test_data
from sklearn.metrics import mean_squared_error
import os, math

user_dataset_path = "data/user_dataset/"

def load_user_dataset(user_dataset_path):
    user_dataset_files = [f for f in os.listdir(user_dataset_path) if f.endswith('.pickle')]
    dataset = []
    for fin in user_dataset_files:
        dataset += load_data(user_dataset_path + fin)

    return dataset

dataset = load_user_dataset(user_dataset_path)
train_X, train_Y, test_X, test_Y = create_train_test_data(dataset)

import tensorflow as tf

rng = np.random
learning_rate = 0.01
training_epochs = 20
display_step = 1

n_samples = len(train_X)
n_features = len(train_X[0])
print(n_samples, n_features)

# tf Graph Input
X = tf.placeholder("float")
Y = tf.placeholder("float")

# Set model weights
W = tf.Variable(tf.random_normal([n_features, 1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name="bias")

# Construct a linear model
pred = tf.add(tf.multiply(X, W), b)

# Mean squared error
cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)
# Gradient descent
#  Note, minimize() knows to modify W and b because Variable objects are trainable=True by default
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
done = False
with tf.Session() as sess:
    sess.run(init)

    # Fit all training data
    for epoch in range(training_epochs):
        for k, x in enumerate(train_X):
            y = train_Y[k]
            if not done:
                print(x, y)
                done = True
            sess.run(optimizer, feed_dict={X: x, Y: y})

        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            c = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), \
                "W=", sess.run(W), "b=", sess.run(b))

    print("Optimization Finished!")
    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
    print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')
    print("Testing... (Mean square loss Comparison)")
    testing_cost = sess.run(
        tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * test_X.shape[0]),
        feed_dict={X: test_X, Y: test_Y})  # same function as cost above
    print("Testing cost=", testing_cost)
    print("Absolute mean square loss difference:", abs(
        training_cost - testing_cost))


