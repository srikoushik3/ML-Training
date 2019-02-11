''' Demonstrates linear regression with TensorFlow '''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

# Set constants
N = 1000
learn_rate = 0.1
batch_size = 40
num_batches = 400

# Step 1: Generate input points
x = np.random.normal(size=N)
m_val = np.random.normal(loc=0.5, scale=0.2, size=N)
b_val = np.random.normal(loc=0.5, scale=0.2, size=N)
y = m_val*x + b_val

# Step 2: Create variables and placeholders
x_holder = tf.placeholder(tf.float32, shape=[batch_size])      #holds one batch of x_values
y_holder = tf.placeholder(tf.float32, shape=[batch_size])

m = tf.Variable(tf.random_normal([]))
b = tf.Variable(tf.random_norma([]))

# Step 3: Define model and loss
model = m*x_holder + b  # ma nd b are the varibles that will be changed by the optimizer
loss = tf.reduce_mean(tf.pow(model - y_holder, 2))

# Step 4: Create optimizer
optimizer = tf.train.GradientDescentOptimizer(learn_rate).minimize(loss)

# Step 5: Execute optimizer in a session
