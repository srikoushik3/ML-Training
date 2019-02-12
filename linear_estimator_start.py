''' Demonstrate how estimators can be used for regression '''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

#notes:
'''
Estimators

 - easy to deploy to the cloud

Estimator Classes:

Feature Columns - for each type of data in your dataset
 - first argument of an estimator constructor
 - has dense columns and categorical columns
 - dense columns:
    1) Numeric Columns - numerical values
    - tf.feature_column.numeric_column(key)
 - categprical columns:
    1) Identiy Categorical Columns - categories identified by integers starting from zero
    - tf.feature_column.categorical_column_with_identity(key, num_buckets, default_value=None)
    - ex. directions north(0), south(1), east(2), west(3)

Input Functions - No arguments, training or testing

    For training or testing:
     - returns a tuple containing features and labels
     - each feature associates column keys with data
    For Predictions:
     - returns features for prediction
'''

# Define constants
N = 1000
num_steps = 800

# Step 1: Generate input points
x_train = np.random.normal(size=N)
m = np.random.normal(loc=0.5, scale=0.2, size=N)
b = np.random.normal(loc=1.0, scale=0.2, size=N)
y_train = m * x_train + b

# Step 2: Create a feature column
x_col = tf.feature_column.numeric_column('x_coords')

# Step 3: Create a LinearRegressor

# Step 4: Train the estimator with the generated data

# Step 5: Predict the y-values when x equals 1.0 and 2.0
