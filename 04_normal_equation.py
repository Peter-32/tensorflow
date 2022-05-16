import numpy as np
import tensorflow.compat.v1 as tf
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing
print(tf.__version__)
tf.disable_eager_execution()

housing = fetch_california_housing()
m, n = housing.data.shape
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]

# Initialize
X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
XT = tf.transpose(X)
theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)

# Execute a TensorFlow session on the default graph
with tf.Session() as sess:
    theta_result = theta.eval()
print(theta_result)

# Compare with LinearRegression
    # RESULT: Same result
lr = LinearRegression()
lr.fit(housing_data_plus_bias, housing.target)
print(lr.intercept_, lr.coef_)
