import numpy as np
import tensorflow.compat.v1 as tf
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
print(tf.__version__)
tf.disable_eager_execution()

# Extract data
housing = fetch_california_housing()
m, n = housing.data.shape
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]

# StandardScaler
ss = StandardScaler()
scaled_housing_data_plus_bias = ss.fit_transform(housing_data_plus_bias)

# Initialize
n_epochs = 1000
learning_rate = 0.01
X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
training_op = optimizer.minimize(mse)
init = tf.global_variables_initializer()

# Execute a TensorFlow session on the default graph
with tf.Session() as sess:
    sess.run(init)

    # Iterate on epochs
    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("Epoch", epoch, "MSE =", mse.eval())
        sess.run(training_op)

    # Best theta
    best_theta = theta.eval()
print("best_theta:", best_theta)

# Compare with LinearRegression
    # RESULT: Same result
y = housing.target.reshape(-1, 1)
lr = LinearRegression()
lr.fit(scaled_housing_data_plus_bias, y)
y_pred = lr.predict(scaled_housing_data_plus_bias)
print(lr.intercept_, lr.coef_, mean_squared_error(y, y_pred))
