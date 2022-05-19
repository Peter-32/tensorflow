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

def fetch_batch(epoch, batch_index, batch_size):
    X_batch = np.c_[np.ones((m, 1)), housing.data][batch_index*batch_size:batch_index*(batch_size+1)]
    y_batch = housing.target.reshape(-1, 1)[batch_index*batch_size:batch_index*(batch_size+1)]
    ss = StandardScaler()
    X_batch = ss.fit_transform(X_batch)
    if epoch % 100 == 0:
        print("Epoch:", epoch)
    return X_batch, y_batch


# Initialize
n_epochs = 1000
learning_rate = 0.01
batch_size = 100
n_batches = int(np.ceil(m / batch_size))
X = tf.placeholder(tf.float32, shape=(None, n + 1))
y = tf.placeholder(tf.float32, shape=(None, 1), name="y")
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
        for batch_index in range(n_batches):
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

    # Best theta
    best_theta = theta.eval()
print("best_theta:", best_theta)
