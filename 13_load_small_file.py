import numpy as np
import tensorflow.compat.v1 as tf
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
print(tf.__version__)
tf.disable_eager_execution()

housing = fetch_california_housing()
m, n = housing.data.shape
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name="theta")
init = tf.global_variables_initializer()
saver = tf.train.import_meta_graph("./tmp/my_model_final.ckpt.meta")

# Execute a TensorFlow session on the default graph
with tf.Session() as sess:
    sess.run(init)
    saver.restore(sess, "./tmp/my_model_final.ckpt")
    best_theta = theta.eval()
print("best_theta:", best_theta)
