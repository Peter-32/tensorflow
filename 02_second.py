import tensorflow.compat.v1 as tf
print(tf.__version__)
tf.disable_eager_execution()


# Define x and y
x = tf.Variable(3, name="x")
y = tf.Variable(4, name = "y")

# Start a session
with tf.Session() as sess:
    sess.run(x.initializer)
    sess.run(y.initializer)

    # Define f
    f = x*x*y + y + 2

    result = f.eval()
print(result)
sess.close()
