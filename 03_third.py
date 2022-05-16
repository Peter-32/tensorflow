import tensorflow.compat.v1 as tf
print(tf.__version__)
tf.disable_eager_execution()

# Define x and y
x1 = tf.Variable(1)
print(x1.graph is tf.get_default_graph())

# Graph
graph = tf.Graph()
with graph.as_default():
    x2 = tf.Variable(2)

# Graph
print(x2.graph is graph)
print(x2.graph is tf.get_default_graph())

w = tf.constant(3)
x = w + 2
y = x + 5
z = x * 3

# eval() is an easy way to do this
with tf.Session() as sess:
    print(y.eval())
    print(z.eval())

with tf.Session() as sess:
    y_val, z_val = sess.run([y, z])
    print(y_val)
    print(z_val)
