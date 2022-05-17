import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

# Example 1
a = tf.constant(5, name="a")
with tf.Session() as sess:
    result1 = a.eval()

# Example 2
a = tf.Variable(6, name="a")
b = a * 2
c = b * 2
with tf.Session() as sess:
    sess.run(a.initializer)
    result2 = c.eval()

# Print results
print(result1, result2)
