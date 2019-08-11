import tensorflow as tf

# Create a constant op
#This op is added as a node to the dafault graph
hello = tf.constant("Hello, Tensorflow!")

# seart a TF session
sess = tf.Session()

# run the op and get result
print(sess.run(hello))  # b'Hello, Tensorflow!'

