'''
Created on 2017. 12. 6.

@author: jaehyeong
'''
import tensorflow as tf

# training data
x_data = [[1,2,1],[1,3,2],[1,3,4],[1,5,5],[1,7,5],[1,2,5],[1,6,6],[1,7,7]]
y_data = [[0,0,1],[0,0,1],[0,0,1],[0,1,0],[0,1,0],[0,1,0],[1,0,0],[1,0,0]]
# testing data
x_test = [[2,1,1],[3,1,2],[3,3,4]]
y_test = [[0,0,1],[0,0,1],[0,0,1]]

X = tf.placeholder("float", [None, 3])
Y = tf.placeholder("float", [None, 3])
W = tf.Variable(tf.random_normal([3,3]), name='weight')
b = tf.Variable(tf.random_normal([3]), name='bias')

hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)
cost = tf.reduce_mean(-tf.reduce_sum(Y *tf.log(hypothesis), axis=1))    # cross_entropy
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
'''
learning_rate를 너무 크게하면 over-shooting되고,
너무 작게하면 중간에 step이 끝나거나, local-minima에 빠질 수 있다.
'''
# Correct prediction Test model
prediction = tf.arg_max(hypothesis, 1)
is_correct = tf.equal(prediction, tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# Launch graph
with tf.Session() as sess:
    # Initialize Tensorflow variables
    sess.run(tf.global_variables_initializer())
    for step in range(2001):
        cost_val, W_val, _ = sess.run([cost, W, optimizer],
                                      feed_dict={X: x_data, Y: y_data})
        print(step, cost_val, W_val)
        
    # predict
    print('Prediction : ',sess.run(prediction, feed_dict={X: x_test, Y: y_test}))
    # Calculate the accuracy
    print('Accuracy : ',sess.run(accuracy, feed_dict={X: x_test, Y: y_test}))
