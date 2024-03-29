'''
Created on 2017. 7. 25.

@author: jaehyeong
'''
import numpy as np
import tensorflow as tf

xy = np.loadtxt('../data/data-03-diabetes.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]    # 행은 전체, 열은 처음부터 마지막 전까지
y_data = xy[:, [-1]]    # 행은 전체, 열은 마지막 열만

# placeholders for a tensor will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 8])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([8,1]), name='weight') # [8, 1] = [들어오는 값(X), 나가는 값(Y)]
b = tf.Variable(tf.random_normal([1]), name='bias')     # [1] = [나가는값(Y)]

# Hypothesis using sigmoid
hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

# cost/loss function                                                                                           
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1- hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# Accuracy computation
# True if hypothesis > 0.5 else False
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32) # 0.5보다 크면 1, 작으면 0으로 변환
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))    # 예측한 값과 실제 값을 비교하여 평균

# Train the model
# Launch graph
with tf.Session() as sess:
    # Initialize Tensorflow variables
    sess.run(tf.global_variables_initializer())
    
    for step in range(10001):
        cost_val, _ = sess.run([cost, train], feed_dict={X: x_data, Y: y_data})
        if step % 500 == 0:
            print(step, cost_val)

    # Accuracy report
    h, c, a = sess.run([hypothesis, predicted, accuracy],
                       feed_dict={X: x_data, Y: y_data})
    print("\nHypothesis : ",h, "\nPredict(Y) : ",c, "\nAccuracy : ",a)


