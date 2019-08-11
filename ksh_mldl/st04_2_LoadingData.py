'''
Created on 2017. 7. 18.

@author: anjh

numpy로 파일 읽어와서, 
exam1, exam2, exam3 시험을 통해 final점수를 받았을 때, 이것을 학습하여 
나중 다른 시험의 미래 점수 예측하기
'''
import numpy as np
import tensorflow as tf
tf. set_random_seed(777)    # for reproducibility

# 파일로드 
xy = np.loadtxt('./data/data-01-test-score.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]    # 행은 전체고, 열은 마지막 열(final)만 뺀 데이터(exam1,exam2,exam3)
y_data = xy[:, [-1]]     # 행은 전체고, 열은 마지막 열(final)

# make sure the shape and data are OK
#print(x_data.shape, x_data) # (25, 3)
#print(y_data.shape, y_data) # (25, 1) 

#placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([3,1]), name='weight')     # tf.normal([입력값, 출력값])
b = tf.Variable(tf.random_normal([1]))                      # tf.normal([출력값])

# Hypothesis
hypothesis = tf.matmul(X, W) + b
# simplified cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))
 
# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

# Launch the graph in a session
sess = tf.Session()
# Initializes global variables in the graph
sess.run(tf.global_variables_initializer())
# Set up feed_dict variables inside the loop
for step in range(2001):
    cost_val, hy_val, _ = sess.run(
        [cost, hypothesis, train],
        feed_dict={X: x_data, Y: y_data})
    if step % 10 == 0:
        print(step, "Cost: ", cost_val, "\nPrediction: ",hy_val)

# Ask my score
print("Your score will be ",sess.run(hypothesis,
                                     feed_dict={X: [[100,70,101]]}))
print("Other score will be ",sess.run(hypothesis,
                                      feed_dict={X: [[60,70,110],[90,100,80]]}))




