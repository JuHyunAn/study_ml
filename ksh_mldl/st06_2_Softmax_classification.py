'''
Created on 2017. 12. 5.

@author: jaehyeong
'''
import numpy as np
import tensorflow as tf

# predicting animal type based on various features
xy = np.loadtxt('./data/data-04-zoo.csv', delimiter=',', dtype=np.float32)
x_data = xy[:,0:-1]
y_data = xy[:,[-1]]

nb_classes = 7  # 0 ~ 6

X = tf.placeholder(tf.float32, [None, 16])
Y = tf.placeholder(tf.int32, [None, 1])   # 0 ~ 6

Y_one_hot = tf.one_hot(Y, nb_classes)   # 1 ~ 7 차원이 더해짐 
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes]) # 0 ~ 6 다시 원하는 형태로 reshape

W = tf.Variable(tf.random_normal([16, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

# softmax = exp(logits) / reduce_sum(exp(logits), dim)
logits = tf.matmul(X, W) + b
hypothesis = tf.nn.softmax(logits)

# Cross entropy cost function
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                 labels=Y_one_hot))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# predict
prediction = tf.argmax(hypothesis, 1)
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))  # 예측값과 실제레이블 비교
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Launch graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(2000):
        sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
        if step % 100 == 0:
            loss, acc = sess.run([cost, accuracy], feed_dict={X: x_data, Y: y_data})
            
            print('Step:',step,', loss:',loss,', Acc:',acc)
            
    # see if we can predict
    pred = sess.run(prediction, feed_dict={X: x_data})
    # y_data: (N,1) = flatten => (N, ) matches pred.shape
    for p, y in zip(pred, y_data.flatten()):
        print('[{}] Prediction: {} True T: {}'.format(p==int(y), p, int(y)))
    
    
    
