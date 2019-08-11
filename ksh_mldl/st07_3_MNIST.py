'''
Created on 2017. 12. 6.

@author: jaehyeong
'''
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
nb_classes = 10     # 0 ~ 9 digits recognition

# MNIST data image of shape 28 * 28 = 784
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, nb_classes])

W = tf.Variable(tf.random_normal([784, nb_classes]))
b = tf.Variable(tf.random_normal([nb_classes]))

# hypothesis(using softmax)
hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)

cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# Test model
is_correct = tf.equal(tf.argmax(hypothesis,1), tf.argmax(Y,1))
# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# parameters
epochs = 30         # 1 epoch = 전체 training 데이터셋을 한번 학습
batch_size = 100    # 한번에 몇개씩 학습시킬지

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    # training cycle
    for epoch in range(epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)
        
        for i in range(total_batch):    # batch 반복 횟수
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            c, _ = sess.run([cost, optimizer], feed_dict={X: batch_xs, Y: batch_ys})
            avg_cost += c / total_batch
            
        print('Epoch: ',epoch+1,' cost =',avg_cost)
        # test the model using test sets
        print('Accuracy: ',sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))
        

    # Sample image show and prediction 
    import matplotlib.pyplot as plt
    import random
    
    # get one and predict
    r = random.randint(0, mnist.test._num_examples -1)
    print('label: ',sess.run(tf.argmax(mnist.test.labels[r:r+1], 1)))
    print('prediction: ',sess.run(tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r:r+1]}))
    
    plt.imshow(mnist.test.images[r:r+1].reshape(28, 28), cmap='Greys', interpolation='nearest')
    plt.show()

