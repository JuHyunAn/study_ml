'''
H(x) = W(x) + b
'''
import tensorflow as tf
# 1. Build graph using TF operations
# X and Y data
x_train = [1,2,3]
y_train = [1,2,3]

W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')
'''
cost(W,b)
'''
#Our hypothesis WX+b
hypothesis = x_train * W + b

# cost/loss functions
cost = tf.reduce_mean(tf.square(hypothesis - y_train))
print(cost)
# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

# 2. Run/Update graph and get results
# Launch the graph in a session
sess = tf.Session()
# Initializes global variables in the graph
sess.run(tf.global_variables_initializer())

# Fit the line
for step in range(2001):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(cost), sess.run(W), sess.run(b))
        
        
'''        
# 위의 과정을 placeholders를 이용해서 구현
# Now we can use X and Y in place of x_data and y_data
# placeholders for a tensor that will be always fed using feed_dict
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

#Our hypothesis WX+b
hypothesis = x_train * W + b

# cost/loss functions
cost = tf.reduce_mean(tf.square(hypothesis - y_train))
print(cost)
# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

# 2. Run/Update graph and get results
# Launch the graph in a session
sess = tf.Session()
# Initializes global variables in the graph
sess.run(tf.global_variables_initializer())

# Fit the line with new training data
for step in range(2001):
    cost_val, W_val, b_val,_ = sess.run([cost, W, b, train],
                    feed_dict={X: [1,2,3,4,5],
                               Y:[2.1,3.1,4.1,5.1,6.1]})
    if step % 20 == 0:
        print(step, cost_val, W_val, b_val)
'''



