'''
Created on 2017. 7. 18.

@author: anjh

tensorflow로 Queue Runners형식으로 파읽읽어오기
'''
import tensorflow as tf

# 데이터 읽어오기 ( 여러개의 파일을 리스트로 읽을 수 있다)
filename_queue = tf.train.string_input_producer(['data-01-test-score.csv'], shuffle=False,
                                                name = 'filename_queue')
reader = tf.TextLineReader()    # 텍스트 라인 읽기
key, value = reader.read(filename_queue)    # key, value 

# Default values, in case of empty columns. Also specifies the type of the decoded result
record_defaults = [[0.],[0.],[0.],[0.]] # 각 필드의 데이터 타입 형태
xy = tf.decode_csv(value, record_defaults=record_defaults)

# collect batches of csv in
train_x_batch, train_y_batch = tf.train.batch([xy[:,0:-1], xy[:[-1]], batch_size=10)   # 한번에 10개씩 가져와라 

#placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 3]) # X 갯수
Y = tf.placeholder(tf.float32, shape=[None, 1]) # Y 갯수

W = tf.Variable(tf.random_normal([3,1]), name='weight')     # tf.normal([x갯수, y갯수])
b = tf.Variable(tf.random_normal([1]))                      # tf.normal([y갯수])

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

# Start populating the filename queue
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

for step in range(2001):
    x_batch, y_batch = sess.run([train_x_batch, train_y_batch])
    cost_val, hy_val, _ = sess.run(
        [cost, hypothesis, train],
        feed_dict={X: x_data, Y: y_data})
    if step % 10 == 0:
        print(step, "Cost: ", cost_val, "\nPrediction: ",hy_val)
        
coord.request_stop()
coord.join(threads)




'''
# shuffle_batch
min_after_dequeue = 10000
capacity = min_after_dequeue + 3 * batch_size
example_batch, label_batch = tf.train.shuffle_batch(
    [example, label], batch_size=batch_size, capacity = capacity,
    min_after_dequeue = min_after_dequeue)
'''                                              



