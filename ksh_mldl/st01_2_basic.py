"""
tensorflow 동작원리
1. Build graph using Tensorflow operations
2. feed data and run graph(operations)
    sess.run(op)
3. update variables in the graph( and return values)
"""

import tensorflow as tf

# node 생성
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)

# 위의 node를 출력하면 tensor정보가 나온다
print("node1 : ", node1,"node2 : ",node2)
print("node3 : ",node3)

# session을 만들고, session을 실행시켜야 함
sess = tf.Session()
print("node1, node2 : ", sess.run([node1, node2]))
print("node3 : ", sess.run(node3))

# Placeholder -> 값을 넘겨준다
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b  # provides a shortcut for tf.add(a,b)

print(sess.run(adder_node, feed_dict={a:3, b:4.5}))
print(sess.run(adder_node, feed_dict={a:[1,3], b:[2,4]}))


