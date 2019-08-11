'''
Simple convolution layer

3x3x1 image
2x2x1 filter w
stride 1x1
'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

sess = tf.InteractiveSession()
image = np.array([[[[1],[2],[3]],
                   [[4],[5],[6]],
                   [[7],[8],[9]]]], dtype=np.float32)

print('image.shape : ',image.shape)      # (1, 3, 3, 1)
plt.imshow(image.reshape(3,3), cmap='Greys')
plt.show()

# sampling(filter : 2,2,1,1, stride : 1x1, padding : VALID)
weight = tf.constant([[[[1.]],[[1.]]],
                      [[[1.]],[[1.]]]])
print('weight.shape : ',weight.shape)
 
# stride 적용
conv2d = tf.nn.conv2d(image, weight, strides=[1, 1, 1, 1], padding='VALID')
conv2d_img = conv2d.eval()
print('conv2d_img.shape : ', conv2d_img.shape)

conv2d_img = np.swapaxes(conv2d_img, 0, 3)
for i, one_img in enumerate(conv2d_img):
    print(one_img.reshape(2,2))
    plt.subplot(1,2,i+1), plt.imshow(one_img.reshape(2,2), cmap='gray')
    plt.show()


################################################################################
# sampling(filter : 2,2,1,1, stride : 1x1, padding : SAME)
# padding='SAME'  -> 입력의 사이즈와 출력되는 convolution 사이즈가 같게함   즉, 3x3x1 -> 3x3x1
weight = tf.constant([[[[1.]],[[1.]]],
                      [[[1.]],[[1.]]]])
print('weight.shape : ',weight.shape)
 
# stride 적용
conv2d = tf.nn.conv2d(image, weight, strides=[1, 1, 1, 1], padding='SAME')
conv2d_img = conv2d.eval()
print('conv2d_img.shape : ', conv2d_img.shape)

conv2d_img = np.swapaxes(conv2d_img, 0, 3)
for i, one_img in enumerate(conv2d_img):
    print(one_img.reshape(3,3))
    plt.subplot(1,3,i+1), plt.imshow(one_img.reshape(3,3), cmap='gray')
    plt.show()



################################################################################
# 3 filters(2,2,1,3)
weight = tf.constant([[[[1.,10.,-1.]],[[1.,10.,-1.]]],
                      [[[1.,10.,-1.]],[[1.,10.,-1.]]]])
print('weight.shape : ',weight.shape)
 
# stride 적용
conv2d = tf.nn.conv2d(image, weight, strides=[1, 1, 1, 1], padding='SAME')
conv2d_img = conv2d.eval()
print('conv2d_img.shape : ', conv2d_img.shape)

conv2d_img = np.swapaxes(conv2d_img, 0, 3)
for i, one_img in enumerate(conv2d_img):
    print(one_img.reshape(3,3))
    plt.subplot(1,3,i+1), plt.imshow(one_img.reshape(3,3), cmap='gray')
    plt.show()

plt.imshow(conv2d_img.reshape(3,3), cmap='Greys')
plt.show()
