from keras.datasets import mnist
from keras.utils import to_categorical
from keras import models
from keras import layers

# load_data
(train_images, train_labels),(test_images, test_labels) = mnist.load_data()
print(train_images.shape)
print(test_images.shape)

# 0 ~ 1 사이의 값으로 
train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32') / 255

# categorize
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# network
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28*28, )))
network.add(layers.Dense(10, activation='softmax'))

# compile
network.compile(optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy'])

network.fit(train_images, train_labels, epochs=5, batch_size=128)

test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_loss: ',test_loss)
print('test_acc: ',test_acc)