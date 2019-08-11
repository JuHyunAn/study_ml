import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import reuters
from keras.utils.np_utils import to_categorical
from keras import models, layers

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000) # 높은 빈도 단어 1만개로 제한
print(len(train_data))  # 8982
print(len(test_data))	# 2246
print(len(np.unique(train_labels)))	# 46개의 토픽
print(train_labels) 

# vectorize
def vectorize_sequences(sequences, dimension=10000):
	results = np.zeros((len(sequences), dimension))
	for i, sequences in enumerate(sequences):
		results[i, sequences] = 1
	return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

# one-hot encoding
def to_one_hot(labels, dimension=46):
	results = np.zeros((len(labels), dimension))
	for i, labels  in enumerate(labels):
		results[i, labels] = 1
	return results

one_hot_train_labels = to_one_hot(train_labels)
one_hot_test_labels = to_one_hot(test_labels)
# one_hot_train_labels = to_categorical(train_labels) # 위 함수와 동일한 결과 
# one_hot_test_labels = to_categorical(test_labels)

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

model.compile(optimizer='rmsprop', 
	loss='categorical_crossentropy',
	metrics=['accuracy'])

# validation set
x_val = x_train[:1000]
partial_x_train = x_train[1000:]
y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

history = model.fit(partial_x_train, partial_y_train,
					epochs=9, batch_size=512, validation_data=(x_val, y_val))

# plot
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1,len(loss)+1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs'); plt.ylabel('Loss')
plt.legend(); plt.show()

acc = history.history['acc']
val_acc = history.history['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs'); plt.ylabel('Accuracy')
plt.legend(); plt.show()

# evaluate
results = model.evaluate(x_test, one_hot_test_labels)
print(results)

# predict
pred = model.predict(x_test)
print(pred[0])
print(np.sum(pred[0]))	# softmax의 원소 합은 1
print(np.argmax(pred[0]))	# 확률이 가장 큰 값이 예측 클래스 