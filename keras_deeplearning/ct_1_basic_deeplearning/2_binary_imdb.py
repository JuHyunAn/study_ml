import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import imdb
from keras import models, layers
from keras import optimizers, losses, metrics

(train_data, train_labels),(test_data, test_labels) = imdb.load_data(num_words=10000) # 자주 나탄나는 단어 1만개만 사용

print(train_data[0])
print(train_labels[0])

# ## 리뷰 데이터를 다시 원래의 영어 단어로 변환 ##
# word_index = imdb.get_word_index() # 단어와 정수 인덱스를 매핑한 dict
# reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
# decoded_review = ''.join([reverse_word_index.get(i-3,'?') for i in train_data[0]])
# print(decoded_review)

# 정수 시퀀스를 이진 행렬로 인코딩하기
def vectorize_sequence(sequences, dimension=10000):
	results = np.zeros((len(sequences), dimension))

	for i, sequences in enumerate(sequences):
		results[i, sequences] = 1
	return results

# vector 변환
x_train = vectorize_sequence(train_data) 
x_test = vectorize_sequence(test_data)
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')
print(x_train[0])
print(y_train[0])

# validation set
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

# model define
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# model compile
model.compile(optimizer='rmsprop',
			loss = 'binary_crossentropy',
			metrics=['accuracy'])

# fit
history = model.fit(partial_x_train,
				partial_y_train,
				epochs=20,
				batch_size=512,
				validation_data=(x_val, y_val))

# fit() -> history 객체 반환 
history_dict = history.history
print(history_dict.keys()) # dict_keys(['val_loss', 'val_acc', 'loss', 'acc'])

# 훈련과 검증에 대한 손실 plot
loss = history_dict['loss']
val_loss = history_dict['val_loss']
epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')			# 'bo' : 파란색 점
plt.plot(epochs, val_loss, 'b', label='Validation loss')	# 'b' : 파란색 실선
plt.title('Training and Validation loss')
plt.xlabel('Epochs'); plt.ylabel('Loss')
plt.legend(); plt.show()

# 정확도 plot
acc = history_dict['acc']
val_acc = history_dict['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs'); plt.ylabel('Accuracy')
plt.legend(); plt.show()
