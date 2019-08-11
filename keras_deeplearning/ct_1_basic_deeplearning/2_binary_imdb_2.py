import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import imdb
from keras import models, layers
from keras import optimizers, losses, metrics

(train_data, train_labels),(test_data, test_labels) = imdb.load_data(num_words=10000) # 자주 나타나는 단어 1만개만 사용

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
#print(x_train[0])
#print(y_train[0])

# validation set
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

# model define
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(16, activation='relu')) # layer 추가 
model.add(layers.Dense(1, activation='sigmoid'))

# model compile
model.compile(optimizer='rmsprop',
			loss = 'mse',			# mean squared error
			metrics=['accuracy'])
# fit
model.fit(x_train, y_train, epochs=4, batch_size=512) # epochs을 4로 제한 

# test set에 평가
results = model.evaluate(x_test, y_test)
print(results)	# [0.2945874176502228, 0.8832]

# predict
pred = model.predict(x_test)
#print(pred)