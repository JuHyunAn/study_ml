# Dropout 층 추가 

import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import imdb
from keras import models, layers
from keras import optimizers, losses, metrics

(train_data, train_labels),(test_data, test_labels) = imdb.load_data(num_words=10000) # 자주 나탄나는 단어 1만개만 사용

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

# original model 
original_model = models.Sequential()
original_model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
original_model.add(layers.Dense(16, activation='relu'))
original_model.add(layers.Dense(1, activation='sigmoid'))
original_model.compile(optimizer='rmsprop',
                       loss='binary_crossentropy',
                       metrics=['accuracy'])

original_hist = original_model.fit(x_train, y_train,
									epochs=20, batch_size=512, validation_data=(x_test, y_test))

# Drop out 추가한 모델 
dropout_model = models.Sequential()
dropout_model.add(layers.Dense(16, activation='relu',input_shape=(10000,)))
dropout_model.add(layers.Dropout(0.5))
dropout_model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
dropout_model.add(layers.Dropout(0.5))
dropout_model.add(layers.Dense(1, activation='sigmoid'))
dropout_model.compile(optimizer='rmsprop',
				loss='binary_crossentropy', 
				metrics=['accuracy'])

dropout_hist = dropout_model.fit(x_train, y_train,
						epochs=20, batch_size=512, validation_data=(x_test, y_test))

# val_loss
original_model_val_loss = original_hist.history['val_loss']
dropout_model_val_loss = dropout_hist.history['val_loss']
epochs = range(1, len(original_model_val_loss) + 1)

# original model vs l2 model plots
plt.plot(epochs, original_model_val_loss, 'b+', label='Original model')
plt.plot(epochs, dropout_model_val_loss, 'bo', label='Dropout model')
plt.xlabel('Epochs'); plt.ylabel('Validation loss')
plt.legend(); plt.show()