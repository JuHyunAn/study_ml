'''
# 기존 모델과 l2규제가 추가된 모델과의 비교 
# L1 : 가중치의 절대값에 비례하는 비용이 추가
- kernel_regularizer=regularizers.l1(0.001)
# L2 : 가중치의 제곱에 비례하는 비용이 추가
 - kernel_regularizer=regularizers.l2(0.001)
# L1+L2
- kernel_regularizer=regularizers.l1_l2(l1=0.0001, l2=0.0001)
'''
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import imdb
from keras import models, layers, regularizers

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

def vectorize_sequence(sequences, dimension=10000):
	results = np.zeros((len(sequences), dimension))

	for i, sequences in enumerate(sequences):
		results[i, sequences] = 1
	return results

# vectorize
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

# l2 규제 추가한 모델 
l2_model = models.Sequential()
l2_model.add(layers.Dense(16, kernel_regularizer=regularizers.l2(0.001),
							activation='relu', input_shape=(10000,)))
l2_model.add(layers.Dense(16, kernel_regularizer=regularizers.l2(0.001),
							activation='relu'))
l2_model.add(layers.Dense(1, activation='sigmoid'))
l2_model.compile(optimizer='rmsprop',
				loss='binary_crossentropy', 
				metrics=['accuracy'])

l2_hist = l2_model.fit(x_train, y_train,
						epochs=20, batch_size=512, validation_data=(x_test, y_test))

# val_loss
epochs = range(1, len(original_model_val_loss) + 1)
original_model_val_loss = original_hist.history['val_loss']
l2_model_val_loss = l2_hist.history['val_loss']

# original model vs l2 model plots
plt.plot(epochs, original_model_val_loss, 'b+', label='Original model')
plt.plot(epochs, l2_model_val_loss, 'bo', label='L2 regulaized model')
plt.xlabel('Epochs'); plt.ylabel('Validation loss')
plt.legend(); plt.show()