import keras 
from keras import layers, models
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.callbacks import TensorBoard

max_features = 2000
max_len = 500

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
x_train = sequence.pad_sequences(x_train, maxlen=max_len)	# max_len 차원 만큼 패딩 추가
x_test = sequence.pad_sequences(x_test, maxlen=max_len)

model = models.Sequential()
model.add(layers.Embedding(max_features, 128, input_length=max_len, name='embed'))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.MaxPooling1D(5))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.GlobalMaxPool1D())
model.add(layers.Dense(1))
model.summary()

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

## tensorboard
callbacks = [TensorBoard(
					log_dir='my_log_dir',
					histogram_freq=1,		# 1에포크마다 활성화 출력의 히스토그램 기록
					embeddings_freq=1,		# 1에포크마다 임베딩 데이터 기록
					embeddings_layer_names='embed',
					embeddings_data=x_test)]

history = model.fit(x_train, y_train,
					epochs=10,
					batch_size=128,
					validation_split=0.2,
					callbacks=callbacks)