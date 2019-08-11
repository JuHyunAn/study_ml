import matplotlib.pyplot as plt
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras import models, layers

max_features = 10000 
maxlen = 500
(input_train, y_train), (input_test, y_test) = imdb.load_data(num_words=max_features)

# preprocessing
input_train = sequence.pad_sequences(input_train, maxlen=maxlen) # 모델 입력으로 사용하기 위해 2D 정수 tensor로 변환 
input_test = sequence.pad_sequences(input_test, maxlen=maxlen)

# 양방향 RNN
model = models.Sequential()
model.add(layers.Embedding(max_features, 32))
model.add(layers.Bidirectional(layers.LSTM(32, dropout=0.5, recurrent_dropout=0.5)))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
				loss='binary_crossentropy',
				metrics=['acc'])

# fit
history = model.fit(input_train, y_train, epochs=5, batch_size=128, validation_split=0.2)

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc)+1)

# plot
plt.plot(epochs, acc, label='Training acc')
plt.plot(epochs, val_acc, label='Validation acc')
plt.legend()

plt.figure()
plt.plot(epochs, loss, label='Training loss')
plt.plot(epochs, val_loss, label='Validation loss')
plt.legend()
plt.show()