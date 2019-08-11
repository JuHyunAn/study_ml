from keras.datasets import imdb
from keras import layers, models, preprocessing

max_feautres = 10000 # 특성으로 사용할 단어 수
maxlen = 20 # 사용할 텍스트 길이
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_feautres)
# 리스트를 (samples, maxlen) 크기의 2D 정수 텐서로 반환
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

# models
# Embedding 층은 크기가 (samples, sequence_length, embedding_dimensionality)인 3D 실수형 tensor를 반환
model = models.Sequential()
model.add(layers.Embedding(max_feautres, 8, input_length=maxlen))
model.add(layers.Flatten()) # (samples, maxlen*8) 크기의 2D tensor로 펼침
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
			  loss='binary_crossentropy',
			  metrics=['acc'])
print(model.summary())

history = model.fit(x_train, y_train,
					epochs=10, batch_size=32,
					validation_split=0.2)

print('acc: ',history.history['acc'])
print('val_acc: ',history.history['val_acc'])

# plot 
import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc)+1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.xlabel('epochs'); plt.ylabel('acc')
plt.legend()

plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.xlabel('epochs'); plt.ylabel('loss')
plt.legend()
plt.show()