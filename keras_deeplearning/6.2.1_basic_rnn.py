## RNN 구현 
import numpy as np

timesteps = 100	# 입력 시퀀스에 있는 타임스텝의 수
input_features = 32	# 입력 특성 차원
output_features = 64 # 출력 특성 차원 

inputs = np.random.random((timesteps, input_features)) # 입력 데이터: 예제를 위해 생성한 난수
state_t = np.zeros((output_features)) # 초기상태: 모두 0인 벡터

# 랜덤한 가중치 행렬 
W = np.random.random((output_features, input_features))
U = np.random.random((output_features, output_features))
b = np.random.random((output_features, ))

successive_output = []
for input_t in inputs:
	output_t = np.tanh(np.dot(W, input_t)+np.dot(U, state_t)+b) # 입력과 현재 상태(이전출력)을 연결하여 현재 출력을 구함
	successive_output.append(output_t) # 출력을 리스트에 저장
	state_t = output_t # 다음스텝을 위해 네트워크 상태 업데이트 

final_output_sequence = np.stack(successive_output, axis=0) # (timesteps, output_features)
print(final_output_sequence)


## keras SimpleRNN
from keras.datasets import imdb
from keras import models, layers
from keras.preprocessing import sequence

# model = models.Sequential()
# model.add(layers.Embedding(input_dim=10000, output_dim=32))
# model.add(layers.SimpleRNN(32))	# 마지막 타임스텝의 출력만 반환
# model.add(layers.SimpleRNN(32, return_sequences=True)) # 전체 상태 시퀀스 반환 
# print(model.summary())

max_features = 10000 # 특성으로 사용할 단어 수
maxlen = 500 # 사용할 텍스트의 길이

(input_train, y_train), (input_test, y_test) = imdb.load_data(num_words=max_features)
print('train sequence: ',len(input_train))
print('test sequence: ',len(input_test))

# 시퀀스 패딩 (samples x time)
input_train = sequence.pad_sequences(input_train, maxlen=maxlen)
input_test = sequence.pad_sequences(input_test, maxlen=maxlen)
print('input_train 크기: ',input_train.shape)	# (25000, 500)
print('input_test 크기: ',input_test.shape)		# (25000, 500)

model = models.Sequential()
model.add(layers.Embedding(max_features, 32))
model.add(layers.SimpleRNN(32))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
				loss='binary_crossentropy',
				metrics=['acc'])

history = model.fit(input_train, y_train, epochs=5, batch_size=512, validation_split=0.2)


# plot
import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc)+1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.legend()

plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.legend()
plt.show()
