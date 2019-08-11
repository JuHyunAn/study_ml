import os
import numpy as np
import matplotlib.pyplot as plt
from keras import models, layers
from keras import optimizers

data_dir = './datasets/jena_climate'
fname = os.path.join(data_dir, 'jena_climate_2009_2016.csv')
f = open(fname)
data = f.read()
f.close()

lines = data.split('\n')
header = lines[0].split(',') # column name
lines = lines[1:]			 # data
print(header)
print(len(lines))	# 420,551

# numpy 배열로 변환(one-hot)
float_data = np.zeros((len(lines), len(header)-1))
for i, line in enumerate(lines):
	values = [float(x) for x in line.split(',')[1:]]
	float_data[i, :] = values

# 온도(섭씨) 그래프
temp = float_data[:, 1]	# T (degC)
print(temp)
plt.plot(range(len(temp)), temp, label='Temp')
plt.show()	# -> 온도에 주기성이 있음을 확인 
# 처음 10일간 온도 그래프 (10분마다 데이터가 저장되므로 하루에 총 144개의 데이터 포인트 존재)
plt.plot(range(1440), temp[:1440])
plt.show()


## preprocessing 
# 정규화
mean = float_data[:200000].mean(axis=0)
float_data -= mean
std = float_data[:200000].std(axis=0)
float_data /= std

# 시계열 데이터와 타깃을 반환하는 제너레이터 함수
def generator(data, lookback, delay, min_index, max_index, shuffle=False, batch_size=128, step=6):
	if max_index is None:
		max_index = len(data) - delay -1
	i = min_index + lookback
	while 1:
		if shuffle:
			rows = np.random.randint(min_index + lookback, max_index, size=batch_size)
		else:
			if i + batch_size >= max_index:
				i = min_index + lookback
			rows = np.arange(i, min(i +batch_size, max_index))
			i += len(rows)

		samples = np.zeros((len(rows), lookback//step, data.shape[-1]))
		targets = np.zeros((len(rows),))
		for j, row in enumerate(rows):
			indices = range(rows[j] - lookback, rows[j], step)
			samples[j] = data[indices]
			targets[j] = data[rows[j] + delay][1]
		yield samples, targets

## 훈련(처음 20만개), 검증(그다음 10만개), 테스트(나머지)
lookback = 1440
step = 6
delay = 144
batch_size = 128
train_gen = generator(float_data, lookback=lookback, delay=delay, 
					min_index=0, max_index=200000, shuffle=True, step=step, batch_size=batch_size)
val_gen = generator(float_data, lookback=lookback, delay=delay, 
					min_index=200001, max_index=300000, shuffle=True, step=step, batch_size=batch_size)
test_gen = generator(float_data, lookback=lookback, delay=delay, 
					min_index=300001, max_index=None, shuffle=True, step=step, batch_size=batch_size)
val_steps = (300000 - 200001 - lookback) // batch_size # 전체 검증 셋을 순회하기 위해 val_gen에서 추출할 횟수
test_steps = (len(float_data) - 300001 - lookback) // batch_size # 전체 test 셋을 순회하기 위해 test_gen에서 추출할 횟수

## 스태킹 순환 층 
model = models.Sequential() 
model.add(layers.GRU(32, dropout=0.1, 	# dropout: 입력에 대한 드롭아웃 비율
						recurrent_dropout=0.5, 	# recurrent_dropout: 순환 상태의 드롭아웃 비율 
						return_sequences=True,	# 전체 시퀀스(3D) 출력
						input_shape=(None, float_data.shape[-1])))
model.add(layers.GRU(64, activation='relu', 
						dropout=0.1,
						recurrent_dropout=0.5))
model.add(layers.Dense(1))
model.compile(optimizer=optimizers.RMSprop(), loss='mae')
history = model.fit_generator(train_gen, steps_per_epoch=500, epochs=40, 
								validation_data=val_gen, validation_steps=val_steps)

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss)+1)
# 
plt.plot(epochs, loss, label='Training loss')
plt.plot(epochs, val_loss, label='Validation loss')
plt.legend()
plt.show()