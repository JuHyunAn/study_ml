import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import boston_housing
from keras import layers, models

(train_data, train_targets),(test_data, test_targets) = boston_housing.load_data()
print(train_data.shape)	# (404, 13)
print(test_data.shape)	# (102, 13)
print(train_targets[:5])	# 천 달러 단위

# 특성별 스케일이 상이하므로 -> 정규화
# 특성의 평균을 빼고, 표준편차로 나눔 
mean = train_data.mean(axis=0)
std = train_data.std(axis=0)

train_data -= mean
train_data /= std
test_data -= mean 
test_data /= std

def build_model():
	model = models.Sequential()
	model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
	model.add(layers.Dense(64, activation='relu'))
	model.add(layers.Dense(1))	# 선형 층(regression layer)

	model.compile(optimizer='rmsprop',
			loss='mse', 		# Mean Squared Error
			metrics=['mae'])	# Mean Absolute Error
	return model


# k-fold cross validation 	
k = 4
num_val_samples = len(train_data)//k
num_epochs = 500
all_mae_histories = []
for i in range(k):
	print('처리중인 fold #',i)
	# 검증 데이터 준비
	val_data = train_data[i*num_val_samples : (i+1)*num_val_samples]
	val_target = train_targets[i*num_val_samples : (i+1)*num_val_samples]
	# 훈련 데이터 준비
	partial_train_data = np.concatenate(
							[train_data[: i*num_val_samples], train_data[(i+1)*num_val_samples:]], 
							axis=0)
	partial_train_target = np.concatenate(
							[train_targets[: i*num_val_samples], train_targets[(i+1)*num_val_samples:]],
							axis=0)

	model = build_model()
	history = model.fit(partial_train_data, partial_train_target,
					validation_data=(val_data, val_target),
					epochs=num_epochs, batch_size=1, verbose=0)
	mae_history = history.history['val_mean_absolute_error']
	all_mae_histories.append(mae_history)

# 각 포인트를 이전 포인트의 지수 이동 평균으로 대체 
def smooth_curve(points, factor=0.9):
	smoothed_points = []
	for point in points:
		if smoothed_points:
			previous = smoothed_points[-1]
			smoothed_points.append(previous*factor + point*(1-factor))
		else:
			smoothed_points.append(point)
	return smoothed_points

# score
average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]
smooth_mae_history = smooth_curve(average_mae_history)

# plot
plt.plot(range(1, len(smooth_mae_history)+1), smooth_mae_history)
plt.xlabel('Epochs'); plt.ylabel('Validation MAE')
plt.show()


# 최종 모델 훈련하기 
model = build_model()
model.fit(train_data, train_targets, epochs=80, batch_size=16)
test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)
print(test_mae_score)