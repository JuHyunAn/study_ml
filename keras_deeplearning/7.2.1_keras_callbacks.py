from keras.datasets import mnist
from keras import layers, models
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 0 ~ 1
train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255

# one-hot
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# conv2d model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten()) # 3D tensor -> 1D tensor
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
print(model.summary())




#### callbacks
callbacks_list = [EarlyStopping(monitor='val_acc', patience=1),		# 1 에포크 이상(2 에포크 동안) 검증 정확도가 향상 되지 않으면 훈련 중지
				ModelCheckpoint(filepath='my_model.h5', monitor='val_loss', save_best_only=True)]	# 훈련 동안 가장 좋은 모델만 저장

# ReduceLROnPlateau
# -> 검증 손실이 향상되지 않을 때 학습률을 작게 함 
# -> 손실 곡선이 평탄할 때 학습률을 작게 하거나 크게 하면 지역 최솟값에서 효과적으로 빠져나올 수 있음
callbacks_list_2 = [ReduceLROnPlateau(monitor='val_loss',
										factor=0.1,		# callback이 호줄될 때 학습률을 10배로 줄임
										patience=10),	# 10 에포크 동안 향상되지 않으면 콜백을 호출
					ModelCheckpoint(filepath='my_model.h5', monitor='val_loss', save_best_only=True)]

# model compile
model.compile(optimizer='rmsprop',
			loss = 'categorical_crossentropy', metrics=['accuracy'])

model.fit(train_images, train_labels, 
				epochs=100, batch_size=64, 
				callbacks=callbacks_list_2, 
				validation_split=0.2)	# callback이 val_loss와 val_acc를 모니터링하기 때문에 검증데이터를 전달

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('test_loss:',test_loss)
print('test_acc: ',test_acc)