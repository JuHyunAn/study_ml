import os
import numpy as np
from keras import models, layers, optimizers
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator

base_dir = './datasets/cats_and_dogs_small'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

# VGG16 합성 곱 기반 층 
conv_base = VGG16(weights='imagenet',		# 가중치 체크포인트 지정
				  include_top = False,		# 최상위 완전 연결 분류기를 포함할지 안할지 지정
				  input_shape=(150,150,3))

# 합성곱 기반 층 위에 완전 연결 분류기 추가 
model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# 마지막 3개의 합성곱 층을 미세조정(fine-tuning)
# -> block4_pool까지 모든 층이 동결되고, block5_conv1 ~ conv3 층은 학습 대상이 됨
conv_base.trainable = False

set_trainable = False
for layer in conv_base.layers:
	if layer.name == 'block5_conv1': # block_conv1부터는 학습 대상 
		set_trainable = True
	if set_trainable == True:
		layer.trainable = True
	else:
		layer.trainable = False

# model compile
model.compile(loss='binary_crossentropy',
			  optimizer=optimizers.RMSprop(lr=1e-5),
			  metrics=['acc'])

# preprocessing
train_datagen = ImageDataGenerator(
						rescale=1./255,
						rotation_range=40,
						width_shift_range=0.2, height_shift_range=0.2,
						shear_range=0.2, zoom_range=0.2,
						horizontal_flip=True,
						fill_mode='nearest')
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
									train_dir,
									target_size=(150,150),
									batch_size=20,
									class_mode='binary')
validation_generator = test_datagen.flow_from_directory(
									validation_dir,
									target_size=(150,150),
									batch_size=20,
									class_mode='binary')
test_generator = test_datagen.flow_from_directory(
									test_dir,
									target_size=(150,150),
									batch_size=20,
									class_mode='binary')

# model fit
history = model.fit_generator(
					train_generator,
					steps_per_epoch=100,
					epochs=100,
					validation_data=validation_generator,
					validation_steps=50)

# plot
import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc)+1)

# 지수이동평균(exponential moving averages)을 통해 그래프를 부드럽게 표현 
def smooth_curve(points, factor=0.8):
	smoothed_points = []
	for point in points:
		if smoothed_points:
			previous = smoothed_points[-1]
			smoothed_points.append(previous*factor+point*(1-factor))
		else:
			smoothed_points.append(point)
	return smoothed_points

plt.plot(epochs, smooth_curve(acc), 'bo', label='Training acc')
plt.plot(epochs, smooth_curve(val_acc), 'b', label='Validation acc')
plt.legend()

plt.figure()
plt.plot(epochs, smooth_curve(loss), 'bo', label='Training loss')
plt.plot(epochs, smooth_curve(val_loss), 'b', label='Validation loss')
plt.legend()
plt.show()

# 테스트 셋 평가
test_loss, test_acc =  model.evaluate_generator(test_generator, steps=50)
print(test_loss, test_acc)