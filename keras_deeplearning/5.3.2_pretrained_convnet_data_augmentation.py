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
model.add(conv_base)	# VGG16 모델 추가
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
print(model.summary())
conv_base.trainable = False	# 사전 학습 모델을 동결(freezing) -> 훈련동안 가중치가 업데이트 되는 것을 방지

model.compile(loss='binary_crossentropy',
				optimizer=optimizers.RMSprop(lr=2e-5),
				metrics=['acc'])

# 동결된 합성곱 기반 층과 함께 모델을 end-to-end로 훈련
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
					epochs=30,
					validation_data=validation_generator,
					validation_steps=50)

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
plt.title('Training and Validation loss')
plt.legend()
plt.show()