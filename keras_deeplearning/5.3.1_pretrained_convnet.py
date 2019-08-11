import os
import numpy as np
from keras import models, layers, optimizers
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator

# VGG16 합성 곱 기반 층 
conv_base = VGG16(weights='imagenet',		# 가중치 체크포인트 지정
				  include_top = False,		# 최상위 완전 연결 분류기를 포함할지 안할지 지정
				  input_shape=(150,150,3))


# 사전 훈련된 합성곱 기반 층을 사용한 특성 추출하기 
base_dir = './datasets/cats_and_dogs_small'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

datagen = ImageDataGenerator(rescale=1./255)
batch_size = 20

def extract_features(directory, sample_count):
	features = np.zeros(shape=(sample_count, 4, 4, 512))
	labels = np.zeros(shape=(sample_count))
	generator = datagen.flow_from_directory(
							directory,
							target_size=(150,150),
							batch_size=batch_size,
							class_mode='binary')
	i = 0
	for inputs_batch, labels_batch in generator:
		features_batch = conv_base.predict(inputs_batch)
		features[i*batch_size : (i+1)*batch_size] = features_batch
		labels[i*batch_size : (i+1)*batch_size] = labels_batch
		i+=1
		if i * batch_size >= sample_count:
			break
	return features, labels

train_features, train_labels = extract_features(train_dir, 2000)
validation_features, validation_labels = extract_features(validation_dir, 1000)
test_features, test_labels = extract_features(test_dir, 1000)

# 완전 연결 분류기 정의
model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=4*4*512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
				loss='binary_crossentropy',
				metrics=['acc'])

history = model.fit(train_features, train_labels,
					epochs=30, 
					batch_size=20,
					validation_data=(validation_features, validation_labels))

# plot
import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc)+1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation_acc')
plt.title('Training and Validation accuracy')
plt.legend()

plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()
plt.show()