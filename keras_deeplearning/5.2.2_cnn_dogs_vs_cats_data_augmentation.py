from keras import layers, models
from keras import optimizers

train_dir = './datasets/cats_and_dogs_small/train'
validation_dir = './datasets/cats_and_dogs_small/validation'
test_dir = './datasets/cats_and_dogs_small/test'

# model define
model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', 
				optimizer=optimizers.RMSprop(lr=1e-4),
				metrics=['acc'])

# preprocessing
from keras.preprocessing.image import ImageDataGenerator
# 데이터 증식 
train_datagen = ImageDataGenerator(
						rescale=1./255,
						rotation_range=40,		# 랜덤하게 사진을 회전시킬 각도 범위
						width_shift_range=0.2,	# 수평과 수직으로 랜덤하게 평행 이동시킬 범위
						height_shift_range=0.2,	
						shear_range=0.2,		# 랜덤하게 전단 변환을 적용할 각도 범위
						zoom_range=0.2,			# 랜덤하게 사진을 확대할 범위
						horizontal_flip=True)	# 랜덤하게 이미지를 수평으로 뒤집음 
test_datagen = ImageDataGenerator(rescale=1./255)	# 검증 및 test 데이터는 데이터 증식을 하면 안됨!

train_generator = train_datagen.flow_from_directory(
									train_dir,
									target_size=(150,150),
									batch_size=32,
									class_mode='binary')
validation_generator = test_datagen.flow_from_directory(
									validation_dir,
									target_size=(150,150),
									batch_size=32,
									class_mode='binary')
test_generator = test_datagen.flow_from_directory(
									test_dir,
									target_size=(150,150),
									batch_size=32,
									class_mode='binary')

# model fit
history = model.fit_generator(
					train_generator,
					steps_per_epoch=100,
					epochs=100,
					validation_data=validation_generator,
					validation_steps=50)

# model save
model.save('cats_and_dogs_small_2.h5')

# acc, loss graph
import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc)+1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and Validation accuracy')
plt.legend()

plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss , 'b', label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()
plt.show()