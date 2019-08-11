from keras import layers, models
from keras import optimizers

train_dir = './datasets/cats_and_dogs_small/train'
validation_dir = './datasets/cats_and_dogs_small/validation'

# model define
model = models.Sequential()
model.add(layers.SeparableConv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.SeparableConv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.SeparableConv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.SeparableConv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())	# 6272
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
print(model.summary())

model.compile(loss='binary_crossentropy',
				optimizer=optimizers.RMSprop(lr=1e-4),
				metrics=['acc'])

# preprocessing
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255) # 스케일 조정
test_datagen = ImageDataGenerator(rescale=1./255)  # 스케일 조정

train_generator = train_datagen.flow_from_directory(
										train_dir, 				# 타깃 디렉토리
										target_size=(150, 150),	# 모든 이미지를 150x150으로
										batch_size=20, class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
										validation_dir,
										target_size=(150, 150),
										batch_size=20, class_mode='binary')

# model fit 
history = model.fit_generator(
					train_generator, 
					steps_per_epoch=100, # batch가 20이므로 2000개 모두 처리하려면 100
					epochs=20,
					validation_data=validation_generator,
					validation_steps=50) # batch가 20이므로 1000개 모두 처리하려면 50

# model save
model.save('cats_and_dogs_small_1.h5')


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