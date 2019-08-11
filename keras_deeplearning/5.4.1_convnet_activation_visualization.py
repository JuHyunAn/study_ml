import numpy as np
from keras import models

model = models.load_model('cats_and_dogs_small_2.h5')
print(model.summary())

# 개별 이미지 전처리 
from keras.preprocessing import image
img_path = './datasets/cats_and_dogs_small/test/cats/cat.1700.jpg'

img = image.load_img(img_path, target_size=(150,150))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)	# 차원을 추가하여 이미지를 4D tensor로 변경 
img_tensor /= 255
print(img_tensor.shape)	# (1, 150, 150, 3)

# test img 출력
import matplotlib.pyplot as plt
plt.imshow(img_tensor[0])
plt.show()

# 입력 텐서와 출력 텐서의 리스트로 모델 객체 만들기 
layer_output = [layer.output for layer in model.layers[:8]]	# 상위 8개 층의 출력을 추출
activation_model = models.Model(inputs=model.input, outputs=layer_output)

# 예측모드로 모델 실행
activations = activation_model.predict(img_tensor) # 층의 활성화마다 하나씩 8개의 numpy배열로 이루어진 리스트르 반환 
first_layer_activation = activations[0]
print(first_layer_activation.shape)	# (1, 148, 148, 32) -> 첫번째 합성곱 층의 활성화 값
plt.matshow(first_layer_activation[0, :, :, 19], cmap='viridis') # 20번째 채널 시각화(엣지)
plt.matshow(first_layer_activation[0, :, :, 15], cmap='viridis') # 16번째 채널 시각화(밝은 녹색 점)
plt.show()

# 네트워크의 모든 활성화 시각화 
layer_names = []
for layer in model.layers[:8]:	# 상위 8개 층만 
	layer_names.append(layer) # 층의 이름을 그래프 제목으로 사용 

images_per_row = 16
for layer_name, layer_activation in zip(layer_names, activations):	# 특성화 맵
	n_features = layer_activation.shape[-1]	# 특성 맵에 있는 특성의 수
	size = layer_activation.shape[1] # 특성 맵의 크기는 (1, size, size, n_features)
	n_cols = n_features // images_per_row # 활성화 채널을 그리기 위한 그리드 크기 
	display_grid = np.zeros((size*n_cols, images_per_row*size))

	for col in range(n_cols):	# 각 활성화를 하나의 큰 그리드에 채움
		for row in range(images_per_row):
			channel_image = layer_activation[0, :, :, col*images_per_row]
			# 그래프로 나타내기 좋게 특성 처리 
			channel_image -=  channel_image.mean()
			channel_image *= 64
			channel_image += 128
			channel_image = np.clip(channel_image, 0, 255).astype('uint8')
			display_grid[col*size : (col+1)*size, row*size : (row+1)*size] = channel_image

	scale = 1. / size
	plt.figure(figsize=(scale*display_grid.shape[1], scale*display_grid.shape[0]))
	plt.title(layer_name)
	plt.grid(False)
	plt.imshow(display_grid, aspect='auto', cmap='viridis')

plt.show()