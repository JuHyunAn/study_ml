import numpy as np
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.preprocessing import image
from keras import backend as K

model = VGG16(weights='imagenet')

img_path = './datasets/creative_commons_elephant.jpg'
img = image.load_img(img_path, target_size=(224,224))

x = image.img_to_array(img) # (224, 224, 3) 크기의 넘파이 float32 배열
x = np.expand_dims(x, axis=0) # 차원을 추가하여 (1, 224, 224, 3) 크기의 배치로 배열 변환
x = preprocess_input(x) # 채널별 컬러 정규화 수행 

preds = model.predict(x)
print('Predicted: ',decode_predictions(preds, top=3)[0]) # decode_predictions()
# -> ImageNet 데이터셋에 대한 예측결과에서 top 매개변수에 지정된 수만큼 최상위 항목 반환
print(np.argmax(preds[0])) # 386


# Grad-CAM 알고리즘 
african_elephant_output = model.output[:, 386] # 예측벡터의 '아프리카 코끼리' 항목

last_conv_layer = model.get_layer('block5_conv3') # VGG16의 마지막 합성곱 층인 block_conv3의 특성 맵
grads = K.gradients(african_elephant_output, last_conv_layer)[0] # '아프리카 코끼리' 클래스의 그래디언트
pooled_grads = K.mean(grads, axis=(0,1,2)) # 특성 맵 채널별 그래디언트 평균값이 담긴 (512,) 크기의 벡터
iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]]) # 특성 맵 출력을 구함
pooled_grads_value, conv_layer_output_value = iterate([x])

for i in range(512): # '아프리카 코끼리' 클래스에 대한 채널 중요도를 특성 맵 배열의 채널에 곱함
	conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

heatmap = np.mean(conv_layer_output_value, axis=1)
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
plt.matshow(heatmap)
plt.show()


# 원본 이미지에 히트맵 덧붙이기 
import cv2
img = cv2.imread(img_path)
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0])) # heatmap을 원본 이미지 크기에 맞게 변경
heatmap = np.uint8(255*heatmap) # heatmap을 RGB포맷으로 변환
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET) # heatmap으로 변환
superimposed_img = heatmap * 0.4 + img # 히트맵의 강도 
cv2.imwrite('./datasets/elephant_cam.jpg', superimposed_img)