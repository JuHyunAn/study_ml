import os
import numpy as np

imdb_dir = 'C:/Users/USER/Downloads/deep-learning-with-python-notebooks-master/datasets/aclImdb'
train_dir = os.path.join(imdb_dir, 'train')

## 훈련 데이터를 문자열 리스트로 구성/리뷰 레이블 구성 
texts = []	# 문자열 list
labels = []	# 레이블 list 
for label_type in ['neg','pos']:
	dir_name = os.path.join(train_dir, label_type)
	for fname in os.listdir(dir_name):
		if fname[-4:] == '.txt':	# 파일 확장자가 .txt이면 
			f = open(os.path.join(dir_name, fname), encoding='utf8')
			texts.append(f.read())
			f.close()

			if label_type == 'neg':
				labels.append(0)
			else:
				labels.append(1)

print(texts[0])
print(labels[0])

## preprocessing 
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

maxlen = 100	# 100개 단어까지
training_samples = 200	# 훈련 샘플
validation_samples = 10000	# 검증 샘플
max_words = 10000	# 가장 높은 빈도 1만개 단어 

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)	# word_index 구축
sequence = tokenizer.texts_to_sequences(texts)	# 문자열을 정수 인덱스의 리스트로 변환 
print(sequence[0])

word_index = tokenizer.word_index
print('%s개의 고유한 토큰.' % len(word_index))	# 88582개의 고유한 토큰.

data = pad_sequences(sequence, maxlen=maxlen)	# 모델 입력으로 사용하기 위해 2D 정수 tensor로 변환 
print(data)

labels = np.asarray(labels)
print('data tensor size: ',data.shape)		# (25000, 100)
print('label tensor size: ',labels.shape)	# (25000,)

# train set과 validation set으로 분할 
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val = data[training_samples: training_samples+validation_samples]
y_val = labels[training_samples: training_samples+validation_samples]

# GloVe 단어 임베딩 파일 파싱
glove_dir = 'C:/Users/USER/Downloads/deep-learning-with-python-notebooks-master/datasets/'

embedding_index = {}
f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'), encoding='utf8')
for line in f:
	values = line.split()
	word = values[0]
	coefs = np.asarray(values[1:], dtype='float32')
	embedding_index[word] = coefs
f.close()
print('%s개의 단어 벡터.' % len(embedding_index))

# GloVe 단어 임베딩 행렬 
embedding_dim = 100
embedding_matrix = np.zeros((max_words, embedding_dim)) # (10000, 100)
for word, i in word_index.items():
	if i < max_words:
		embedding_vector = embedding_index.get(word)
		if embedding_vector is not None:
			embedding_matrix[i] = embedding_vector # embedding_index에 없는 단어는 모두 0이 됨

## 모델 정의하기 
from keras import models, layers
model = models.Sequential()
model.add(layers.Embedding(max_words, embedding_dim, input_length=maxlen))
model.add(layers.Flatten())
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
print(model.summary())

# 사전 훈련된 단어 임베딩을 Embedding 층에 로드
model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False # Embedding 층 동결

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(x_train, y_train,
					epochs=10, batch_size=32,
					validation_data = (x_val, y_val))
model.save_weights('pre_trained_glove_model.h5')


# # plot
# import matplotlib.pyplot as plt
# acc = history.history['acc']
# val_acc = history.history['val_acc']
# loss = history.history['loss']
# val_loss = history.history['val_loss']
# epochs = range(1, len(acc)+1)

# plt.plot(epochs, acc, 'bo', label='Training acc')
# plt.plot(epochs, val_acc, 'b', label='Validation acc')
# plt.legend()

# plt.figure()
# plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.legend()
# plt.show()

# 테스트 데이터 토큰화
test_dir = os.path.join(imdb_dir,'test')

texts = []
labels = []
for label_type in ['neg','pos']:
	dir_name = os.path.join(test_dir, label_type)
	for fname in sorted(os.listdir(dir_name)):
		if fname[-4:] == '.txt':
			f = open(os.path.join(dir_name, fname), encoding='utf8')
			texts.append(f)
			if label_type == 'neg':
				labels.append(0)
			else:
				labels.append(1)

sequences = tokenizer.texts_to_sequences(texts)
x_test = pad_sequences(sequences, maxlen=maxlen)
y_test = np.asarray(labels)

# 모델 평가
model.load_weights('pre_trained_glove_model.h5')
test_acc, test_loss = model.evalute(x_test, y_test)
print('test_acc:',test_acc)
print('test_loss:',test_loss)