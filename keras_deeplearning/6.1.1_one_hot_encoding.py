import numpy as np
import string

## 단어 수준의 원-핫 인코딩 
samples = ['The cat sat on the mat.', 'The dog ate my homework.']

token_index = {}
for sample in samples:
	for word in sample.split():	# 샘플을 토큰으로 나눔
		if word not in token_index:
			token_index[word] = len(token_index) + 1

max_length = 10
results = np.zeros(shape=(len(samples), max_length, max(token_index.values())+1)) # 결과를 저장할 배열 

for i, sample in enumerate(samples):
	for j, word in list(enumerate(sample.split()))[:max_length]:
		index = token_index.get(word)
		results[i, j, index] = 1
print('단어 수준 one-hot\n',results)


## 문자 수준의 원-핫 인코딩 
samples = ['The cat sat on the mat.', 'The dog ate my homework.']
characters = string.printable	# 출력가능한 모든 아스키 문자
token_index = dict(zip(characters, range(1, len(characters)+1)))

max_length = 50
results = np.zeros((len(samples), max_length, max(token_index.values())+1))

for i, sample in enumerate(samples):
	for j, character in enumerate(sample):
		index = token_index.get(character)
		results[i, j, index] = 1
print('문자 수준 one-hot\n',results)


## 케라스를 사용한 단어 수준의 원-핫 인코딩
from keras.preprocessing.text import Tokenizer

samples = ['The cat sat on the mat.', 'The dog ate my homework.']
tokenizer = Tokenizer(num_words = 1000) # 가장 빈도가 높은 1000개의 단어만 선택
tokenizer.fit_on_texts(samples) # 단어 인덱스 구축 

sequences = tokenizer.texts_to_sequences(samples) # 문자열을 정수 인덱스의 리스트로 반환
one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary') # mode: 'binary', 'count', 'freq', 'tfidf'
# text_to_matrix() : text를 sequence list로 바꿔주는 text_to_sequences()와 
#                    sequence list를 numpy 배열로 바꿔주는 sequences_to_matrix()메서드를 차례로 호출
word_index = tokenizer.word_index # 계산된 단어 인덱스 
print(word_index)	# {'the': 1, 'cat': 2, 'sat': 3, 'on': 4, 'mat': 5, 'dog': 6, 'ate': 7, 'my': 8, 'homework': 9}
print('%s개의 고유한 토큰' % len(word_index))


## 해싱 기법을 사용한 단어 수준의 원-핫 인코딩 
samples = ['The cat sat on the mat.', 'The dog ate my homework.']
dimensionality = 1000 # 단어를 크기가 1,000인 벡터로 저장
max_length = 10
results = np.zeros((len(samples), max_length, dimensionality))

for i, sample in enumerate(samples):
	for j, word in list(enumerate(sample.split()))[:max_length]:
		index = abs(hash(word)) % dimensionality # 단어를 해싱하여 0과 1000사이의 랜덤한 정수 인덱스로 반환 
		results[i, j, index] = 1
print('해싱 one-hot',results)