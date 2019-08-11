import numpy as np
from keras import models, layers, Input
from keras.utils import to_categorical


text_vocab_size = 10000
question_vocab_size = 10000
answer_vocab_size = 500

text_input = Input(shape=(None,), dtype='int32', name='text')	# 길이가 정해지지않은 정수 시퀀스 
embedded_text = layers.Embedding(text_vocab_size, 64)(text_input)	# 입력을 크기가 64인 벡터의 시퀀스로 임베딩
encoded_text = layers.LSTM(32)(embedded_text)	# LSTM을 사용하여 이 벡터들을 하나의 벡터로 인코딩


question_input = Input(shape=(None,), dtype='int32',  name='question')
embedded_question = layers.Embedding(question_vocab_size, 32)(question_input)
encoded_question = layers.LSTM(16)(embedded_question)

# 인코딩된 질문과 텍스트를 연결
concatenated = layers.concatenate([encoded_text, encoded_question], axis=1)
answer = layers.Dense(answer_vocab_size, activation='softmax')(concatenated)

# 모델 객체 생성 - 2개의 입력과 출력을 주입
model = models.Model([text_input, question_input], answer)
model.compile(optimizer='rmsprop',
				loss='categorical_crossentropy',
				metrics=['acc'])

## 데이터 주입
num_samples = 1000
max_length = 100
text = np.random.randint(1, text_vocab_size, size=(num_samples, max_length)) # 랜덤한 numpy 데이터 생성
question = np.random.randint(1, question_vocab_size, size=(num_samples, max_length))
answers = np.random.randint(0, answer_vocab_size, size=num_samples)
answers = to_categorical(answers)

model.fit([text, question], answers, epochs=50, batch_size=128)
# = model.fit({'text':text, 'question':question}, answer, ~)
