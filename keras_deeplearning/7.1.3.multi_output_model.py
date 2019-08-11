from keras import models, layers, Input

vocab_size = 50000
num_income_groups = 10

posts_input = Input(shape=(None,), dtype='int32', name='posts')
embedded_posts = layers.Embedding(vocab_size, 256)(posts_input)
x = layers.Conv1D(128, 5, activation='relu')(embedded_posts)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dense(128, activation='relu')(X)

# output layers
age_pred = layers.Dense(1, name='age')(x)
income_pred = layers.Dense(num_income_groups, activation='softmax', name='income')(x)
gender_pred = layers.Dense(1, activation='sigmoid', name='gender')

model = models.Model(posts_input, [age_pred, income_pred, gender_pred])

 
mode.compile(optimizer='rmsprop',
				loss = ['mse', 'categorical_crossentropy', 'binary_crossentropy'],
				loss_weights = [0.25, 1., 10.])
model.fit(posts, [age_targets, income_targets, gender_targets], epochs=10, batch_size=64)


# model.compile(optimizer='rmsprop',
# 				loss={'age':'mse', 'income':'categorical_crossentropy', 'gender':'binary_crossentropy'},
# 				loss_weights={'age':0.25, 'income':1., 'gender':10.})

# model.fit(posts, {'age':age_targets, 'income':income_targets, 'gender':gender_targets}, 
# 					epochs=10, batch_size=128)
