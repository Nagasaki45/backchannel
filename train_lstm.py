"""
Training an ML algorithm to predict listeners backchannel behaviours (head nods).

To run on a GPU:

  CUDA_VISIBLE_DEVICES="1" python train_lstm.py  # change "1" to the GPU id
"""

import itertools

from keras import layers
from keras.models import Sequential
import pandas as pd

import training


print('Preparing data')
with pd.HDFStore('data.hdf') as store:
    X, y = training.prepare_training_data_lstm(store)

print('Build model')
model = Sequential()
model.add(layers.LSTM(128, input_shape=X.shape[1:], return_sequences=True))
model.add(layers.LSTM(64, return_sequences=True))
model.add(layers.LSTM(32))
# 2 categories for 0 and 1. 'softmax' for classification.
model.add(layers.Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

for iteration in itertools.count(1):
    print('~' * 50)
    print(f'- Iteration {iteration}')

    model.fit(X, y, batch_size=128, epochs=1)

    model.save('model_lstm.hdf')
