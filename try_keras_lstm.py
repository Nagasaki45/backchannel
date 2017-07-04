'''Example script to generate text from Nietzsche's writings.
At least 20 epochs are required before the generated text
starts sounding coherent.
It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.
If you try this script on new data, make sure your corpus
has at least ~100k characters. ~1M is better.

To run on a GPU:

  CUDA_VISIBLE_DEVICES="1" python try_keras_lstm.py  # change "1" to the GPU id
'''

import itertools
import random
import sys

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np

MAXLEN = 40
STEP = 3
NEURONS = 128
BATCH_SIZE = 128
DIVERSITIES = [0.2, 0.5, 1.0, 1.2]
CHARS_TO_GENERATE = 400

path = get_file('nietzsche.txt', origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
text = open(path).read().lower()
print('corpus length:', len(text))

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of MAXLEN characters
sentences = []
next_chars = []
for i in range(0, len(text) - MAXLEN, STEP):
    sentences.append(text[i: i + MAXLEN])
    next_chars.append(text[i + MAXLEN])
print('nb sequences:', len(sentences))

print('Vectorization...')
X = np.zeros((len(sentences), MAXLEN, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

# build the model: a single LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(NEURONS, input_shape=(MAXLEN, len(chars))))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.01))


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


# train the model, output generated text after each iteration
for iteration in itertools.count(1):
    print('~' * 50)
    print(f'- Iteration {iteration}')

    model.fit(X, y, batch_size=BATCH_SIZE, epochs=1)

    start_index = random.randint(0, len(text) - MAXLEN - 1)
    sentence = text[start_index: start_index + MAXLEN]
    print(f'- Generating with seed: "{sentence}"')
    sentence = list(sentence)  # to characters for appending predictions

    for diversity in DIVERSITIES:

        print(f'- diversity: {diversity}')

        for _ in range(CHARS_TO_GENERATE):
            x = np.zeros((1, MAXLEN, len(chars)))
            for t, char in enumerate(sentence[-MAXLEN:]):
                x[0, t, char_indices[char]] = 1.

            preds = model.predict(x, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]
            sentence.append(next_char)

        print(''.join(sentence))
