"""
Training an ML algorithm to predict listeners backchannel behaviours (head nods)
"""

import pickle
import random
import time

import numpy as np
import pandas as pd
from sklearn import neighbors
from sklearn import model_selection
from sklearn import metrics

RESAMPLING_PERIOD = 100  # millis
WINDOW_SIZE = 30
BEHAVIOURS = ['SpeechSilent']


def prepare_interaction(df, behaviours=BEHAVIOURS):
    """
    - Resample the data.
    - Drop rows that are labeled as "Start".
    - Convert the "speaker_behaviour" column to hot-ones.
    - Filter hot-ones by `behaviours`.
    """
    # L stands for millis
    df = df.resample(f'{RESAMPLING_PERIOD}L').first().ffill()

    df = df[df.speaker_behaviour != 'Start']

    hot_ones = pd.get_dummies(df.speaker_behaviour)
    hot_ones = [hot_ones[key] for key in hot_ones.columns if key in BEHAVIOURS]
    columns = hot_ones + [df.speaker_eye, df.listener_nod]
    return pd.concat(columns, axis=1).astype(bool)


def to_samples(df, n=WINDOW_SIZE):
    """
    Break an interaction into many samples in the form of `np.ndarray`.
    Each sample contains features from points in time up until the
    corrent one (including), and the `listener_nod` of the current time.
    """
    values = df.values
    num_of_samples = len(values) - n + 1
    X = np.array([_take_X(values, i, n) for i in range(num_of_samples)])
    Y = values[n - 1:, -1][:, np.newaxis]
    return np.hstack([X, Y])


def _take_X(values, i, n):
    return values[i:i + n, :-1].flatten()


def set_prediction(clf, df, n=WINDOW_SIZE):
    """
    Add the prediction to the table.
    """
    samples = to_samples(df, n)
    X = samples[:, :-1]
    Ypredict = clf.predict(X)
    Ypredict = np.concatenate([[np.nan] * (n - 1), Ypredict])
    df['prediction'] = Ypredict


def prepare_training_data(store):
    """
    Create a large table for the entire dataset, ready for ML.
    """
    partials = [] 
    for key in store:
        df = prepare_interaction(store[key])
        samples = to_samples(df)
        partials.append(samples)
    return np.vstack(partials)


def timeit(function, iterations):
    """
    Measure execution time per iteration.
    """
    time_passed = 0
    for iteration in range(iterations):
        start = time.time()
        function()
        end = time.time()
        time_passed += end - start
    return time_passed / iterations * 1000


def main():
    print('Preparing data')
    with pd.HDFStore('data.hdf') as store:
        data = prepare_training_data(store)

    Xtrain, Xtest, Ytrain, Ytest = model_selection.train_test_split(
            data[:, :-1],
            data[:, -1]
    )

    print('Training model')
    clf = neighbors.KNeighborsClassifier(n_jobs=-1)  # Utilize all cores
    clf.fit(Xtrain, Ytrain)

    print('Evaluating model')
    Ypredict = clf.predict(Xtest)
    print(metrics.classification_report(Ytest, Ypredict))

    print('Measuring performance')
    sample = Xtrain[random.randint(0, len(Xtrain) - 1), :]
    performance = timeit(lambda: clf.predict([sample]), iterations=100)
    print(f'{performance:.02f} ms/prediction')
    
    print('Persisting')
    with open('model.pickle', 'wb') as f:
        pickle.dump(clf, f)


if __name__ == '__main__':
    main()
