"""
Utils for ML.
"""

import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import settings


def prepare_interaction(df, behaviours=settings.BEHAVIOURS):
    """
    - Resample the data.
    - Drop rows that are labeled as "Start".
    - Convert the "speaker_behaviour" column to hot-ones.
    - Filter hot-ones by `behaviours`.
    """
    # L stands for millis
    df = df.resample(f'{settings.RESAMPLING_PERIOD}L').first().ffill()

    df = df[df.speaker_behaviour != 'Start']

    hot_ones = pd.get_dummies(df.speaker_behaviour)
    hot_ones = [hot_ones[key] for key in hot_ones.columns if key in behaviours]
    columns = hot_ones + [df.speaker_eye, df.listener_nod]
    return pd.concat(columns, axis=1).astype(bool)


def to_samples(df, n):
    """
    Break an interaction into many samples in the form of `np.ndarray`.
    Each sample contains features from points in time up until the
    corrent one (including), and the `listener_nod` of the current time.
    """
    values = df.values
    num_of_samples = len(values) - n + 1
    X = np.array([_take_X(values, i, n) for i in range(num_of_samples)])
    y = values[n - 1:, -1][:, np.newaxis]
    return np.hstack([X, y])


def _take_X(values, i, n):
    return values[i:i + n, :-1].flatten()


def prepare_training_data(store, window_size=settings.WINDOW_SIZE):
    """
    Create the X and y arrays, ready for ML.
    """
    partials = []
    for key in store:
        df = prepare_interaction(store[key])
        samples = to_samples(df, window_size)
        partials.append(samples)
    data = np.vstack(partials)
    X = data[:, :-1]
    y = data[:, -1]
    return X, y


def plot_data(df, *, title=None, figsize=settings.FIGSIZE):
    """
    Plot the (already prepared for training) interaction for investigation.
    """
    spreaded = df - np.arange(len(df.columns)) * 2
    spreaded.plot(figsize=figsize)

    # Put a legend to the right of the current axis
    plt.gca().legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Hide Y axis labels
    plt.yticks([])

    if title is not None:
        plt.title(title)
