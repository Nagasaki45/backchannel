"""
Predict backchannels based on speaker behaviour using the trained model.
"""

import numpy as np
import pandas as pd

import settings


def prepare_data_for_prediction(data, ids, samples, timestamp=None):
    """
    Prepare new data for feeding the model. Simplify working with multiple
    listeners in the same call, to speed up the prediction process.

    `data`: a dict of ID (int) to pd.Dataframe, indexed by time. DATA IS
            MUTATED IN PLACE.
    `ids`: a list of numbers representing the IDs of the listeners to
           generate backchannels for.
    `samples`: a list of feature vectors in the same length as `ids`. For each
               ID, these features are stacked to the end of the table in `data`,
               which is then resampled and sent the model to generate
               a prediction.
    `timestamp`: a way to specify the timestamp of the current sample, instead
                 of using current time.

    Returns: an X vector with one row per ID.
    """
    if timestamp is None:
        timestamp = pd.to_datetime('now')

    X = np.zeros((len(ids), settings.WINDOW_SIZE * len(samples[0])))

    for (i, id_), sample in zip(enumerate(ids), samples):
        update_data(data, id_, sample, timestamp)

        values = (data[id_]
                  .resample(f'{settings.RESAMPLING_PERIOD}L')
                  .first()
                  .ffill()
                  .values[-settings.WINDOW_SIZE:]
                  .flatten())

        X[i, -len(values):] = values

    return X


def predict(clf, data, ids, samples, timestamp=None):
    X = prepare_data_for_prediction(data, ids, samples, timestamp)
    return clf.predict(X)


def update_data(data, id_, sample, timestamp):
    """
    Get or create the pd.DataFrame, add the sample, and remove old data.
    """
    try:
        df = data[id_]
    except KeyError:
        df = pd.DataFrame([sample], index=[timestamp])
    else:
        df.loc[timestamp] = sample

    millis_to_keep = settings.RESAMPLING_PERIOD * settings.WINDOW_SIZE
    up_to = timestamp - pd.to_timedelta(millis_to_keep, 'ms')
    data[id_] = df.loc[up_to:]
