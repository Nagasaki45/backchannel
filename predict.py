"""
Predict backchannels based on speaker behaviour using the trained model.
"""

import numpy as np
import pandas as pd

import settings


def new_state():
    """
    Get an initial empty state.
    """
    return {'thresholds': {}}


def prepare_data_for_prediction(state, ids, samples, timestamp=None):
    """
    Prepare new data for feeding the model. Simplify working with multiple
    listeners in the same call, to speed up the prediction process.

    `state`: an internal state. Get an initial one with `new_state`.
    `ids`: a list of numbers representing the IDs of the listeners to
           generate backchannels for.
    `samples`: a list of feature vectors in the same length as `ids`.
    `timestamp`: a way to specify the timestamp of the current sample, instead
                 of using current time.

    Returns: a `{state, X}` tuple with `state` being an internal state that
             should pass to future calls and `X` being a vector with one
             row per ID.
    """
    if timestamp is None:
        timestamp = pd.to_datetime('now')

    X = np.zeros((len(ids), settings.WINDOW_SIZE * len(samples[0])))

    for (i, id_), sample in zip(enumerate(ids), samples):
        state = update_state(state, id_, sample, timestamp)

        values = (state[id_]
                  .resample(f'{settings.RESAMPLING_PERIOD}L')
                  .first()
                  .ffill()
                  .values[-settings.WINDOW_SIZE:]
                  .flatten())

        X[i, -len(values):] = values

    return state, X


def predict(state, clf, ids, samples, type_=None):
    """
    Predict backchannels for a list of active listeners in batch.

    `state`: an internal state. Get an initial one with `empty_state`.
    `clf`: a trained model.
    `ids`: a list of numbers representing the IDs of the listeners to
           generate backchannels for.
    `samples`: a list of feature vectors in the same length as `ids`. For each
               ID, these features are stacked to the end of the table in `data`,
               which is then resampled and sent the model to generate
               a prediction.
    `type_`: one of `['classifier', 'proba', 'dekok']`. Defaults to
             'classifier'.

    # Types

    - classifier: zeros and ones indicate backchannel predictions.
    - proba: predicted probability for backchannel.
    - dekok: a binary classifier that depends on probability estimation of
             the model and varying threshold. The threshold starts at 1 and
             is constantly decreasing, until the probability > threshold.
             Then, a backchannel is predicted and the threashold resets to 1.
             For more information see de Kok and Heylen 2012.

    Returns: a `{state, yhat}` tuple with `state` being an internal state that
             should pass to future calls and `yhat` being vector with one
 prediction (int or float, depending on type) per ID.
    """
    assert len(ids) == len(samples), 'ids and samples must be of sample length'

    if type_ is None:
        type_ = 'classifier'

    assert type_ in ['classifier', 'proba', 'dekok'], f'Got {type_}'

    if len(ids) == 0:
        return state, np.array([])

    state, X = prepare_data_for_prediction(state, ids, samples)

    if type_ == 'classifier':
        # tolist to fix serialization issue http://bugs.python.org/issue18303
        yhat = clf.predict(X).tolist()
    if type_ == 'proba':
        proba = clf.predict_proba(X)[:, 1]
        yhat = proba.tolist()
    if type_ == 'dekok':
        proba = clf.predict_proba(X)[:, 1]
        state, yhat = _dekok(state, ids, proba)

    return state, yhat


def _dekok(state, ids, probas, timestamp=None):
    """
    Predict a backchannel if the probability is higher than a decreasing
    threshold. When beckchannel is predicted the threshold resets to 1.
    """
    if timestamp is None:
        timestamp = pd.to_datetime('now')

    thresholds = state['thresholds']
    predictions = []

    for id_, proba in zip(ids, probas):

        try:
            old_timestamp, threshold = thresholds[id_]
        except KeyError:  # First prediction for this id
            prediction = False
            threshold = 1
        else:
            seconds_passed = (timestamp - old_timestamp).seconds
            threshold *= settings.DEKOK_DECREASE ** seconds_passed
            if proba > threshold:
                prediction = True
                threshold = 1
            else:
                prediction = False

        predictions.append(prediction)
        thresholds[id_] = (timestamp, threshold)

    assert len(predictions) == len(ids)
    return state, predictions


def update_state(state, id_, sample, timestamp):
    """
    Update the state with a new sample manually, without generating a
    prediction.

    Returns a new state.
    """
    try:
        df = state[id_]
    except KeyError:
        df = pd.DataFrame([sample], index=[timestamp])
    else:
        df.loc[timestamp] = sample

    millis_to_keep = settings.RESAMPLING_PERIOD * settings.WINDOW_SIZE
    up_to = timestamp - pd.to_timedelta(millis_to_keep, 'ms')
    state[id_] = df.loc[up_to:]
    return state


class BackchannelPredictor:
    """
    An OO wrapper around `new_state` and `predict` that maintain state.
    """
    def __init__(self, clf):
        self.clf = clf
        self._state = new_state()

    def predict(self, ids, samples, type_=None):
        new_state, predictions = predict(
            self._state, self.clf, ids, samples, type_=type_
        )
        self._state = new_state
        return predictions
