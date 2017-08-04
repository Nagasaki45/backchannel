import math
from unittest import mock

import numpy as np
import pandas as pd

import predict


@mock.patch('predict.settings')
def test_prepare_data_for_prediction(settings_mock):
    settings_mock.RESAMPLING_PERIOD = 500  # millis
    settings_mock.WINDOW_SIZE = 4

    df = pd.DataFrame([
        [1, 10],
        [2, 20],
        [3, 30],
    ])
    df.index = pd.to_datetime([f'01.01.2000 00:00:0{x}' for x in range(3)])

    data = {0: df}  # id to pd.DataFrame
    sample = np.array([4, 40])
    timestamp = pd.to_datetime('01.01.2000 00:00:03')

    expected = np.array([
        [2, 20, 3, 30, 3, 30, 4, 40]
    ])

    _, result = predict.prepare_data_for_prediction(data, [0], [sample], timestamp)
    np.testing.assert_equal(result, expected)


@mock.patch('predict.settings')
def test_prepare_data_for_prediction_not_enough_samples(settings_mock):
    settings_mock.RESAMPLING_PERIOD = 1000  # millis
    settings_mock.WINDOW_SIZE = 4

    df = pd.DataFrame([
        [1, 10]
    ])
    df.index = pd.to_datetime(['01.01.2000 00:00:00'])

    data = {0: df}  # id to pd.DataFrame
    sample = np.array([2, 20])
    timestamp = pd.to_datetime('01.01.2000 00:00:01')

    expected = np.array([
        [0, 0, 0, 0, 1, 10, 2, 20]
    ])

    _, result = predict.prepare_data_for_prediction(data, [0], [sample], timestamp)
    np.testing.assert_equal(result, expected)


@mock.patch('predict.settings')
def test_prepare_data_for_prediction_id_not_in_data(settings_mock):
    settings_mock.RESAMPLING_PERIOD = 1000  # millis
    settings_mock.WINDOW_SIZE = 2

    data = {}
    ids = [0]
    samples = np.array([[1, 10]])
    timestamp = pd.to_datetime('01.01.2000 00:00:01')

    expected = np.array([[0, 0, 1, 10]])

    _, result = predict.prepare_data_for_prediction(data, ids, samples, timestamp)
    np.testing.assert_equal(result, expected)


def test_update_state_mutating_data_correctly():
    state = predict.new_state()
    sample = np.array([1, 10])
    timestamp = pd.to_datetime('01.01.2000 00:00:01')

    state = predict.update_state(state, 0, sample, timestamp)

    assert 0 in state
    df = state[0]
    assert len(df) == 1
    np.testing.assert_equal(df.loc[timestamp], sample)


def test_bug_in_X_shape():
    """
    It was due to X indexed by listener ID and not by running index.
    Calling the function with id == 1 exposed the bug.
    """
    data = {}
    ids = [1]
    samples = np.array([[1, 10, 100]])

    predict.prepare_data_for_prediction(data, ids, samples)


def test_no_input_data_bug():
    """
    Don't crash if the input data is empty, just respond empty results.
    """
    state = predict.new_state()
    clf = mock.Mock()
    _, prediction = predict.predict(state, clf, [], [])
    assert len(prediction) == 0


def test_dekok_first_prediction():
    state = predict.new_state()
    ids = [1]
    probas = [0.5]  # Doesn't really matter
    _, predictions = predict._dekok(state, ids, probas)
    assert predictions == [0]


def run_dekok_twice():
    state = predict.new_state()
    ids = [1]

    # First one populate the state
    timestamp = pd.to_datetime('01.01.2000 00:00:00')
    state, predictions = predict._dekok(state, ids, [0.5], timestamp)

    # After 1 second it should be positive
    timestamp = pd.to_datetime('01.01.2000 00:00:01')
    return predict._dekok(state, ids, [0.5], timestamp)


@mock.patch('predict.settings')
def test_dekok_predict_positive(settings_mock):
    # Fall to DEKOK_DECREASE of the previous threshold each second
    settings_mock.DEKOK_DECREASE = 0.4
    settings_mock.MIN_THRESHOLD = 0
    settings_mock.MAX_THRESHOLD = 1
    _, predictions = run_dekok_twice()
    assert predictions == [1]


@mock.patch('predict.settings')
def test_dekok_predict_negative(settings_mock):
    settings_mock.DEKOK_DECREASE = 0.6
    settings_mock.MIN_THRESHOLD = 0
    settings_mock.MAX_THRESHOLD = 1
    _, predictions = run_dekok_twice()
    assert predictions == [0]


@mock.patch('predict.settings')
def test_dekok_positive_resets_the_threshold(settings_mock):
    settings_mock.DEKOK_DECREASE = 0.4
    settings_mock.MIN_THRESHOLD = 0
    settings_mock.MAX_THRESHOLD = 1
    state, _ = run_dekok_twice()  # should end up positive
    _, threshold = state['thresholds'][1]
    assert threshold == 1


def test_map_value():
    cases = [
        {'value': 1, 'min_': 0, 'max_': 1, 'expected': 1},
        {'value': 1, 'min_': 0, 'max_': 0.5, 'expected': 0.5},
        {'value': 0.5, 'min_': 0, 'max_': 0.5, 'expected': 0.25},
        {'value': 0.5, 'min_': 0.1, 'max_': 0.5, 'expected': 0.3},
    ]
    for case in cases:
        expected = case.pop('expected')
        assert math.isclose(predict.map_value(**case), expected)
