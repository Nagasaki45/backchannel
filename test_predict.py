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

    result = predict.prepare_data_for_prediction(data, [0], [sample], timestamp)
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

    result = predict.prepare_data_for_prediction(data, [0], [sample], timestamp)
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

    result = predict.prepare_data_for_prediction(data, ids, samples, timestamp)
    np.testing.assert_equal(result, expected)


def test_update_data_mutating_data_correctly():
    data = {}
    sample = np.array([1, 10])
    timestamp = pd.to_datetime('01.01.2000 00:00:01')

    predict.update_data(data, 0, sample, timestamp)

    assert 0 in data
    df = data[0]
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
