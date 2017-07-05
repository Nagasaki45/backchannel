import numpy as np
import pandas as pd

from training import _reshape_data_for_lstm
from training import to_samples


def test_to_samples():
    example = pd.DataFrame([
        [1, 100],
        [2, 200],
        [3, 300],
        [4, 400],
    ])
    expected_2 = np.array([
        [1, 2, 200],
        [2, 3, 300],
        [3, 4, 400],
    ])
    expected_3 = np.array([
        [1, 2, 3, 300],
        [2, 3, 4, 400],
    ])
    np.testing.assert_array_equal(to_samples(example, n=2), expected_2)
    np.testing.assert_array_equal(to_samples(example, n=3), expected_3)


def test_reshape_data_for_lstm():
    original_X = np.array([
        [10, 11, 20, 21, 30, 31],
        [20, 21, 30, 31, 40, 41],
        [30, 31, 40, 41, 50, 51],
    ])
    original_y = np.array([0, 1, 2])

    expected_X = np.array([
        [
            [10, 11],
            [20, 21],
            [30, 31],
        ],
        [
            [20, 21],
            [30, 31],
            [40, 41],
        ],
        [
            [30, 31],
            [40, 41],
            [50, 51],
        ],
    ])
    expected_y = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ])

    X, y = _reshape_data_for_lstm(original_X, original_y, window_size=3)
    np.testing.assert_array_equal(X, expected_X)
    np.testing.assert_array_equal(y, expected_y)
