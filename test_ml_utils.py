import numpy as np
import pandas as pd

from ml_utils import to_samples


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
