import numpy as np


def remove_outliers(data: list, m=5.0) -> list:
    """
    Remove outliers from a list of values following the approach presented in https://stackoverflow.com/a/16562028.
    :param data: list of values.
    :param m: maximum ratio between the difference (value - median) and the median of these differences, to NOT be
    considered an outlier. Decreasing the [m] ratio increases the number of detected outliers (observations closer
    to the median are considered as outliers).
    :return: the received list of values without the outliers.
    """
    # Compute distance of each value from the median
    data = np.asarray(data)
    d = np.abs(data - np.median(data))
    # Compute the median of these distances
    mdev = np.median(d)
    # Compute the ratio between each distance and the median of distances
    s = d / (mdev if mdev else 1.0)
    # Keep values with a ratio lower than the specified threshold
    return data[s < m].tolist()
