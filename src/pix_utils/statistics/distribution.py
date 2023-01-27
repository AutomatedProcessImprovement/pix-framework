import math
import sys
from collections import Counter
from dataclasses import dataclass

import numpy as np
import scipy.stats as st
from scipy.stats import wasserstein_distance


@dataclass
class QBPDurationDistribution:
    type: str = "NORMAL"
    mean: str = "NaN"  # Warning! these values are always interpreted as seconds
    arg1: str = "NaN"
    arg2: str = "NaN"
    unit: str = "seconds"  # This is the unit to show in the interface by transforming the values in seconds


class DurationDistribution:
    def __init__(
            self,
            name: str = "fix",  # supported 'fix', 'expon', 'norm', 'uniform', 'lognorm', and 'gamma'
            mean: float = 0.0, var: float = 0.0, std: float = 0.0,
            minimum: float = 0.0, maximum: float = 0.0,
    ):
        self.name = name
        self.mean = mean
        self.var = var
        self.std = std
        self.min = minimum
        self.max = maximum

    def generate_sample(self, size: int) -> list:
        sample = []
        if self.name == "fix":
            sample = [self.mean] * size
        elif self.name == "expon":
            # 'loc' displaces the samples, a loc=100 will be the same as adding 100 to each sample taken from a loc=1
            sample = st.expon.rvs(loc=self.min, scale=self.mean - self.min, size=size)
        elif self.name == "norm":
            sample = st.norm.rvs(loc=self.mean, scale=self.std, size=size)
        elif self.name == "uniform":
            sample = st.uniform.rvs(loc=self.min, scale=self.max - self.min, size=size)
        elif self.name == "lognorm":
            # If the distribution corresponds to a 'lognorm' with loc!=0, the estimation is done wrong
            # dunno how to take that into account
            pow_mean = pow(self.mean, 2)
            phi = math.sqrt(self.var + pow_mean)
            mu = math.log(pow_mean / phi)
            sigma = math.sqrt(math.log(phi ** 2 / pow_mean))
            sample = st.lognorm.rvs(sigma, loc=0, scale=math.exp(mu), size=size)
        elif self.name == "gamma":
            # If the distribution corresponds to a 'gamma' with loc!=0, the estimation is done wrong
            # dunno how to take that into account
            sample = st.gamma.rvs(pow(self.mean, 2) / self.var, loc=0, scale=self.var / self.mean, size=size)
        # Return generated sample
        return sample

    def scale_distribution(self, alpha: float) -> 'DurationDistribution':
        return DurationDistribution(
            name=self.name,
            mean=self.mean * alpha,  # Mean: scaled by multiplying by [alpha]
            var=self.var * alpha * alpha,  # Variance: scaled by multiplying by [alpha]^2
            std=self.std * alpha,  # STD: scaled by multiplying by [alpha]
            minimum=self.min * alpha,  # Min: scaled by multiplying by [alpha]
            maximum=self.max * alpha  # Max: scaled by multiplying by [alpha]
        )

    def to_prosimos_distribution(self) -> dict:
        # Initialize empty list of params
        distribution_params = []
        # Add specific params depending on distribution
        if self.name == "fix":
            distribution_params += [
                {'value': self.mean}  # fixed value
            ]
        elif self.name == "expon":
            distribution_params += [
                {'value': self.min},  # loc
                {'value': self.mean - self.min},  # scale
                {'value': self.min},  # min
                {'value': self.max}  # max
            ]
        elif self.name == "norm":
            distribution_params += [
                {'value': self.mean},  # loc
                {'value': self.std},  # scale
                {'value': self.min},  # min
                {'value': self.max}  # max
            ]
        elif self.name == "uniform":
            distribution_params += [
                {'value': self.min},  # loc
                {'value': self.max - self.min},  # scale
                {'value': self.min},  # min
                {'value': self.max}  # max
            ]
        elif self.name == "lognorm":
            # If the distribution corresponds to a 'lognorm' with loc!=0, the estimation is done wrong
            # dunno how to take that into account
            pow_mean = pow(self.mean, 2)
            phi = math.sqrt(self.var + pow_mean)
            mu = math.log(pow_mean / phi)
            sigma = math.sqrt(math.log(phi ** 2 / pow_mean))
            distribution_params += [
                {'value': sigma},  # sigma
                {'value': 0},  # loc
                {'value': math.exp(mu)},  # scale
                {'value': self.min},  # min
                {'value': self.max}  # max
            ]
        elif self.name == "gamma":
            # If the distribution corresponds to a 'gamma' with loc!=0, the estimation is done wrong
            # dunno how to take that into account
            distribution_params += [
                {'value': pow(self.mean, 2) / self.var},  # a
                {'value': 0},  # loc
                {'value': self.var / self.mean},  # scale
                {'value': self.min},  # min
                {'value': self.max}  # max
            ]
        # Return dict with the distribution data as expected by PROSIMOS
        return {'distribution_name': self.name, 'distribution_params': distribution_params}

    def to_qbp_distribution(self) -> QBPDurationDistribution:
        # Initialize empty distribution
        qbp_distribution = None
        # Parse distribution
        if self.name == 'fix':
            qbp_distribution = QBPDurationDistribution(
                type="FIXED",
                mean=str(self.mean),
                arg1="0",
                arg2="0"
            )
        elif self.name == 'expon':
            # For the XML mean=0 and arg2=0
            qbp_distribution = QBPDurationDistribution(
                type="EXPONENTIAL",
                mean="0",
                arg1=str(self.mean),
                arg2="0"
            )
        elif self.name == 'norm':
            # For the XML arg1=std and arg2=0
            qbp_distribution = QBPDurationDistribution(
                type="NORMAL",
                mean=str(self.mean),
                arg1=str(self.std),
                arg2="0"
            )
        elif self.name == 'uniform':
            # For the XML the mean is always 3600, arg1=min and arg2=max
            qbp_distribution = QBPDurationDistribution(
                type="UNIFORM",
                mean="3600",
                arg1=str(self.min),
                arg2=str(self.max)
            )
        elif self.name == 'lognorm':
            # For the XML arg1=var and arg2=0
            qbp_distribution = QBPDurationDistribution(
                type="LOGNORMAL",
                mean=str(self.mean),
                arg1=str(self.var),
                arg2="0"
            )
        elif self.name == 'gamma':
            # For the XML arg1=var and arg2=0
            qbp_distribution = QBPDurationDistribution(
                type="GAMMA",
                mean=str(self.mean),
                arg1=str(self.var),
                arg2="0"
            )
        # Return parsed distribution
        return qbp_distribution


def get_best_fitting_distribution(data: list, remove_outliers: bool = False) -> DurationDistribution:
    """
    Discover the distribution (exponential, normal, uniform, log-normal, and gamma) that best fits the values in [data].

    :param data:            Values to fit a distribution for.
    :param remove_outliers: If true, remove outliers from the sample.

    :return: the best fitting distribution.
    """
    # Filter outliers
    filtered_data = _reject_outliers(data) if remove_outliers else data
    # Check for fixed value
    fix_value = _check_fix(filtered_data)
    if fix_value is not None:
        # If it is a fixed value, infer distribution
        distribution = DurationDistribution("fix", fix_value, 0.0, 0.0, fix_value, fix_value)
    else:
        # Otherwise, compute basic statistics and try with other distributions
        mean = np.mean(filtered_data)
        var = np.var(filtered_data)
        std = np.std(filtered_data)
        d_min = min(filtered_data)
        d_max = max(filtered_data)
        # Create distribution candidates
        dist_candidates = [
            DurationDistribution("expon", mean, var, std, d_min, d_max),
            DurationDistribution("norm", mean, var, std, d_min, d_max),
            DurationDistribution("uniform", mean, var, std, d_min, d_max)
        ]
        if mean != 0:
            dist_candidates += [DurationDistribution("lognorm", mean, var, std, d_min, d_max)]
            if var != 0:
                dist_candidates += [DurationDistribution("gamma", mean, var, std, d_min, d_max)]
        # Search for the best one within the candidates
        best_distribution = None
        best_emd = sys.float_info.max
        for distribution_candidate in dist_candidates:
            # Generate a list of observations from the distribution
            generated_data = distribution_candidate.generate_sample(len(filtered_data))
            # Compute its distance with the observed data
            emd = wasserstein_distance(filtered_data, generated_data)
            # Update the best distribution if better
            if emd < best_emd:
                best_emd = emd
                best_distribution = distribution_candidate
        # Set the best distribution as the one to return
        distribution = best_distribution
    # Return best distribution
    return distribution


def _reject_outliers(data, m=8.):
    # https://stackoverflow.com/a/16562028
    data = np.asarray(data)
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d / mdev if mdev else 0.
    return data[s < m]


def _check_fix(data_list, delta=5):
    value = None
    counter = Counter(data_list)
    counter[None] = 0
    for d1 in counter:
        if (counter[d1] > counter[value]) and (sum([abs(d1 - d2) < delta for d2 in data_list]) / len(data_list) > 0.95):
            # If the value [d1] is more frequent than the current fixed one [value]
            # and
            # the ratio of values similar (or with a difference lower than [delta]) to [d1] is more than 90%
            # update value
            value = d1
    # Return fixed value with more apparitions
    return value


def get_observations_histogram(data: list, num_bins: int = 20, remove_outliers: bool = False) -> dict:
    """
    Build a histogram with the values in [data], with [num_bins] bins. It builds the histogram, computes the CDF and the values of each
    bin of the CDF.

    :param data:            Data to build the histogram.
    :param num_bins:        Number of bins to use in the histogram.
    :param remove_outliers: If true, remove outliers from the sample.

    :return: A dict with the histogram in Prosimos format, storing the CDF values and middle points.
    """
    filtered_durations = _reject_outliers(data) if remove_outliers else data
    bins = np.linspace(min(filtered_durations), max(filtered_durations), num_bins + 1)
    hist, _ = np.histogram(filtered_durations, bins=bins)
    cdf = np.cumsum(hist)
    cdf = cdf / cdf[-1]
    bin_midpoints = (bins[:-1] + bins[1:]) / 2
    return {
        'distribution_name': "histogram_sampling",
        'histogram_data': {
            'cdf': [float(num) for num in cdf],
            'bin_midpoints': [float(num) for num in bin_midpoints]
        }
    }
