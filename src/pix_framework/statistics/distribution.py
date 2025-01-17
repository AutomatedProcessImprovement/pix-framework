import math
import sys
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Union, Optional

import numpy as np
import scipy.stats as st
from scipy.stats import wasserstein_distance

from pix_framework.statistics.utils import remove_outliers


class DistributionType(Enum):
    UNIFORM = "uniform"
    NORMAL = "norm"
    TRIANGULAR = "triang"
    EXPONENTIAL = "expon"
    LOG_NORMAL = "lognorm"
    GAMMA = "gamma"
    FIXED = "fix"

    @staticmethod
    def from_string(value: str) -> "DistributionType":
        name = value.lower()
        if name == "uniform":
            return DistributionType.UNIFORM
        elif name in ("norm", "normal"):
            return DistributionType.NORMAL
        elif name in ("triang", "triangular"):
            return DistributionType.TRIANGULAR
        elif name in ("expon", "exponential"):
            return DistributionType.EXPONENTIAL
        elif name in ("lognorm", "log_normal", "lognormal"):
            return DistributionType.LOG_NORMAL
        elif name == "gamma":
            return DistributionType.GAMMA
        elif name in ["fix", "fixed"]:
            return DistributionType.FIXED
        else:
            raise ValueError(f"Unknown distribution: {value}")


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
        name: Union[str, DistributionType] = "fix",
        mean: Optional[float] = None,
        var: Optional[float] = None,
        std: Optional[float] = None,
        minimum: Optional[float] = None,
        maximum: Optional[float] = None
    ):
        self.type = DistributionType.from_string(name) if isinstance(name, str) else name
        self.mean = mean
        self.var = var
        self.std = std
        self.min = minimum
        self.max = maximum

    def generate_sample(self, size: int) -> list:
        """
        Generates a sample of [size] elements following this (self) distribution parameters. The elements are
        positive, and within the limits [self.min, self.max].

        :param size: number of elements to add to the sample.
        :return: list with the elements of the sample.
        """
        # Instantiate empty sample list
        sample = []
        i = 0  # flag for iteration limit
        # Generate until full of values within limits
        while len(sample) < size and i < 100:
            # Generate missing elements
            local_sample = self._generate_raw_sample(size - len(sample))
            # Filter out negative and out of limits elements
            local_sample = [element for element in local_sample if element >= 0.0]
            if self.min is not None:
                local_sample = [element for element in local_sample if element >= self.min]
            if self.max is not None:
                local_sample = [element for element in local_sample if element <= self.max]
            # Add generated elements to sample
            sample += local_sample
            i += 1
        # Check if all elements got generated
        if len(sample) < size:
            default_value = self._replace_out_of_bounds_value()
            print(f"Warning when generating sample of distribution {self}. "
                  "Too many iterations generating durations out of the distribution limits! "
                  f"Filling missing values with default ({default_value})!")
            sample += [default_value] * (size - len(sample))
        # Return complete sample
        return sample

    def _generate_raw_sample(self, size: int) -> list:
        """
        Generates a sample of [size] elements following this (self) distribution parameters not ensuring that the
        returned elements are within the interval [self.min, self.max] and positive.

        :param size: number of elements to add to the sample.
        :return: list with the elements of the sample.
        """
        sample = []
        if self.type == DistributionType.FIXED:
            sample = [self.mean] * size
        elif self.type == DistributionType.EXPONENTIAL:
            # 'loc' displaces the samples, a loc=100 will be the same as adding 100 to each sample taken from a loc=1
            scale = self.mean - self.min
            if scale < 0.0:
                print("Warning! Trying to generate EXPON sample with 'mean' < 'min', using 'mean' as scale value.")
                scale = self.mean
            sample = st.expon.rvs(loc=self.min, scale=scale, size=size)
        elif self.type == DistributionType.NORMAL:
            sample = st.norm.rvs(loc=self.mean, scale=self.std, size=size)
        elif self.type == DistributionType.UNIFORM:
            sample = st.uniform.rvs(loc=self.min, scale=self.max - self.min, size=size)
        elif self.type == DistributionType.LOG_NORMAL:
            # If the distribution corresponds to a 'lognorm' with loc!=0, the estimation is done wrong
            # dunno how to take that into account
            pow_mean = pow(self.mean, 2)
            phi = math.sqrt(self.var + pow_mean)
            mu = math.log(pow_mean / phi)
            sigma = math.sqrt(math.log(phi ** 2 / pow_mean))
            sample = st.lognorm.rvs(sigma, loc=0, scale=math.exp(mu), size=size)
        elif self.type == DistributionType.GAMMA:
            # If the distribution corresponds to a 'gamma' with loc!=0, the estimation is done wrong
            # dunno how to take that into account
            sample = st.gamma.rvs(
                pow(self.mean, 2) / self.var,
                loc=0,
                scale=self.var / self.mean,
                size=size,
            )
        return sample

    def _replace_out_of_bounds_value(self):
        new_value = None
        if self.mean is not None and self.mean >= 0.0:
            # Set to mean
            new_value = self.mean
        if self.min is not None and self.max is not None:
            # If we have boundaries, check that mean is inside
            if new_value is None or new_value < self.min or new_value > self.max:
                # Invalid, set to middle between min and max
                new_value = self.min + ((self.max - self.min) / 2)
        if new_value is None or new_value < 0.0:
            new_value = 0.0
        # Return fixed value
        return new_value

    def scale_distribution(self, alpha: float) -> "DurationDistribution":
        return DurationDistribution(
            name=self.type,
            mean=self.mean * alpha,  # Mean: scaled by multiplying by [alpha]
            var=self.var * alpha * alpha,  # Variance: scaled by multiplying by [alpha]^2
            std=self.std * alpha,  # STD: scaled by multiplying by [alpha]
            minimum=self.min * alpha,  # Min: scaled by multiplying by [alpha]
            maximum=self.max * alpha,  # Max: scaled by multiplying by [alpha]
        )

    def to_prosimos_distribution(self) -> dict:
        """
        Specifies input to Prosimos
        """
        # Initialize empty list of params
        distribution_params = []
        # Add specific params depending on distribution
        if self.type == DistributionType.FIXED:
            distribution_params += [{"value": float(self.mean)}]  # fixed value
        elif self.type == DistributionType.EXPONENTIAL:
            distribution_params += [
                {"value": float(self.mean)},  # mean
                {"value": float(self.min)},  # min
                {"value": float(self.max)},  # max
            ]
        elif self.type == DistributionType.NORMAL:
            distribution_params += [
                {"value": float(self.mean)},  # loc
                {"value": float(self.std)},  # scale
                {"value": float(self.min)},  # min
                {"value": float(self.max)},  # max
            ]
        elif self.type == DistributionType.UNIFORM:
            distribution_params += [{"value": float(self.min)}, {"value": float(self.max)}]  # min (from)  # max (to)
        elif self.type == DistributionType.LOG_NORMAL:
            # If the distribution corresponds to a 'lognorm' with loc!=0, the estimation is done wrong
            # dunno how to take that into account
            distribution_params += [
                {"value": float(self.mean)},  # mean
                {"value": float(self.var)},  # variance
                {"value": float(self.min)},  # min
                {"value": float(self.max)},  # max
            ]
        elif self.type == DistributionType.GAMMA:
            # If the distribution corresponds to a 'gamma' with loc!=0, the estimation is done wrong
            # dunno how to take that into account
            distribution_params += [
                {"value": float(self.mean)},  # mean
                {"value": float(self.var)},  # variance
                {"value": float(self.min)},  # min
                {"value": float(self.max)},  # max
            ]
        else:
            raise ValueError(f"Unsupported distribution: {self}")
        # Return dict with the distribution data as expected by PROSIMOS
        return {
            "distribution_name": self.type.value,
            "distribution_params": distribution_params,
        }

    def to_qbp_distribution(self) -> QBPDurationDistribution:
        # Initialize empty distribution
        qbp_distribution = None
        # Parse distribution
        if self.type == DistributionType.FIXED:
            qbp_distribution = QBPDurationDistribution(type="FIXED", mean=str(self.mean), arg1="0", arg2="0")
        elif self.type == DistributionType.EXPONENTIAL:
            # For the XML mean=0 and arg2=0
            qbp_distribution = QBPDurationDistribution(type="EXPONENTIAL", mean="0", arg1=str(self.mean), arg2="0")
        elif self.type == DistributionType.NORMAL:
            # For the XML arg1=std and arg2=0
            qbp_distribution = QBPDurationDistribution(type="NORMAL", mean=str(self.mean), arg1=str(self.std), arg2="0")
        elif self.type == DistributionType.UNIFORM:
            # For the XML the mean is always 3600, arg1=min and arg2=max
            qbp_distribution = QBPDurationDistribution(
                type="UNIFORM", mean="3600", arg1=str(self.min), arg2=str(self.max)
            )
        elif self.type == DistributionType.LOG_NORMAL:
            # For the XML arg1=var and arg2=0
            qbp_distribution = QBPDurationDistribution(
                type="LOGNORMAL", mean=str(self.mean), arg1=str(self.var), arg2="0"
            )
        elif self.type == DistributionType.GAMMA:
            # For the XML arg1=var and arg2=0
            qbp_distribution = QBPDurationDistribution(type="GAMMA", mean=str(self.mean), arg1=str(self.var), arg2="0")
        # Return parsed distribution
        return qbp_distribution

    @staticmethod
    def from_dict(distribution_dict: dict) -> "DurationDistribution":
        """
        Deserialize function distribution provided as a dictionary
        """
        distr_name = DistributionType(distribution_dict["distribution_name"])
        distr_params = distribution_dict["distribution_params"]

        distribution = None

        if distr_name == DistributionType.FIXED:
            # no min and max values
            distribution = DurationDistribution(
                name=distr_name,
                mean=float(distr_params[0]["value"]),
            )
        elif distr_name == DistributionType.EXPONENTIAL:
            # TODO: discuss whether we need to differentiate min and scale.
            # right now, min is used for calculating the scale
            distribution = DurationDistribution(
                name=distr_name,
                mean=float(distr_params[0]["value"]),
                minimum=float(distr_params[1]["value"]),
                maximum=float(distr_params[2]["value"]),
            )
        elif distr_name == DistributionType.UNIFORM:
            distribution = DurationDistribution(
                name=distr_name,
                minimum=float(distr_params[0]["value"]),
                maximum=float(distr_params[1]["value"]),
            )
        elif distr_name == DistributionType.NORMAL:
            distribution = DurationDistribution(
                name=distr_name,
                mean=float(distr_params[0]["value"]),
                std=float(distr_params[1]["value"]),
                minimum=float(distr_params[2]["value"]),
                maximum=float(distr_params[3]["value"]),
            )
        elif distr_name == DistributionType.LOG_NORMAL:
            distribution = DurationDistribution(
                name=distr_name,
                mean=float(distr_params[0]["value"]),
                var=float(distr_params[1]["value"]),
                minimum=float(distr_params[2]["value"]),
                maximum=float(distr_params[3]["value"]),
            )
        elif distr_name == DistributionType.GAMMA:
            distribution = DurationDistribution(
                name=distr_name,
                mean=float(distr_params[0]["value"]),
                var=float(distr_params[1]["value"]),
                minimum=float(distr_params[2]["value"]),
                maximum=float(distr_params[3]["value"]),
            )
        return distribution

    def to_simple_function_call(self) -> str:
        if self.type == DistributionType.FIXED:
            return f"fix({self.mean})"
        elif self.type == DistributionType.UNIFORM:
            return f"uniform({self.min}, {self.max})"
        elif self.type == DistributionType.NORMAL:
            return f"norm({self.mean}, {self.std}, {self.min}, {self.max})"
        elif self.type == DistributionType.EXPONENTIAL:
            return f"expon({self.mean}, {self.min}, {self.max})"
        elif self.type == DistributionType.LOG_NORMAL:
            return f"lognorm({self.mean}, {self.var}, {self.min}, {self.max})"
        elif self.type == DistributionType.GAMMA:
            return f"gamma({self.mean}, {self.var}, {self.min}, {self.max})"
        else:
            raise ValueError(f"Unsupported distribution: {self.type}")

    def __str__(self):
        return "DurationDistribution(name: {}, mean: {}, var: {}, std: {}, min: {}, max: {})".format(
            self.type.value, self.mean, self.var, self.std, self.min, self.max
        )


def get_best_fitting_distribution(
    data: list,
    filter_outliers: bool = False,
    outlier_threshold: float = 20.0,
) -> DurationDistribution:
    """
    Discover the distribution (exponential, normal, uniform, log-normal, and gamma) that best fits the values in [data].

    :param data: Values to fit a distribution for.
    :param filter_outliers: If true, remove outliers from the sample.
    :param outlier_threshold: Threshold to consider an observation an outlier. Increasing this outlier increases the
                              flexibility of the detection method, i.e., an observation needs to be further from the
                              mean to be considered as outlier.

    :return: the best fitting distribution.
    """
    # Filter outliers
    filtered_data = remove_outliers(data, outlier_threshold) if filter_outliers else data
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
            DurationDistribution("uniform", mean, var, std, d_min, d_max),
        ]
        if mean != 0:
            dist_candidates += [DurationDistribution("lognorm", mean, var, std, d_min, d_max)]
            # if var != 0:
            #     dist_candidates += [DurationDistribution("gamma", mean, var, std, d_min, d_max)]
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


def _check_fix(data: list, delta=5):
    value = None
    counter = Counter(data)
    counter[None] = 0
    for d1 in counter:
        if (counter[d1] > counter[value]) and (sum([abs(d1 - d2) < delta for d2 in data]) / len(data) > 0.95):
            # If the value [d1] is more frequent than the current fixed one [value]
            # and
            # the ratio of values similar (or with a difference lower than [delta]) to [d1] is more than 90%
            # update value
            value = d1
    # Return fixed value with more apparitions
    return value


def get_observations_histogram(
    data: list,
    num_bins: int = 20,
    filter_outliers: bool = False,
    outlier_threshold: float = 20.0,
) -> dict:
    """
    Build a histogram with the values in [data], with [num_bins] bins. It builds the histogram, computes the CDF and the values of each
    bin of the CDF.

    :param data: Data to build the histogram.
    :param num_bins: Number of bins to use in the histogram.
    :param filter_outliers: If true, remove outliers from the sample.
    :param outlier_threshold: Threshold to consider an observation an outlier. Increasing this outlier increases the
                              flexibility of the detection method, i.e., an observation needs to be further from the
                              mean to be considered as outlier.

    :return: A dict with the histogram in Prosimos format, storing the CDF values and middle points.
    """
    filtered_durations = remove_outliers(data, outlier_threshold) if filter_outliers else data
    bins = np.linspace(min(filtered_durations), max(filtered_durations), num_bins + 1)
    hist, _ = np.histogram(filtered_durations, bins=bins)
    cdf = np.cumsum(hist)
    cdf = cdf / cdf[-1]
    bin_midpoints = (bins[:-1] + bins[1:]) / 2
    return {
        "distribution_name": "histogram_sampling",
        "histogram_data": {
            "cdf": [float(num) for num in cdf],
            "bin_midpoints": [float(num) for num in bin_midpoints],
        },
    }
