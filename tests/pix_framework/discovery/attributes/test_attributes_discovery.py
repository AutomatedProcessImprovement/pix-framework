import numpy as np
import pandas as pd
import pytest
import logging
import json
from scipy import stats

from pix_framework.discovery.attributes.attribute_discovery import discover_attributes
from pix_framework.io.event_log import EventLogIDs
from pix_framework.statistics.distribution import DurationDistribution

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def log_pretty_json(json_object):
    formatted_json_string = json.dumps(json_object, indent=4)
    logger.info(formatted_json_string)


SAMPLE_SIZE = 10000
CONFIDENCE_THRESHOLD = 0.95
DISTRIBUTION_SAMPLE_SIZE = 10000

# Custom LOG_IDS /doesn't match any of pre-default schemas from prosimos (enable_time)/
PROSIMOS_LOG_IDS = EventLogIDs(
    case="case_id",
    activity="activity",
    enabled_time="enable_time",
    start_time="start_time",
    end_time="end_time",
    resource="resource",
)

CASE_ATTRIBUTES_FILE_PATH = "tests/pix_framework/assets/case_attributes_log.csv.gz"
EVENT_ATTRIBUTES_FILE_PATH = "tests/pix_framework/assets/event_attributes_log.csv.gz"
CASE_AND_EVENT_ATTRIBUTES_FILE_PATH = "tests/pix_framework/assets/case_and_event_attributes_log.csv.gz"

files_to_discover = [
    CASE_ATTRIBUTES_FILE_PATH,
    EVENT_ATTRIBUTES_FILE_PATH,
    CASE_AND_EVENT_ATTRIBUTES_FILE_PATH
]

# Expected values for each file_to_discover file
expected_values_for_files = {
    CASE_ATTRIBUTES_FILE_PATH: {
        "case_attributes": {
            "application_status": {
                "key": {
                    "valid": 0.9,
                    "invalid": 0.1
                },
                "threshold": 0.05
            },
            "fixed_value": {
                "name": "fixed",
                "value": 240,
                "threshold": 0.05
            },
            "expon_distribution": {
                "name": "expon",
                "value": [75.0, 25.0, 100.0],
                "threshold": 0.05
            },
            "normal_distribution": {
                "name": "norm",
                "value": [50, 15, 25, 100],
                "threshold": 0.05
            },
            "uniform_distribution": {
                "name": "uniform",
                "value": [25, 100],
                "threshold": 0.05
            }
        }
    },
    EVENT_ATTRIBUTES_FILE_PATH: {
        "event_attributes": {
            "Return application back to applicant": {
                "application_status": {
                    "key": {
                        "valid": 0.75,
                        "invalid": 0.25
                    },
                    "threshold": 0.05
                },
                "fixed_value": {
                    "name": "fixed",
                    "value": 330.0,
                    "threshold": 0.05
                }
            },
            "Assess loan risk": {
                "expon_distribution": {
                    "name": "expon",
                    "value": [150.0, 50.0, 200.0],
                    "threshold": 0.05
                },
                "normal_distribution": {
                    "name": "norm",
                    "value": [50.0, 15.0, 25.0, 100.0],
                    "threshold": 0.05
                },
                "uniform_distribution": {
                    "name": "norm",
                    "value": [25.0, 100.0],
                    "threshold": 0.05
                }
            }
        }
    },
    CASE_AND_EVENT_ATTRIBUTES_FILE_PATH: {
        "case_attributes": {
            "application_status": {
                "key": {
                    "valid": 0.9,
                    "invalid": 0.1
                },
                "threshold": 0.05
            },
            "fixed_value": {
                "name": "fix",
                "value": 240.0,
                "threshold": 0.05
            },
            "expon_distribution": {
                "name": "expon",
                "value": [75.0, 25.0, 100.0],
                "threshold": 0.05
            },
            "normal_distribution": {
                "name": "norm",
                "value": [50.0, 15.0, 25.0, 100.0],
                "threshold": 0.05
            },
            "uniform_distribution": {
                "name": "uniform",
                "value": [10.0, 300.0],
                "threshold": 0.05
            }
        },
        "event_attributes": {
            "Return application back to applicant": {
                "application_status_2": {
                    "key": {
                        "valid": 0.75,
                        "invalid": 0.25
                    },
                    "threshold": 0.05
                },
                "fixed_value_2": {
                    "name": "fixed",
                    "value": 330.0,
                    "threshold": 0.05
                }
            },
            "Assess loan risk": {
                "expon_distribution_2": {
                    "name": "expon",
                    "value": [150.0, 50.0, 200.0],
                    "threshold": 0.05
                },
                "normal_distribution_2": {
                    "name": "norm",
                    "value": [150.0, 45.0, 75.0, 300.0],
                    "threshold": 0.05
                },
                "uniform_distribution_2": {
                    "name": "uniform",
                    "value": [1.0, 500.0],
                    "threshold": 0.05
                }
            }
        }
    }
}


def fetch_attributes_from_file(file_name, size=None, confidence_threshold=1.0):
    event_log = pd.read_csv(file_name, compression='gzip')

    if size or size != 0:
        subset_cases = event_log.drop_duplicates(subset='case_id').head(size)
        event_log = event_log[event_log['case_id'].isin(subset_cases['case_id'])]

    attributes = discover_attributes(event_log, log_ids=PROSIMOS_LOG_IDS, confidence_threshold=confidence_threshold)
    return attributes


def validate_discrete(attribute, expected_values):
    # Validate param structure
    assert 'name' in attribute
    assert attribute['name'] in expected_values, \
        f"Attribute {attribute['name']} not found in expected values. Expected: {list(expected_values.keys())}"
    assert 'type' in attribute and attribute['type'] == 'discrete'
    assert 'values' in attribute

    threshold = expected_values[attribute['name']].get('threshold', 0.01)
    total_probability = 0
    expected_keys = set(expected_values[attribute['name']]['key'].keys())
    present_keys = set(value['key'] for value in attribute['values'])

    # Validate param names
    assert present_keys.issubset(expected_keys), \
        f"Unexpected keys found. Expected: {expected_keys}, but got: {present_keys}"
    assert expected_keys.issubset(present_keys), \
        f"Missing keys in attribute. Expected: {expected_keys}, but only found: {present_keys}"

    # Validate value structure and values
    for value in attribute['values']:
        assert 'key' in value
        assert 'value' in value
        total_probability += value['value']

        expected_probability = expected_values[attribute['name']]['key'].get(value['key'], None)
        if expected_probability is not None:
            assert abs(value['value'] - expected_probability) <= threshold, \
                f"For key {value['key']}, expected probability: {expected_probability}, but got: {value['value']}"

    # Validate total accuracy with threshold
    assert abs(total_probability - 1) <= threshold, \
        f"Total probabilities discrepancy. Expected sum close to 1, but got: {total_probability}"


def validate_fixed(attribute, expected_values):
    # Validate param structure
    assert 'name' in attribute
    assert attribute['name'] in expected_values, f"Attribute {attribute['name']} not found in expected values"
    assert 'type' in attribute and attribute['type'] == 'continuous'
    assert 'values' in attribute
    assert attribute['values']['distribution_name'] == 'fix'
    assert len(attribute['values']['distribution_params']) == 1

    threshold = expected_values[attribute['name']].get('threshold', 0.01)
    expected_value = expected_values[attribute['name']].get("value", None)

    # Validate total accuracy with threshold
    if expected_value is not None:
        assert abs(attribute['values']['distribution_params'][0]['value'] - expected_value) / abs(
            expected_value) <= threshold, \
            f"Got {attribute['values']['distribution_params'][0]['value']} but expected {expected_value}"


def get_param(params, index, default=None):
    # Helper function to get array values with default values
    try:
        return params[index]
    except IndexError:
        return default


def create_duration_distribution(name, params):
    # Map and create DurationDistribution
    distribution_mappings = {
        "expon": {
            "mean": get_param(params, 0),
            "minimum": get_param(params, 1),
            "maximum": get_param(params, 2)
        },
        "norm": {
            "mean": get_param(params, 0),
            "std": get_param(params, 1),
            "minimum": get_param(params, 2),
            "maximum": get_param(params, 3)
        },
        "uniform": {
            "minimum": get_param(params, 0),
            "maximum": get_param(params, 1)
        },
        "lognorm": {
            "mean": get_param(params, 0),
            "var": get_param(params, 1),
            "minimum": get_param(params, 2),
            "maximum": get_param(params, 3)
        },
        "gamma": {
            "mean": get_param(params, 0),
            "var": get_param(params, 1),
            "minimum": get_param(params, 2),
            "maximum": get_param(params, 3)
        }
    }

    if name not in distribution_mappings:
        raise ValueError(f"Unsupported distribution: {name}")

    return DurationDistribution(name=name, **distribution_mappings[name])


def validate_distribution(attribute, expected_values):
    # Get actual distribution name & values
    distribution_name = attribute["values"]["distribution_name"]
    actual_params_values = [param["value"] for param in attribute["values"]["distribution_params"]]

    distribution_actual = create_duration_distribution(distribution_name, actual_params_values)

    # Get expected distribution name & values
    expected_distribution_name = expected_values.get(attribute["name"], {}).get("name", [])
    expected_params_values = expected_values.get(attribute["name"], {}).get("value", [])
    expected_threshold = expected_values.get(attribute["name"], {}).get("threshold", [])

    distribution_expected = create_duration_distribution(expected_distribution_name, expected_params_values)

    # Generate samples based on distributions
    sample_generated = distribution_actual.generate_sample(DISTRIBUTION_SAMPLE_SIZE)
    sample_expected = distribution_expected.generate_sample(DISTRIBUTION_SAMPLE_SIZE)

    # Assert by mean
    actual_mean = np.mean(sample_generated)
    expected_mean = np.mean(sample_expected)
    mean_diff = abs(actual_mean - expected_mean) / expected_mean

    if mean_diff <= expected_threshold:
        logger.info(distribution_actual)
        logger.info(actual_mean)
        logger.info(distribution_expected)
        logger.info(expected_mean)
        logger.info(f"MEAN DIFF {attribute['name']}: {mean_diff}")
    assert mean_diff <= expected_threshold, \
        f"Mean difference between generated and expected samples for attriubte {attribute['name']} is {mean_diff}, which is higher than a threshold.{expected_threshold}"

    # Assert by Chi-squared test
    hist_generated, bins = np.histogram(sample_generated, bins=10, density=True)
    hist_expected, _ = np.histogram(sample_expected, bins=bins, density=True)

    chi2, p_value_chi2 = stats.chisquare(hist_generated, hist_expected)
    assert p_value_chi2 > 0.1, \
        f"Generated histogram for {attribute['name']} does not match expected histogram. Chi-squared p-value: {p_value_chi2}"


def validate_case_attributes(attributes, expected_values):
    # Validation flow for case attributes
    for attribute in attributes:
        if attribute['type'] == 'discrete':
            validate_discrete(attribute, expected_values)
        elif attribute['type'] == 'continuous':
            if attribute['values']['distribution_name'] == 'fix':
                validate_fixed(attribute, expected_values)
            else:
                validate_distribution(attribute, expected_values)


def validate_event_attributes(attributes, expected_values):
    # Validation flow for event attributes
    for event in attributes:
        assert event["event_id"] in expected_values, f"Event {event['event_id']} not found in expected event attributes"

        expected_event = expected_values[event["event_id"]]

        for attribute in event["attributes"]:
            if attribute['type'] == 'discrete':
                validate_discrete(attribute, expected_event)
            elif attribute['type'] == 'continuous':
                if attribute['values']['distribution_name'] == 'fix':
                    validate_fixed(attribute, expected_event)
                else:
                    validate_distribution(attribute, expected_event)


@pytest.mark.parametrize("file_name", files_to_discover)
def test__attributes(file_name):
    attributes = fetch_attributes_from_file(file_name, SAMPLE_SIZE, CONFIDENCE_THRESHOLD)

    log_pretty_json(attributes)

    file_expected_values = expected_values_for_files[file_name]
    expected_case_attributes = file_expected_values.get("case_attributes", {})
    expected_event_attributes = file_expected_values.get("event_attributes", {})

    validate_case_attributes(attributes.get("case_attributes", []), expected_case_attributes)
    validate_event_attributes(attributes.get("event_attributes", []), expected_event_attributes)


