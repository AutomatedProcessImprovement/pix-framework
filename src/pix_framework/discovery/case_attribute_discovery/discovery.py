from statistics import mean

import pandas as pd
from pandas.api.types import is_numeric_dtype

from pix_framework.io.event_log import EventLogIDs
from pix_framework.statistics.distribution import get_best_fitting_distribution


def discover_case_attributes(
    event_log: pd.DataFrame, log_ids: EventLogIDs, avoid_columns: list = None, confidence_threshold: float = 1.0
) -> list:
    """
    Get the attributes in the event log that could be a case attribute, i.e. those attributes which value is fixed through all the process
    case.

    :param event_log:               event log to analyze.
    :param log_ids:                 mapping for the IDs of the columns in the event log.
    :param avoid_columns:           list of columns to not consider during the analysis.
    :param confidence_threshold:    confidence value for an attribute to be considered a case attribute (average percentage of events within
                                    a case where the value can be different). By default, 1.0, meaning the value cannot change within each
                                    trace.

    :return: a list with the IDs of the columns detected as case attributes, their type (discrete/continuous) and the values they take.
    """
    # Get columns to analyze
    if avoid_columns is None:
        avoid_columns = [log_ids.case, log_ids.activity, log_ids.start_time, log_ids.end_time, log_ids.resource]
    columns = [column for column in event_log.columns if column not in avoid_columns]
    # Go over the columns of the event log to check if any attribute is not changing through the traces
    case_attributes = []
    for attribute in columns:
        # Get distribution of attribute values among cases
        attribute_distribution = event_log.groupby([log_ids.case, attribute]).size()
        # Get confidence of a single value in each case
        confidences = []
        for case_id, frequencies in attribute_distribution.groupby(level=0):
            confidences += [frequencies.max() / frequencies.sum()]
        # If the average of confidences is higher than the threshold, consider it a case_attribute
        if mean(confidences) >= confidence_threshold:
            if _is_discrete_variable(event_log[attribute]):  # Discrete variable
                # Compute number of times each value is the most frequent in the case
                values = {value: 0 for value in event_log[attribute].unique()}
                for case_id, frequencies in attribute_distribution.groupby(level=0):
                    main_attribute = frequencies.idxmax()[
                        1
                    ]  # Second element of the index (attribute value) with the highest frequency
                    values[main_attribute] += 1
                # Relativize frequencies
                num_cases = len(event_log[log_ids.case].unique())
                # Add case attribute specification
                case_attributes += [
                    {
                        "name": attribute,
                        "type": "discrete",
                        "values": [
                            {"key": value, "probability": values[value] / num_cases}
                            for value in values
                            if values[value] > 0
                        ],
                    }
                ]
            else:  # Continuous variable
                # Get data distribution
                data = []
                for case_id, frequencies in attribute_distribution.groupby(level=0):
                    data += [
                        frequencies.idxmax()[1]
                    ]  # Second element of the index (attribute value) with the highest frequency
                # Fit data best distribution
                best_distribution = get_best_fitting_distribution(data)
                # Add case attribute specification
                case_attributes += [
                    {"name": attribute, "type": "continuous", "values": best_distribution.to_prosimos_distribution()}
                ]
    # Return list with discovered case attributes
    return case_attributes


def _is_discrete_variable(values: pd.Series) -> bool:
    return not is_numeric_dtype(values)  # TODO more complex check? there could be numeric discrete attributes
