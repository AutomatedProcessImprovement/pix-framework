import time

from pix_framework.discovery.attributes.metrics import get_metrics_by_type


def log_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} executed in {end_time - start_time:.2f} seconds")
        return result
    return wrapper


def subtract_lists(main_list, subtract_list):
    return [item for item in main_list if item not in subtract_list]


def print_results_table(model_results, metric_names):
    metric_labels = []
    for name in metric_names:
        metric_labels.extend([f"g_{name}", f"e_{name}", f"g_{name}_avg", f"e_{name}_avg", f"g_{name}_dev", f"e_{name}_dev"])

    header_parts = ["{:<25}", "{:<30}", "{:<6}"] + ["{:>20}" for _ in metric_labels]
    row_parts = ["{:<25}", "{:<30}", "{:<6}"] + ["{:>20.5e}" for _ in metric_labels]
    header_format = " ".join(header_parts)
    row_format = " ".join(row_parts)

    header = ["Attribute", "Model", "Type"] + metric_labels
    print(header_format.format(*header))

    for attr, data in model_results.items():
        models_data = data['models']
        for model_name, model_data in models_data.items():
            total_scores = model_data.get('total_scores', {'global': {}, 'event': {}})
            attr_type = data.get('type', 'Undefined')
            model_name = model_name.replace(' ', '_')
            row_values = [attr, model_name, attr_type]

            for metric_name in metric_names:
                g_score = total_scores['global'].get(metric_name, float('nan'))
                e_score = total_scores['event'].get(metric_name, float('nan'))

                g_mean = total_scores['global'].get(f'{metric_name}_avg', float('nan'))
                e_mean = total_scores['event'].get(f'{metric_name}_avg', float('nan'))

                g_deviation = total_scores['global'].get(f'{metric_name}_dev', float('nan'))
                e_deviation = total_scores['event'].get(f'{metric_name}_dev', float('nan'))

                row_values.extend([g_score, e_score, g_mean, e_mean, g_deviation, e_deviation])

            print(row_format.format(*row_values))


def print_case_results_table(model_results):
    discrete_metrics = get_metrics_by_type("discrete")
    continuous_metrics = get_metrics_by_type("continuous")

    discrete_attributes = {}
    continuous_attributes = {}

    # Categorize attributes based on presence of discrete/continuous metrics
    for attr, metrics in model_results.items():
        if any(metric in discrete_metrics for metric in metrics.keys()):
            discrete_attributes[attr] = metrics
        else:
            continuous_attributes[attr] = metrics

    # Define a function to print a table for given attributes
    def print_attributes_table(attributes, title):
        print(f"{title} Case Attributes\n{'=' * (len(title) + 11)}")
        if not attributes:
            print("No attributes to display.\n")
            return

        metric_names = list(next(iter(attributes.values())).keys())
        header_format = "{:<30} " + " ".join(["{:>15}" for _ in metric_names])
        row_format = "{:<30} " + " ".join(["{:>15.4f}" for _ in metric_names])

        print(header_format.format("Attribute", *metric_names))

        for attr, metrics in attributes.items():
            print(row_format.format(attr, *[metrics.get(metric, float('nan')) for metric in metric_names]))
        print("\n")

    print_attributes_table(discrete_attributes, "Discrete")
    print_attributes_table(continuous_attributes, "Continuous")