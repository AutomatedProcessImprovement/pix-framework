import datetime
import json

weekday_str = ["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY"]


def multitasking_to_prosimos(res_multitasks: dict, workloads: dict, is_local: bool):
    multitask_probability = []
    if is_local:
        for res in res_multitasks:
            multitask_probability.append({
                "resource_id": res,
                "r_workload": workloads[res],
                "weekly_probability": _format_weekly_granules(res_multitasks[res])
            })
        return {
            "type": "local",
            "values": multitask_probability
        }

    else:
        for res in res_multitasks:
            multitask_probability.append({
                "resource_id": res,
                "r_workload": workloads[res],
                "multitask_info": _multitasking_probabilities(res_multitasks[res])
            })
        return {
            "type": "global",
            "values": multitask_probability
        }


def _format_weekly_granules(weekly_multi: dict):
    weekly_granularity = []
    for wd in weekly_multi:
        weekly_granularity.append(_format_weekday(wd, weekly_multi[wd]))
    return weekly_granularity


def _format_weekday(wd: int, granules: dict):
    result = []
    duration = 1440 // len(granules)
    current_time = datetime.time(0, 0, 0)
    for gr in granules:
        hour, minute = divmod(current_time.hour * 60 + current_time.minute + duration, 60)
        if hour == 24:
            hour = 0
        next_time = datetime.time(hour, minute)
        if len(gr) > 1:
            result.append({
                "from": weekday_str[wd],
                "to":  weekday_str[wd],
                "beginTime": str(current_time),
                "endTime": str(next_time),
                "multitask_info": _multitasking_probabilities(gr)
            })
        current_time = next_time
    return result


def _multitasking_probabilities(task_prob):
    probabilities = []
    for i in range(1, len(task_prob)):
        probabilities.append({
            "parallel_tasks": i,
            "probability": task_prob[i]
        })
    return probabilities


def extend_prosimos_json(in_json_file_path, out_json_file_path, res_multitasks: dict, workloads: dict, is_local: bool):
    try:
        with open(in_json_file_path, 'r') as file:
            simulation_params = json.load(file)
    except FileNotFoundError:
        # If the file doesn't exist, start with an empty dictionary or list
        simulation_params = {}
    json_multitask = multitasking_to_prosimos(res_multitasks, workloads, is_local)
    simulation_params.update({"multitask": json_multitask})

    # Write the updated data back to the file
    with open(out_json_file_path, 'w') as file:
        json.dump(simulation_params, file, indent=4)







