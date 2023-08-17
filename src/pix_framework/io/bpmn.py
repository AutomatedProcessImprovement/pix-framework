from pathlib import Path

from lxml import etree


def get_activities_ids_by_name_from_bpmn(model_path: Path) -> dict:
    """
    Returns activities' IDs accessed by activity name from the model.

    Sample output: { 'Register Order': '1', 'Verify Order': '2' }
    """
    xml = etree.parse(str(model_path))
    root = xml.getroot()
    namespace = {"xmlns": "http://www.omg.org/spec/BPMN/20100524/MODEL"}
    values = {}
    for process in root.findall("xmlns:process", namespace):
        for task in process.findall("xmlns:task", namespace):
            task_id = task.get("id")
            task_name = task.get("name")
            values[task_name] = task_id
    return values


def get_activities_names_from_bpmn(model_path: Path) -> list[str]:
    """
    Returns activities' names from the model.

    Sample output: ['Register Order', 'Verify Order']
    """
    xml = etree.parse(str(model_path))
    root = xml.getroot()
    namespace = {"xmlns": "http://www.omg.org/spec/BPMN/20100524/MODEL"}
    values = []
    for process in root.findall("xmlns:process", namespace):
        for task in process.findall("xmlns:task", namespace):
            task_name = task.get("name")
            values.append(task_name)
    return values
