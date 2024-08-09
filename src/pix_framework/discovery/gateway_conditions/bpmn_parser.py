import xml.etree.ElementTree as ET
from enum import Enum
from replayer import BPMNGraph, BPMN, ElementInfo

bpmn_schema_url = "http://www.omg.org/spec/BPMN/20100524/MODEL"
simod_ns = {"qbp": "http://www.qbp-simulator.com/Schema201212"}
bpmn_element_ns = {"xmlns": bpmn_schema_url}


class EVENT_TYPE(Enum):
    MESSAGE = 'MESSAGE'
    TIMER = 'TIMER'
    LINK = 'LINK'
    SIGNAL = 'SIGNAL'
    TERMINATE = 'TERMINATE'
    UNDEFINED = 'UNDEFINED'


def parse_simulation_model(bpmn_path):
    tree = ET.parse(bpmn_path)
    root = tree.getroot()

    to_extract = {
        "xmlns:task": BPMN.TASK,
        "xmlns:startEvent": BPMN.START_EVENT,
        "xmlns:endEvent": BPMN.END_EVENT,
        "xmlns:exclusiveGateway": BPMN.EXCLUSIVE_GATEWAY,
        "xmlns:parallelGateway": BPMN.PARALLEL_GATEWAY,
        "xmlns:inclusiveGateway": BPMN.INCLUSIVE_GATEWAY,
        "xmlns:eventBasedGateway": BPMN.EVENT_BASED_GATEWAY,
        "xmlns:intermediateCatchEvent": BPMN.INTERMEDIATE_EVENT,
    }

    bpmn_graph = BPMNGraph()
    elements_map = dict()
    for process in root.findall("xmlns:process", bpmn_element_ns):
        for xmlns_key in to_extract:
            for bpmn_element in process.findall(xmlns_key, bpmn_element_ns):
                name = (
                    bpmn_element.attrib["name"]
                    if "name" in bpmn_element.attrib and len(bpmn_element.attrib["name"]) > 0
                    else bpmn_element.attrib["id"]
                )
                elem_general_type: BPMN = to_extract[xmlns_key]

                event_type = (
                    _get_event_type_from_element(name, bpmn_element) if BPMN.is_event(elem_general_type) else None
                )
                e_info = ElementInfo(elem_general_type, bpmn_element.attrib["id"], name, event_type)

                bpmn_graph.add_bpmn_element(bpmn_element.attrib["id"], e_info)
                elements_map[e_info.id] = {"in": 0, "out": 0, "info": e_info}

        # Counting incoming/outgoing flow arcs to handle cases of multiple in/out arcs simultaneously
        pending_flow_arcs = list()
        for flow_arc in process.findall("xmlns:sequenceFlow", bpmn_element_ns):
            # Fixing the case in which a task may have multiple incoming/outgoing flow-arcs
            pending_flow_arcs.append(flow_arc)
            if flow_arc.attrib["sourceRef"] in elements_map:
                elements_map[flow_arc.attrib["sourceRef"]]["out"] += 1
            if flow_arc.attrib["targetRef"] in elements_map:
                elements_map[flow_arc.attrib["targetRef"]]["in"] += 1
            # bpmn_graph.add_flow_arc(flow_arc.attrib["id"], flow_arc.attrib["sourceRef"], flow_arc.attrib["targetRef"])

        # Adding fake gateways for tasks with multiple incoming/outgoing flow arcs
        join_gateways = dict()
        split_gateways = dict()
        for t_id in elements_map:
            e_info = elements_map[t_id]["info"]
            if e_info.type is BPMN.TASK:
                if elements_map[t_id]["in"] > 1:
                    _add_fake_gateway(bpmn_graph, "xor_join_%s" % t_id, BPMN.EXCLUSIVE_GATEWAY, t_id, join_gateways)
                if elements_map[t_id]["out"] > 1:
                    _add_fake_gateway(bpmn_graph, "and_split_%s" % t_id, BPMN.PARALLEL_GATEWAY, t_id, split_gateways, False)
            elif e_info.type is BPMN.END_EVENT:
                if elements_map[t_id]["in"] > 1:
                    _add_fake_gateway(bpmn_graph, "or_join_%s" % t_id, BPMN.INCLUSIVE_GATEWAY, t_id, join_gateways)
            elif e_info.is_gateway():
                if elements_map[t_id]["in"] > 1 and elements_map[t_id]["out"] > 1:
                    _add_fake_gateway(bpmn_graph, "join_%s" % t_id, e_info.type, t_id, join_gateways)

        for flow_arc in pending_flow_arcs:
            source_id = flow_arc.attrib["sourceRef"]
            target_id = flow_arc.attrib["targetRef"]
            if source_id in split_gateways:
                source_id = split_gateways[source_id]
            if target_id in join_gateways:
                target_id = join_gateways[target_id]
            bpmn_graph.add_flow_arc(flow_arc.attrib["id"], source_id, target_id)

    bpmn_graph.encode_or_join_predecesors()
    bpmn_graph.validate_model()
    return bpmn_graph


def _add_fake_gateway(bpmn_graph, g_id, g_type, t_id, e_map, in_front=True):
    bpmn_graph.add_bpmn_element(g_id, ElementInfo(g_type, g_id, g_id, None))
    if in_front:
        bpmn_graph.add_flow_arc("%s_%s" % (g_id, t_id), g_id, t_id)
    else:
        bpmn_graph.add_flow_arc("%s_%s" % (t_id, g_id), t_id, g_id)
    e_map[t_id] = g_id


def _get_event_type_from_element(name: str, bpmn_element):
    children = list(bpmn_element)

    for child in children:
        if "EventDefinition" in child.tag:
            type_name = child.tag.split("}")[1]
            switcher = {
                "timerEventDefinition": EVENT_TYPE.TIMER,
                "messageEventDefinition": EVENT_TYPE.MESSAGE,
                "linkEventDefinition": EVENT_TYPE.LINK,
                "signalEventDefinition": EVENT_TYPE.SIGNAL,
                "terminateEventDefinition": EVENT_TYPE.TERMINATE,
            }

            event_type = switcher.get(type_name, EVENT_TYPE.UNDEFINED)
            if event_type == EVENT_TYPE.UNDEFINED:
                print(f"WARNING: {name} event has an undefined event type")

            return event_type
