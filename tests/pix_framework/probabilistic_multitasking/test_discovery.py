from pathlib import Path
import pandas as pd

from pix_framework.discovery.probabilistic_multitasking.discovery import calculate_multitasking, MultiType
from pix_framework.discovery.probabilistic_multitasking.model_serialization import extend_prosimos_json
from pix_framework.io.event_log import DEFAULT_CSV_IDS

assets_dir = Path(__file__).parent.parent / "assets/multitasking"


class BreakAllLoops(Exception):
    pass


def test_discover_global_multitasking():
    # event_log = pd.read_csv(assets_dir / "Application_to_Approval_Government_Agency.csv")
    probabilities = calculate_multitasking(pd.read_csv(assets_dir / "sequential.csv"))

    valid_p = True
    p_resources = probabilities[0]
    for resource in p_resources:
        assert p_resources[resource][1] == 1.0
        for p in p_resources[resource]:
            if p > 1.0:
                valid_p = False
                break
        if not valid_p:
            break

    assert valid_p

    # extend_prosimos_json(assets_dir / "sequential.json", assets_dir / "sequential.json", probabilities, False)


def test_discover_local_multitasking():
    probabilities = calculate_multitasking(pd.read_csv(assets_dir / "sequential.csv"),
                                           MultiType.LOCAL, 60)
    valid_p = True
    try:
        p_resources = probabilities[0]
        for resource in p_resources:
            for wd in p_resources[resource]:
                for gr_list in p_resources[resource][wd]:
                    for p in gr_list:
                        if p > 1.0:
                            raise BreakAllLoops
    except BreakAllLoops:
        assert False
    assert valid_p
    # extend_prosimos_json(assets_dir / "sequential.json", probabilities, True)






def _read_event_log(log_path: Path):
    event_log = pd.read_csv(log_path)
    if DEFAULT_CSV_IDS.enabled_time in event_log:
        event_log[DEFAULT_CSV_IDS.enabled_time] = pd.to_datetime(event_log[DEFAULT_CSV_IDS.enabled_time], utc=True)
    event_log[DEFAULT_CSV_IDS.start_time] = pd.to_datetime(event_log[DEFAULT_CSV_IDS.start_time], utc=True)
    event_log[DEFAULT_CSV_IDS.end_time] = pd.to_datetime(event_log[DEFAULT_CSV_IDS.end_time], utc=True)
    return event_log
