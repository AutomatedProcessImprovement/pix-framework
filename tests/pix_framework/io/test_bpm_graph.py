from pathlib import Path
import pytest
from pix_framework.io.bpm_graph import BPMNGraph
from pix_framework.io.event_log import EventLogIDs, read_csv_log
from typing import Dict, List, Tuple

assets_dir = Path(__file__).parent.parent / "assets"


class CaseExpectedResults:
    def __init__(
        self,
        case_id: str,
        flow_frequencies: Dict[str, int],
        missed_tokens: [Dict[str, int]] = None,
        left_tokens: [Dict[str, int]] = None
    ):
        self.case_id = case_id
        self.flow_frequencies = flow_frequencies
        self.missed_tokens = missed_tokens or {}
        self.left_tokens = left_tokens or {}


class FlowArcAnalysisTester:
    """Class to handle flow arc analysis testing logic"""

    def __init__(self, model_path: Path, log_path: Path, model_name: str = None):
        self.model_path = model_path
        self.log_path = log_path
        self.model_name = model_name or model_path.stem
        self.bpmn_graph = None
        self.event_log = None
        self.case_results = {}
        self.log_ids = EventLogIDs(case="case_id", activity="activity", end_time="end_time")

    def load_model_and_log(self) -> None:
        self.bpmn_graph = BPMNGraph.from_bpmn_path(self.model_path)
        self.event_log = read_csv_log(self.log_path, self.log_ids)
        self.event_log = self.event_log.sort_values(self.log_ids.end_time)

        print("\n\n" + "="*80)
        print(f"ANALYZING MODEL: {self.model_name}")
        print("="*80)

    def analyze_cases(self) -> Dict[str, Dict]:
        self.case_results = {}
        case_count = len(self.event_log[self.log_ids.case].unique())
        print(f"\nFound {case_count} cases in the event log")

        for case_id, case_events in self.event_log.groupby(self.log_ids.case):
            print("\n" + "-"*80)
            print(f"CASE {case_id} ANALYSIS")
            print("-"*80)

            flow_arcs_frequency = {}

            trace = case_events[self.log_ids.activity].tolist()

            is_correct, fired_tasks, left_tokens, frequency_count, missed_tokens = self.bpmn_graph.replay_trace(
                trace, flow_arcs_frequency
            )

            trace_str = " → ".join(trace)
            print(f"Trace: {trace_str}")
            print(f"Conformance: {'✓ Correct' if is_correct else '✗ Incorrect'}")
            print(f"Fired Tasks: {[i for i, v in enumerate(fired_tasks) if v]} Length: {len(fired_tasks)}")
            if left_tokens:
                print(f"Left Tokens: {left_tokens}")
            else:
                print("Left Tokens: None")

            # # left tokens - tokens that "were not consumed"
            # for flow_id in pending:
            #     left_tokens[flow_id] = left_tokens.get(flow_id, 0) + 1

            # # missed tokens - missing to be consumed tokens
            # for i, fired in enumerate(fired_tasks):
            #     if not fired and i < len(trace):
            #         activity = trace[i]
            #         if activity in self.bpmn_graph.from_name:
            #             task_id = self.bpmn_graph.from_name[activity]
            #             for flow_id in self.bpmn_graph.element_info[task_id].incoming_flows:
            #                 missed_tokens[flow_id] = missed_tokens.get(flow_id, 0) + 1

            self.case_results[str(case_id)] = {
                'flow_frequencies': frequency_count,
                'missed_tokens': missed_tokens,
                'left_tokens': left_tokens
            }

            self.print_flow_arc_analysis(frequency_count, missed_tokens, left_tokens)
        return self.case_results

    def print_flow_arc_analysis(
            self,
            flow_arcs_frequency: Dict[str, int],
            missed_tokens: Dict[str, int],
            left_tokens: Dict[str, int]
    ) -> None:
        print("\nFLOW ARC ANALYSIS FOR THIS CASE:")
        print("-" * 80)

        print(f"{'FLOW ID':<10} {'SOURCE → TARGET':<30} {'FREQUENCY':<10} {'MISSED':<10} {'LEFT':<10}")
        print("-" * 80)

        def numeric_sort_key(flow_id):
            if flow_id.startswith("f") and flow_id[1:].isdigit():
                return int(flow_id[1:])
            return flow_id

        sorted_flow_ids = sorted(self.bpmn_graph.flow_arcs.keys(), key=numeric_sort_key)

        for flow_id in sorted_flow_ids:
            source = self.bpmn_graph.element_info[self.bpmn_graph.flow_arcs[flow_id][0]].name
            target = self.bpmn_graph.element_info[self.bpmn_graph.flow_arcs[flow_id][1]].name

            flow_name = f"{source} → {target}"
            frequency = flow_arcs_frequency.get(flow_id, 0)
            missed = missed_tokens.get(flow_id, 0)
            left = left_tokens.get(flow_id, 0)

            print(f"{flow_id:<10} {flow_name:<30} {frequency:<10} {missed:<10} {left:<10}")

    def assert_case_results(self, expected_results_by_case: List[CaseExpectedResults]) -> None:
        all_flow_ids = set(self.bpmn_graph.flow_arcs.keys())

        print("\n" + "="*80)
        print(f"ASSERTION RESULTS FOR MODEL: {self.model_name}")
        print("="*80)

        for expected in expected_results_by_case:
            case_id = expected.case_id

            print(f"\nChecking case {case_id}:")

            if case_id not in self.case_results:
                print(f"  ✗ ERROR: Case ID {case_id} not found in event log")
                assert case_id in self.case_results, f"Case ID {case_id} not found in event log"
                continue

            result = self.case_results[case_id]
            all_passed = True

            for flow_id in all_flow_ids:
                expected_freq = expected.flow_frequencies.get(flow_id, 0)
                actual_freq = result['flow_frequencies'].get(flow_id, 0)
                if actual_freq != expected_freq:
                    all_passed = False
                    print(f"  ✗ Flow {flow_id} frequency: expected {expected_freq}, got {actual_freq}")
                    assert actual_freq == expected_freq, (
                        f"Case {case_id}, Flow {flow_id} frequency mismatch: "
                        f"expected {expected_freq}, got {actual_freq}"
                    )

            for flow_id in all_flow_ids:
                expected_missed = expected.missed_tokens.get(flow_id, 0)
                actual_missed = result['missed_tokens'].get(flow_id, 0)
                if actual_missed != expected_missed:
                    all_passed = False
                    print(f"  ✗ Flow {flow_id} missed tokens: expected {expected_missed}, got {actual_missed}")
                    assert actual_missed == expected_missed, (
                        f"Case {case_id}, Flow {flow_id} missed tokens mismatch: "
                        f"expected {expected_missed}, got {actual_missed}"
                    )

            for flow_id in all_flow_ids:
                expected_left = expected.left_tokens.get(flow_id, 0)
                actual_left = result['left_tokens'].get(flow_id, 0)
                if actual_left != expected_left:
                    all_passed = False
                    print(f"  ✗ Flow {flow_id} left tokens: expected {expected_left}, got {actual_left}")
                    assert actual_left == expected_left, (
                        f"Case {case_id}, Flow {flow_id} left tokens mismatch: "
                        f"expected {expected_left}, got {actual_left}"
                    )

            if all_passed:
                print(f"  ✓ All assertions passed")


def run_flow_arc_analysis(
        model_path: Path,
        log_path: Path,
        expected_results_by_case: List[CaseExpectedResults],
        model_name: str = None
) -> Tuple[bool, Dict[str, Dict]]:
    tester = FlowArcAnalysisTester(model_path, log_path, model_name)
    tester.load_model_and_log()
    case_results = tester.analyze_cases()
    tester.assert_case_results(expected_results_by_case)
    return True, case_results


TEST_CASES = [
    {
        "model_path": assets_dir / "simple_model.bpmn",
        "log_path": assets_dir / "simple_model_event_log.csv",
        "model_name": "Simple Sequential Process",
        "expected_results": [
            # Case 1: A->B->C
            CaseExpectedResults(
                case_id="1",
                flow_frequencies={
                    "f1": 1,
                    "f2": 1,
                    "f3": 1,
                    "f4": 1,
                },
                missed_tokens={
                    "f1": 0,
                    "f2": 0,
                    "f3": 0,
                    "f4": 0,
                },
                left_tokens={
                    "f1": 0,
                    "f2": 0,
                    "f3": 0,
                    "f4": 0,
                }
            ),
            # Case 2: Only B
            CaseExpectedResults(
                case_id="2",
                flow_frequencies={
                    "f1": 1,
                    "f2": 1,
                    "f3": 1,
                    "f4": 0,
                },
                missed_tokens={
                    "f1": 0,
                    "f2": 1,
                    "f3": 0,
                    "f4": 0,
                },
                left_tokens={
                    "f1": 1,
                    "f2": 0,
                    "f3": 1,
                    "f4": 0,
                }
            ),
        ]
    },
    {
        "model_path": assets_dir / "exclusive_gateway_simple.bpmn",
        "log_path": assets_dir / "exclusive_gateway_event_log.csv",
        "model_name": "Exclusive Gateway Process",
        "expected_results": [
            # Case 1: Only C
            CaseExpectedResults(
                case_id="1",
                flow_frequencies={
                    "f1": 1,
                    "f2": 0,
                    "f3": 0,
                    "f4": 1,
                    "f5": 0,
                    "f6": 1,
                    "f7": 1,
                },
                missed_tokens={
                    "f1": 0,
                    "f2": 0,
                    "f3": 0,
                    "f4": 1,
                    "f5": 0,
                    "f6": 0,
                    "f7": 0,
                },
                left_tokens={
                    "f1": 1,
                    "f2": 0,
                    "f3": 0,
                    "f4": 0,
                    "f5": 0,
                    "f6": 0,
                    "f7": 0,
                }
            ),
        ]
    },
    {
        "model_path": assets_dir / "and_xor_model.bpmn",
        "log_path": assets_dir / "and_xor_event_log.csv",
        "model_name": "AND & XOR Gateways Simple Process",
        "expected_results": [
            # Case 1: A->D
            CaseExpectedResults(
                case_id="1",
                flow_frequencies={
                    "f1": 1,
                    "f2": 1,
                    "f3": 1,
                    "f4": 1,
                    "f5": 0,
                    "f6": 0,
                    "f7": 0,
                    "f8": 0,
                    "f9": 1,
                    "f10": 0,
                    "f11": 0,
                },
                missed_tokens={
                    "f1": 0,
                    "f2": 0,
                    "f3": 0,
                    "f4": 0,
                    "f5": 0,
                    "f6": 0,
                    "f7": 0,
                    "f8": 0,
                    "f9": 0,
                    "f10": 0,
                    "f11": 0,
                },
                left_tokens={
                    "f1": 0,
                    "f2": 0,
                    "f3": 1,
                    "f4": 0,
                    "f5": 0,
                    "f6": 0,
                    "f7": 0,
                    "f8": 0,
                    "f9": 1,
                    "f10": 0,
                    "f11": 0,
                }
            ),
        ]
    },
    {
        "model_path": assets_dir / "nested_and_gateways.bpmn",
        "log_path": assets_dir / "nested_and_gateways_event_log.csv",
        "model_name": "Nested AND Gateways Process",
        "expected_results": [
            # Case 1: A->B->C->D->E
            CaseExpectedResults(
                case_id="1",
                flow_frequencies={
                    "f1": 1,
                    "f2": 1,
                    "f3": 1,
                    "f4": 1,
                    "f5": 1,
                    "f6": 1,
                    "f7": 1,
                    "f8": 1,
                    "f9": 1,
                    "f10": 1,
                    "f11": 1,
                    "f12": 1,
                    "f13": 1,
                    "f14": 1,
                    "f15": 1,
                },
                missed_tokens={
                    "f1": 0,
                    "f2": 0,
                    "f3": 0,
                    "f4": 0,
                    "f5": 0,
                    "f6": 0,
                    "f7": 0,
                    "f8": 0,
                    "f9": 0,
                    "f10": 0,
                    "f11": 0,
                    "f12": 0,
                    "f13": 0,
                    "f14": 0,
                    "f15": 0,
                },
                left_tokens={
                    "f1": 0,
                    "f2": 0,
                    "f3": 0,
                    "f4": 0,
                    "f5": 0,
                    "f6": 0,
                    "f7": 0,
                    "f8": 0,
                    "f9": 0,
                    "f10": 0,
                    "f11": 0,
                    "f12": 0,
                    "f13": 0,
                    "f14": 0,
                    "f15": 0,
                }
            ),
        ]
    },
    {
        "model_path": assets_dir / "nested_or_gateways.bpmn",
        "log_path": assets_dir / "nested_or_gateways_event_log.csv",
        "model_name": "Nested OR Gateways Process",
        "expected_results": [
            # Case 1: A->B->C->F
            CaseExpectedResults(
                case_id="1",
                flow_frequencies={
                    "f1": 1,
                    "f2": 1,
                    "f3": 1,
                    "f4": 1,
                    "f5": 1,
                    "f6": 1,
                    "f7": 1,
                    "f8": 1,
                    "f9": 0,
                    "f10": 0,
                    "f11": 0,
                    "f12": 0,
                    "f13": 0,
                    "f14": 1,
                    "f15": 1,
                    "f16": 1,
                },
                missed_tokens={
                    "f1": 0,
                    "f2": 0,
                    "f3": 0,
                    "f4": 0,
                    "f5": 0,
                    "f6": 0,
                    "f7": 0,
                    "f8": 0,
                    "f9": 0,
                    "f10": 0,
                    "f11": 0,
                    "f12": 0,
                    "f13": 0,
                    "f14": 0,
                    "f15": 0,
                    "f16": 0,
                },
                left_tokens={
                    "f1": 0,
                    "f2": 0,
                    "f3": 0,
                    "f4": 0,
                    "f5": 0,
                    "f6": 0,
                    "f7": 0,
                    "f8": 0,
                    "f9": 0,
                    "f10": 0,
                    "f11": 0,
                    "f12": 0,
                    "f13": 0,
                    "f14": 0,
                    "f15": 0,
                    "f16": 0,
                }
            ),
        ]
    },
    {
        "model_path": assets_dir / "xor_or_gateways.bpmn",
        "log_path": assets_dir / "xor_or_gateways_event_log.csv",
        "model_name": "XOR OR gateways Process",
        "expected_results": [
            # Case 1: A->B
            CaseExpectedResults(
                case_id="1",
                flow_frequencies={
                    "f1": 1,
                    "f2": 1,
                    "f3": 0,
                    "f4": 1,
                    "f5": 1,
                    "f6": 1,
                    "f7": 1,
                    "f8": 0,
                    "f9": 1,
                    "f10": 0,
                    "f11": 1,
                    "f12": 1,
                    "f13": 1,
                },
                missed_tokens={
                    "f1": 0,
                    "f2": 0,
                    "f3": 0,
                    "f4": 0,
                    "f5": 0,
                    "f6": 0,
                    "f7": 0,
                    "f8": 0,
                    "f9": 0,
                    "f10": 0,
                    "f11": 0,
                    "f12": 0,
                    "f13": 0,
                },
                left_tokens={
                    "f1": 0,
                    "f2": 0,
                    "f3": 0,
                    "f4": 0,
                    "f5": 0,
                    "f6": 0,
                    "f7": 0,
                    "f8": 0,
                    "f9": 0,
                    "f10": 0,
                    "f11": 0,
                    "f12": 0,
                    "f13": 0,
                }
            ),
        ]
    },
    {
        "model_path": assets_dir / "deep_or_gateways.bpmn",
        "log_path": assets_dir / "deep_or_gateways_event_log.csv",
        "model_name": "Deep Nested OR Gateways Process",
        "expected_results": [
            # Case 1: A->B->C->D->E->F
            CaseExpectedResults(
                case_id="1",
                flow_frequencies={
                    "f1": 1,
                    "f2": 1,
                    "f3": 1,
                    "f4": 1,
                    "f5": 1,
                    "f6": 1,
                    "f7": 1,
                    "f8": 1,
                    "f9": 1,
                    "f10": 1,
                    "f11": 1,
                    "f12": 1,
                    "f13": 1,
                    "f14": 1,
                    "f15": 1,
                    "f16": 1,
                    "f17": 1,
                    "f18": 1,
                    "f19": 1,
                    "f20": 1,
                    "f21": 1,
                    "f22": 1,
                },
                missed_tokens={
                    "f1": 0,
                    "f2": 0,
                    "f3": 0,
                    "f4": 0,
                    "f5": 0,
                    "f6": 0,
                    "f7": 0,
                    "f8": 0,
                    "f9": 0,
                    "f10": 0,
                    "f11": 0,
                    "f12": 0,
                    "f13": 0,
                    "f14": 0,
                    "f15": 0,
                    "f16": 0,
                    "f17": 0,
                    "f18": 0,
                    "f19": 0,
                    "f20": 0,
                    "f21": 0,
                    "f22": 0,
                },
                left_tokens={
                    "f1": 0,
                    "f2": 0,
                    "f3": 0,
                    "f4": 0,
                    "f5": 0,
                    "f6": 0,
                    "f7": 0,
                    "f8": 0,
                    "f9": 0,
                    "f10": 0,
                    "f11": 0,
                    "f12": 0,
                    "f13": 0,
                    "f14": 0,
                    "f15": 0,
                    "f16": 0,
                    "f17": 0,
                    "f18": 0,
                    "f19": 0,
                    "f20": 0,
                    "f21": 0,
                    "f22": 0,
                }
            ),
            CaseExpectedResults(
                case_id="2",
                flow_frequencies={
                    "f1": 1,
                    "f2": 1,
                    "f3": 1,
                    "f4": 1,
                    "f5": 0,
                    "f6": 0,
                    "f7": 0,
                    "f8": 0,
                    "f9": 0,
                    "f10": 0,
                    "f11": 0,
                    "f12": 0,
                    "f13": 0,
                    "f14": 0,
                    "f15": 0,
                    "f16": 0,
                    "f17": 0,
                    "f18": 0,
                    "f19": 0,
                    "f20": 0,
                    "f21": 1,
                    "f22": 1,
                },
                missed_tokens={
                    "f1": 0,
                    "f2": 0,
                    "f3": 0,
                    "f4": 0,
                    "f5": 0,
                    "f6": 0,
                    "f7": 0,
                    "f8": 0,
                    "f9": 0,
                    "f10": 0,
                    "f11": 0,
                    "f12": 0,
                    "f13": 0,
                    "f14": 0,
                    "f15": 0,
                    "f16": 0,
                    "f17": 0,
                    "f18": 0,
                    "f19": 0,
                    "f20": 0,
                    "f21": 0,
                    "f22": 0,
                },
                left_tokens={
                    "f1": 0,
                    "f2": 0,
                    "f3": 0,
                    "f4": 0,
                    "f5": 0,
                    "f6": 0,
                    "f7": 0,
                    "f8": 0,
                    "f9": 0,
                    "f10": 0,
                    "f11": 0,
                    "f12": 0,
                    "f13": 0,
                    "f14": 0,
                    "f15": 0,
                    "f16": 0,
                    "f17": 0,
                    "f18": 0,
                    "f19": 0,
                    "f20": 0,
                    "f21": 0,
                    "f22": 0,
                }
            ),
            CaseExpectedResults(
                case_id="3",
                flow_frequencies={
                    "f1": 1,
                    "f2": 1,
                    "f3": 1,
                    "f4": 1,
                    "f5": 1,
                    "f6": 1,
                    "f7": 1,
                    "f8": 0,
                    "f9": 1,
                    "f10": 1,
                    "f11": 0,
                    "f12": 0,
                    "f13": 0,
                    "f14": 1,
                    "f15": 0,
                    "f16": 1,
                    "f17": 0,
                    "f18": 1,
                    "f19": 1,
                    "f20": 1,
                    "f21": 1,
                    "f22": 1,
                },
                missed_tokens={
                    "f1": 0,
                    "f2": 0,
                    "f3": 0,
                    "f4": 0,
                    "f5": 0,
                    "f6": 0,
                    "f7": 0,
                    "f8": 0,
                    "f9": 0,
                    "f10": 0,
                    "f11": 0,
                    "f12": 0,
                    "f13": 0,
                    "f14": 0,
                    "f15": 0,
                    "f16": 0,
                    "f17": 0,
                    "f18": 0,
                    "f19": 0,
                    "f20": 0,
                    "f21": 0,
                    "f22": 0,
                },
                left_tokens={
                    "f1": 0,
                    "f2": 0,
                    "f3": 0,
                    "f4": 0,
                    "f5": 0,
                    "f6": 0,
                    "f7": 0,
                    "f8": 0,
                    "f9": 0,
                    "f10": 0,
                    "f11": 0,
                    "f12": 0,
                    "f13": 0,
                    "f14": 0,
                    "f15": 0,
                    "f16": 0,
                    "f17": 0,
                    "f18": 0,
                    "f19": 0,
                    "f20": 0,
                    "f21": 0,
                    "f22": 0,
                }
            ),
        ]
    },
    {
        "model_path": assets_dir / "deep_or_gateways_2.bpmn",
        "log_path": assets_dir / "deep_or_gateways_2_event_log.csv",
        "model_name": "Deep OR gateways 2 Process",
        "expected_results": [
            # Case 1: A->B->D
            CaseExpectedResults(
                case_id="1",
                flow_frequencies={
                    "f1": 1,
                    "f2": 1,
                    "f3": 1,
                    "f4": 1,
                    "f5": 1,
                    "f6": 1,
                    "f7": 1,
                    "f8": 1,
                    "f9": 1,
                    "f10": 0,
                    "f11": 0,
                    "f12": 0,
                    "f13": 0,
                    "f14": 0,
                    "f15": 0,
                    "f16": 0,
                    "f17": 1,
                    "f18": 1,
                    "f19": 1,
                    "f20": 1,
                },
                missed_tokens={
                    "f1": 0,
                    "f2": 0,
                    "f3": 0,
                    "f4": 0,
                    "f5": 0,
                    "f6": 0,
                    "f7": 0,
                    "f8": 0,
                    "f9": 0,
                    "f10": 0,
                    "f11": 0,
                    "f12": 0,
                    "f13": 0,
                    "f14": 0,
                    "f15": 0,
                    "f16": 0,
                    "f17": 0,
                    "f18": 0,
                    "f19": 0,
                    "f20": 0,
                },
                left_tokens={
                    "f1": 0,
                    "f2": 0,
                    "f3": 0,
                    "f4": 0,
                    "f5": 0,
                    "f6": 0,
                    "f7": 0,
                    "f8": 0,
                    "f9": 0,
                    "f10": 0,
                    "f11": 0,
                    "f12": 0,
                    "f13": 0,
                    "f14": 0,
                    "f15": 0,
                    "f16": 0,
                    "f17": 0,
                    "f18": 0,
                    "f19": 0,
                    "f20": 0,
                }
            ),
            # CASE: A->C
            CaseExpectedResults(
                case_id="2",
                flow_frequencies={
                    "f1": 1,
                    "f2": 1,
                    "f3": 1,
                    "f4": 1,
                    "f5": 1,
                    "f6": 0,
                    "f7": 1,
                    "f8": 1,
                    "f9": 0,
                    "f10": 1,
                    "f11": 1,
                    "f12": 1,
                    "f13": 1,
                    "f14": 1,
                    "f15": 0,
                    "f16": 0,
                    "f17": 0,
                    "f18": 0,
                    "f19": 0,
                    "f20": 0,
                },
                missed_tokens={
                    "f1": 0,
                    "f2": 0,
                    "f3": 0,
                    "f4": 0,
                    "f5": 0,
                    "f6": 0,
                    "f7": 0,
                    "f8": 0,
                    "f9": 0,
                    "f10": 0,
                    "f11": 0,
                    "f12": 0,
                    "f13": 0,
                    "f14": 0,
                    "f15": 0,
                    "f16": 0,
                    "f17": 0,
                    "f18": 0,
                    "f19": 0,
                    "f20": 0,
                },
                left_tokens={
                    "f1": 0,
                    "f2": 0,
                    "f3": 0,
                    "f4": 0,
                    "f5": 0,
                    "f6": 0,
                    "f7": 0,
                    "f8": 0,
                    "f9": 0,
                    "f10": 0,
                    "f11": 0,
                    "f12": 0,
                    "f13": 0,
                    "f14": 1,
                    "f15": 0,
                    "f16": 0,
                    "f17": 0,
                    "f18": 0,
                    "f19": 0,
                    "f20": 0,
                }
            ),
        ]
    },
    {
        "model_path": assets_dir / "model_1.bpmn",
        "log_path": assets_dir / "model_1_event_log.csv",
        "model_name": "Model with multiple types of split gateways",
        "expected_results": [
            # Case 1: A->B->C->D->E
            CaseExpectedResults(
                case_id="1",
                flow_frequencies={
                    "f1": 1,
                    "f2": 1,
                    "f3": 1,
                    "f4": 1,
                    "f5": 1,
                    "f6": 1,
                    "f7": 0,
                    "f8": 1,
                    "f9": 1,
                    "f10": 1,
                    "f11": 0,
                    "f12": 0,
                    "f13": 0,
                    "f14": 1,
                    "f15": 1,
                    "f16": 1,
                    "f17": 1,
                    "f18": 1,
                    "f19": 1,
                    "f20": 1,
                    "f21": 1,
                },
                missed_tokens={
                    "f1": 0,
                    "f2": 0,
                    "f3": 0,
                    "f4": 0,
                    "f5": 0,
                    "f6": 0,
                    "f7": 0,
                    "f8": 0,
                    "f9": 0,
                    "f10": 0,
                    "f11": 0,
                    "f12": 0,
                    "f13": 0,
                    "f14": 1,
                    "f15": 0,
                    "f16": 0,
                    "f17": 0,
                    "f18": 0,
                    "f19": 0,
                    "f20": 0,
                    "f21": 0,
                },
                left_tokens={
                    "f1": 0,
                    "f2": 0,
                    "f3": 0,
                    "f4": 0,
                    "f5": 0,
                    "f6": 0,
                    "f7": 0,
                    "f8": 0,
                    "f9": 0,
                    "f10": 0,
                    "f11": 0,
                    "f12": 0,
                    "f13": 0,
                    "f14": 0,
                    "f15": 0,
                    "f16": 0,
                    "f17": 0,
                    "f18": 0,
                    "f19": 0,
                    "f20": 0,
                    "f21": 0,
                }
            ),
            # CASE: A->C->D->E
            CaseExpectedResults(
                case_id="2",
                flow_frequencies={
                    "f1": 1,
                    "f2": 1,
                    "f3": 1,
                    "f4": 1,
                    "f5": 1,
                    "f6": 1,
                    "f7": 0,
                    "f8": 1,
                    "f9": 0,
                    "f10": 1,
                    "f11": 0,
                    "f12": 0,
                    "f13": 0,
                    "f14": 0,
                    "f15": 0,
                    "f16": 0,
                    "f17": 0,
                    "f18": 1,
                    "f19": 1,
                    "f20": 1,
                    "f21": 1,
                },
                missed_tokens={
                    "f1": 0,
                    "f2": 0,
                    "f3": 0,
                    "f4": 0,
                    "f5": 0,
                    "f6": 0,
                    "f7": 0,
                    "f8": 0,
                    "f9": 0,
                    "f10": 0,
                    "f11": 0,
                    "f12": 0,
                    "f13": 0,
                    "f14": 0,
                    "f15": 0,
                    "f16": 0,
                    "f17": 0,
                    "f18": 0,
                    "f19": 0,
                    "f20": 0,
                },
                left_tokens={
                    "f1": 0,
                    "f2": 0,
                    "f3": 0,
                    "f4": 0,
                    "f5": 0,
                    "f6": 1,
                    "f7": 0,
                    "f8": 0,
                    "f9": 0,
                    "f10": 0,
                    "f11": 0,
                    "f12": 0,
                    "f13": 0,
                    "f14": 0,
                    "f15": 0,
                    "f16": 0,
                    "f17": 0,
                    "f18": 0,
                    "f19": 0,
                    "f20": 0,
                }
            ),
        ]
    },
    {
        "model_path": assets_dir / "model_2.bpmn",
        "log_path": assets_dir / "model_2_event_log.csv",
        "model_name": "OR gateways with more than 2 outgoing flow arcs",
        "expected_results": [
            # Case 1: A->B->C->D->E
            CaseExpectedResults(
                case_id="1",
                flow_frequencies={
                    "f1": 1,
                    "f2": 1,
                    "f3": 1,
                    "f4": 1,
                    "f5": 1,
                    "f6": 1,
                    "f7": 1,
                    "f8": 1,
                    "f9": 1,
                    "f10": 1,
                    "f11": 1,
                    "f12": 1,
                    "f13": 1,
                    "f14": 1,
                    "f15": 1,
                    "f16": 1,
                },
                missed_tokens={
                    "f1": 0,
                    "f2": 0,
                    "f3": 0,
                    "f4": 0,
                    "f5": 0,
                    "f6": 0,
                    "f7": 0,
                    "f8": 0,
                    "f9": 0,
                    "f10": 0,
                    "f11": 0,
                    "f12": 0,
                    "f13": 0,
                    "f14": 0,
                    "f15": 0,
                    "f16": 0,
                },
                left_tokens={
                    "f1": 0,
                    "f2": 0,
                    "f3": 0,
                    "f4": 0,
                    "f5": 0,
                    "f6": 0,
                    "f7": 0,
                    "f8": 0,
                    "f9": 0,
                    "f10": 0,
                    "f11": 0,
                    "f12": 0,
                    "f13": 0,
                    "f14": 0,
                    "f15": 0,
                    "f16": 0,

                }
            ),
            # CASE: A->E
            CaseExpectedResults(
                case_id="2",
                flow_frequencies={
                    "f1": 1,
                    "f2": 1,
                    "f3": 1,
                    "f4": 1,
                    "f5": 0,
                    "f6": 0,
                    "f7": 0,
                    "f8": 0,
                    "f9": 0,
                    "f10": 0,
                    "f11": 0,
                    "f12": 0,
                    "f13": 0,
                    "f14": 0,
                    "f15": 1,
                    "f16": 1,
                },
                missed_tokens={
                    "f1": 0,
                    "f2": 0,
                    "f3": 0,
                    "f4": 0,
                    "f5": 0,
                    "f6": 0,
                    "f7": 0,
                    "f8": 0,
                    "f9": 0,
                    "f10": 0,
                    "f11": 0,
                    "f12": 0,
                    "f13": 0,
                    "f14": 0,
                    "f15": 0,
                    "f16": 0,
                },
                left_tokens={
                    "f1": 0,
                    "f2": 0,
                    "f3": 0,
                    "f4": 0,
                    "f5": 0,
                    "f6": 0,
                    "f7": 0,
                    "f8": 0,
                    "f9": 0,
                    "f10": 0,
                    "f11": 0,
                    "f12": 0,
                    "f13": 0,
                    "f14": 0,
                    "f15": 0,
                    "f16": 0,
                }
            ),
        ]
    },
]


@pytest.mark.smoke
@pytest.mark.parametrize("test_case", TEST_CASES)
def test_flow_arc_analysis(test_case):
    success, _ = run_flow_arc_analysis(
        test_case["model_path"],
        test_case["log_path"],
        test_case["expected_results"],
        test_case.get("model_name")
    )
    assert success



# ===========   PREVIOUS TEST SUITES  ===============
# @pytest.mark.smoke
# @pytest.mark.parametrize("model_path,log_path", [
#     (assets_dir / "furated_related/simple_model.bpmn", assets_dir / "furated_related/simple_model_event_log.csv")
# ])
# def test_flow_arc_analysis_single(model_path: Path, log_path: Path):

#     selected_case_id = 1

#     bpmn_graph = BPMNGraph.from_bpmn_path(model_path)
#     log_ids = EventLogIDs(case="case_id", activity="activity", end_time="end_time")
#     event_log = read_csv_log(log_path, log_ids)
#     event_log = event_log.sort_values(log_ids.end_time)

#     # Filter to only include the selected case ID
#     if selected_case_id is not None:
#         event_log = event_log[event_log[log_ids.case] == selected_case_id]

#     flow_arcs_frequency = {}
#     # left_tokens = {}

#     for case_id, case_events in event_log.groupby(log_ids.case):
#         task_sequence = case_events[log_ids.activity].tolist()
#         is_correct, fired_tasks, left_tokens, frequency_count, missed_tokens = bpmn_graph.replay_trace(task_sequence, flow_arcs_frequency)

#         # print(f"\nIsCorrect: {is_correct}, FiredTasks: {fired_tasks}, Pending: {pending}")
#         print(f"BPMN flow arcs: {bpmn_graph.flow_arcs}")

#         #left tokens - "were not consumed"
#         # print()
#         # for flow_id in pending:
#         #     left_tokens[flow_id] = left_tokens.get(flow_id, 0) + 1


#     def numeric_sort_key(flow_id):
#         if flow_id.startswith("f") and flow_id[1:].isdigit():
#             return int(flow_id[1:])
#         return flow_id

#     sorted_flow_ids = sorted(bpmn_graph.flow_arcs.keys(), key=numeric_sort_key)

#     print(f"\nFlow Arc Analysis: ")
#     for flow_id in sorted_flow_ids:
#         source = bpmn_graph.element_info[bpmn_graph.flow_arcs[flow_id][0]].name
#         target = bpmn_graph.element_info[bpmn_graph.flow_arcs[flow_id][1]].name
#         print(f"\nFlow {flow_id} ({source} -> {target}):")
#         print(f"  Frequency: {frequency_count.get(flow_id, 0)}")
#         print(f"  Missed tokens: {missed_tokens.get(flow_id, 0)}")
#         print(f"  Left tokens: {left_tokens.get(flow_id, 0)}")

#     # Assertions
#     assert bpmn_graph is not None


#     for flow_id in bpmn_graph.flow_arcs:
#         # assert flow_id in flow_arcs_frequency, f"Flow arc {flow_id} not found in frequency dictionary"
#         # assert isinstance(flow_arcs_frequency[flow_id], int), f"Invalid frequency type for flow {flow_id}"

#         if flow_id in missed_tokens:
#             assert isinstance(missed_tokens[flow_id], int), f"Invalid missed tokens type for flow {flow_id}"

#         if flow_id in left_tokens:
#             assert isinstance(left_tokens[flow_id], int), f"Invalid left tokens type for flow {flow_id}"



@pytest.mark.smoke
@pytest.mark.parametrize("model_path", [(assets_dir / "PurchasingExample.bpmn")])
def test_from_bpmn_path(model_path: Path):
    graph = BPMNGraph.from_bpmn_path(model_path)
    assert graph is not None
    assert graph.starting_event is not None
    assert len(graph.flow_arcs) > 0