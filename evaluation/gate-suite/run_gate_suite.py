#!/usr/bin/env python3
"""Run deterministic Sigma-harness protocol gates.

This suite intentionally uses TraceBackend. It validates the harness state
contract before a real reader/writer is introduced, so failures are attributable
to orchestration rather than model quality.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sigma_mem_harness import MemoryWriteConfig, SigmaConversationHarness  # noqa: E402
from sigma_mem_harness.toy import TraceBackend  # noqa: E402


SCHEDULES = ("turn", "token_interval", "segment")


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            case = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        cases.append(case)
    return cases


def make_write_config(case: dict[str, Any], schedule: str) -> MemoryWriteConfig:
    params = case.get("schedule_params", {})
    if schedule == "turn":
        return MemoryWriteConfig(mode="turn")
    if schedule == "token_interval":
        return MemoryWriteConfig(
            mode="token_interval",
            token_interval_tokens=int(params.get("token_interval_tokens", 2)),
        )
    if schedule == "segment":
        return MemoryWriteConfig(
            mode="segment",
            segment_capacity_tokens=int(params.get("segment_capacity_tokens", 64)),
        )
    raise ValueError(f"unsupported schedule: {schedule}")


def run_case(case: dict[str, Any], schedule: str) -> dict[str, Any]:
    budgets = case["budgets"]
    backend = TraceBackend()
    harness = SigmaConversationHarness(
        backend,
        context_window_tokens=int(budgets["context_window_tokens"]),
        compact_target_tokens=int(budgets["compact_target_tokens"]),
        reserve_tokens=int(budgets.get("reserve_tokens", 0)),
        write_config=make_write_config(case, schedule),
    )
    results = []
    failure = case.get("failure") or {}
    raised: Exception | None = None

    try:
        for index, prompt in enumerate(case["turns"]):
            if index == int(failure.get("before_turn_index", -1)):
                backend.fail_next_prepare = True
            results.append(
                harness.run_turn(
                    prompt,
                    max_new_tokens=int(case["max_new_tokens"]),
                )
            )
        harness.finish()
    except Exception as exc:  # expected for the atomicity gate
        raised = exc

    generation_calls = [
        call for call in backend.calls if call.operation == "generate"
    ]
    readable_ids = [s.segment_id for s in harness.state.readable_segments]
    return {
        "case_id": case["case_id"],
        "schedule": schedule,
        "raised": raised is not None,
        "exception": None if raised is None else repr(raised),
        "results": [asdict(result) for result in results],
        "compact_count": harness.state.compact_count,
        "read_enabled": harness.state.read_enabled,
        "memory_version": harness.state.memory_version,
        "turn_count": len(harness.turns),
        "written_tokens": harness.sigma_written_tokens,
        "readable_tokens": harness.sigma_readable_tokens,
        "pending_segment_tokens": harness.pending_segment_tokens,
        "readable_segment_ids": readable_ids,
        "generation_read_flags": [call.read_enabled for call in generation_calls],
        "write_call_count": len(backend.write_calls),
        "write_calls": [list(call) for call in backend.write_calls],
        "generation_tokens": [list(result.token_ids) for result in results],
    }


def assert_result(case: dict[str, Any], result: dict[str, Any]) -> list[str]:
    expected = case.get("expected", {})
    failures: list[str] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    if "raises" in expected:
        check(result["raised"] is bool(expected["raises"]), f"raises={result['raised']} expected {expected['raises']}")
    else:
        check(not result["raised"], f"unexpected exception: {result['exception']}")

    if "must_compact" in expected:
        actual = result["compact_count"] > 0
        check(actual is bool(expected["must_compact"]), f"must_compact={actual} expected {expected['must_compact']}")
    for key in ("compact_count", "memory_version", "turn_count", "write_call_count", "pending_segment_tokens"):
        if key in expected:
            check(result[key] == expected[key], f"{key}={result[key]} expected {expected[key]}")
    if "min_compacts" in expected:
        check(result["compact_count"] >= expected["min_compacts"], f"compact_count={result['compact_count']} < {expected['min_compacts']}")
    if "read_enabled" in expected:
        check(result["read_enabled"] is bool(expected["read_enabled"]), f"read_enabled={result['read_enabled']} expected {expected['read_enabled']}")
    if "query_read_enabled" in expected and case.get("query_turn_index"):
        index = case["query_turn_index"] - 1
        flags = [x["read_enabled"] for x in result["results"]]
        check(0 <= index < len(flags) and flags[index] is bool(expected["query_read_enabled"]), f"query read flag={flags} expected index {case['query_turn_index']}={expected['query_read_enabled']}")
    if expected.get("all_generation_calls_read") is False:
        check(not any(result["generation_read_flags"]), f"precompact generation read flags={result['generation_read_flags']}")
    if expected.get("all_generation_calls_read_after_first"):
        flags = result["generation_read_flags"]
        check(bool(flags) and flags[0] is False and all(flags[1:]), f"generation read flags={flags}")
    if expected.get("unique_readable_segments"):
        ids = result["readable_segment_ids"]
        check(len(ids) == len(set(ids)), f"duplicate readable segments: {ids}")
    if expected.get("written_tokens_positive"):
        check(result["written_tokens"] > 0, f"written_tokens={result['written_tokens']}")
    if "readable_tokens" in expected:
        check(result["readable_tokens"] == expected["readable_tokens"], f"readable_tokens={result['readable_tokens']} expected {expected['readable_tokens']}")

    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=Path(__file__).with_name("cases.jsonl"))
    parser.add_argument("--case", action="append", dest="case_ids", help="run only this case; repeatable")
    parser.add_argument("--schedule", choices=["all", *SCHEDULES], default="all")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    cases = load_cases(args.cases)
    selected = set(args.case_ids or [])
    if selected:
        cases = [case for case in cases if case["case_id"] in selected]
        missing = selected - {case["case_id"] for case in cases}
        if missing:
            parser.error(f"unknown case(s): {', '.join(sorted(missing))}")
    all_results = []
    failed = 0
    for case in cases:
        schedules = case.get("schedules", list(SCHEDULES)) if args.schedule == "all" else [args.schedule]
        case_results = []
        for schedule in schedules:
            result = run_case(case, schedule)
            errors = assert_result(case, result)
            result["passed"] = not errors
            result["failures"] = errors
            case_results.append(result)
            all_results.append(result)
            status = "PASS" if not errors else "FAIL"
            print(f"{status} {case['case_id']} [{schedule}]")
            for error in errors:
                print(f"  - {error}")
            if errors:
                failed += 1
        if case.get("expected", {}).get("compare_schedule_outputs") and case_results:
            outputs = {json.dumps(item["generation_tokens"], sort_keys=True) for item in case_results}
            if len(outputs) != 1:
                message = "generation outputs differ across schedules"
                for item in case_results:
                    if item["passed"]:
                        item["passed"] = False
                        item["failures"].append(message)
                        failed += 1
                print(f"FAIL {case['case_id']} [cross-schedule]")
                print(f"  - {message}")
    report = {"passed": failed == 0, "results": all_results}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\n{len(all_results) - failed}/{len(all_results)} runs passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
