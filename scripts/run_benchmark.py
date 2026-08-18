#!/usr/bin/env python3
"""Agent benchmark comparison tool.

Run benchmark cases and compare metrics against historical results.

Usage:
    python scripts/run_benchmark.py [--history] [--cases path/to/cases.json]

Output:
- Prints a comparison table showing before/after metrics
- Saves results to .tmp/benchmark_history.json for future comparisons
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Ensure repo root is on path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_CASES = REPO_ROOT / "tests" / "eval" / "benchmark_cases.json"
DEFAULT_HISTORY = REPO_ROOT / ".tmp" / "benchmark_history.json"


def load_cases(path: Path) -> list[dict]:
    """Load benchmark cases from JSON file."""
    if not path.exists():
        print(f"Error: Cases file not found: {path}")
        sys.exit(1)
    return json.loads(path.read_text(encoding="utf-8"))


def load_history(path: Path) -> list[dict]:
    """Load historical benchmark results."""
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


def save_history(path: Path, records: list[dict]) -> None:
    """Append benchmark result to history file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    history = load_history(path)
    history.append(records[-1])  # Save the latest run summary
    # Keep only last 20 runs
    history = history[-20:]
    path.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")


def run_benchmark(
    cases: list[dict],
    workdir: Path | None = None,
) -> dict:
    """Execute all benchmark cases and return metrics."""
    from anthropic import Anthropic
    from tests.eval.lib.runner import AgentLoopRunner

    client = Anthropic()
    runner = AgentLoopRunner(
        workdir=workdir or REPO_ROOT / ".tmp" / "benchmark_workdir",
        client=client,
    )

    results = []
    for case in cases:
        print(f"Running case: {case['id']} ...", end=" ")
        try:
            result = runner.run(case["task"], max_turns=10)
            benchmark = result.to_benchmark(
                case_id=case["id"],
                expected_tool=case["expected_tool"],
            )
            results.append(benchmark)
            status = "✓" if benchmark.tool_correct else "✗"
            print(f"{status} ({benchmark.turns} turns, {benchmark.tokens} tokens)")
        except Exception as exc:
            print(f"FAILED: {exc}")
            results.append(
                type("BenchmarkResult", (), {
                    "case_id": case["id"],
                    "tool_correct": False,
                    "turns": 0,
                    "tokens": 0,
                    "expected_tool": case["expected_tool"],
                    "actual_tools": [],
                })()
            )

    # Compute summary metrics
    total = len(results)
    correct = sum(1 for r in results if r.tool_correct)
    avg_turns = sum(r.turns for r in results) / max(total, 1)
    avg_tokens = sum(r.tokens for r in results) / max(total, 1)

    return {
        "timestamp": datetime.now().isoformat(),
        "tool_accuracy": correct / total if total > 0 else 0.0,
        "avg_turns": round(avg_turns, 2),
        "avg_tokens": round(avg_tokens, 2),
        "total_turns": sum(r.turns for r in results),
        "total_tokens": sum(r.tokens for r in results),
        "cases": [
            {
                "case_id": r.case_id,
                "tool_correct": r.tool_correct,
                "turns": r.turns,
                "tokens": r.tokens,
                "expected_tool": r.expected_tool,
                "actual_tools": r.actual_tools,
            }
            for r in results
        ],
    }


def format_arrow(current: float, baseline: float, higher_is_better: bool = True) -> str:
    """Format a comparison arrow with color indicator."""
    if baseline == 0:
        if current == 0:
            return "  →"
        return "  ↑" if higher_is_better else "  ↓"

    diff = current - baseline
    pct = (diff / baseline) * 100

    if abs(pct) < 0.5:
        return f"  → {pct:+.1f}%"

    arrow = "↑" if (diff > 0) else "↓"
    direction = "better" if (diff > 0) == higher_is_better else "worse"
    color_symbol = "🟢" if direction == "better" else "🔴"

    return f" {arrow} {pct:+.1f}% {color_symbol}"


def print_comparison(
    baseline: dict | None,
    current: dict,
) -> None:
    """Print a comparison table."""
    print("\n" + "=" * 80)
    print("  Agent Benchmark Comparison")
    print("=" * 80)

    if baseline:
        print(f"\nBaseline: {baseline['timestamp']}")
    print(f"Current:  {current['timestamp']}\n")

    print("-" * 80)
    header = f"{'Metric':<20} {'Baseline':>10} {'Current':>10} {'Change':>15}"
    print(header)
    print("-" * 80)

    # Tool Accuracy
    base_acc = baseline.get("tool_accuracy", 0) if baseline else 0
    curr_acc = current["tool_accuracy"]
    acc_arrow = format_arrow(curr_acc, base_acc, higher_is_better=True)
    print(f"{'Tool Accuracy':<20} {base_acc:>9.1%} {curr_acc:>9.1%} {acc_arrow:>15}")

    # Avg Turns
    base_turns = baseline.get("avg_turns", 0) if baseline else 0
    curr_turns = current["avg_turns"]
    turns_arrow = format_arrow(curr_turns, base_turns, higher_is_better=False)
    print(f"{'Avg Turns':<20} {base_turns:>10.2f} {curr_turns:>10.2f} {turns_arrow:>15}")

    # Avg Tokens
    base_tokens = baseline.get("avg_tokens", 0) if baseline else 0
    curr_tokens = current["avg_tokens"]
    tokens_arrow = format_arrow(curr_tokens, base_tokens, higher_is_better=False)
    print(f"{'Avg Tokens':<20} {base_tokens:>10.1f} {curr_tokens:>10.1f} {tokens_arrow:>15}")

    # Total Turns
    base_total_turns = baseline.get("total_turns", 0) if baseline else 0
    curr_total_turns = current["total_turns"]
    total_turns_arrow = format_arrow(curr_total_turns, base_total_turns, higher_is_better=False)
    print(f"{'Total Turns':<20} {base_total_turns:>10} {curr_total_turns:>10} {total_turns_arrow:>15}")

    # Total Tokens
    base_total_tokens = baseline.get("total_tokens", 0) if baseline else 0
    curr_total_tokens = current["total_tokens"]
    total_tokens_arrow = format_arrow(curr_total_tokens, base_total_tokens, higher_is_better=False)
    print(f"{'Total Tokens':<20} {base_total_tokens:>10} {curr_total_tokens:>10} {total_tokens_arrow:>15}")

    print("-" * 80)

    # Per-case details
    print("\nPer-Case Details:")
    print("-" * 80)
    for case in current["cases"]:
        status = "✓ PASS" if case["tool_correct"] else "✗ FAIL"
        print(f"  {case['case_id']:<25} {status:<8} {case['turns']:>3} turns {case['tokens']:>6} tokens")
        if not case["tool_correct"]:
            expected = case["expected_tool"]
            actual = case.get("actual_tools", [])
            print(f"    Expected: {expected}")
            print(f"    Actual:   {', '.join(actual) if actual else '(none)'}")
    print("-" * 80)


def main():
    parser = argparse.ArgumentParser(description="Agent Benchmark Comparison Tool")
    parser.add_argument(
        "--cases",
        type=Path,
        default=DEFAULT_CASES,
        help="Path to benchmark cases JSON file",
    )
    parser.add_argument(
        "--history",
        type=Path,
        default=DEFAULT_HISTORY,
        help="Path to benchmark history file",
    )
    parser.add_argument(
        "--baseline",
        type=int,
        default=-1,
        help="Use Nth historical record as baseline (0-based, -1=latest)",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=None,
        help="Working directory for agent runs",
    )
    args = parser.parse_args()

    # Load cases
    cases = load_cases(args.cases)
    print(f"Loaded {len(cases)} benchmark cases from {args.cases}")

    # Load history
    history = load_history(args.history)
    baseline = None
    if history and args.baseline < len(history):
        baseline = history[args.baseline]
        print(f"Using baseline from: {baseline['timestamp']}")
    elif history:
        print(f"Warning: Invalid baseline index {args.baseline}, using latest")
        baseline = history[-1]
    else:
        print("No historical data found. This will be the first run.")

    # Run benchmark
    print("\nRunning benchmark...")
    print("-" * 80)
    current = run_benchmark(cases, workdir=args.workdir)

    # Print comparison
    print_comparison(baseline, current)

    # Save to history
    save_history(args.history, [current])
    print(f"\nResults saved to {args.history}")
    print("=" * 80)


if __name__ == "__main__":
    main()
