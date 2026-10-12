#!/usr/bin/env python3
"""Summarize LightLLM benchmark JSON files into a Markdown comparison table.

Expected filename example:
    fcfs_rate1_req4000.json
    sjf_rate4_req4000_run2.json

Usage:
    python analyze_lightllm_results.py ./results --metric latency
    python analyze_lightllm_results.py ./results --metric ttft
    python analyze_lightllm_results.py ./results --metric ttft --output ./ttft_comparison.md

Percentiles are computed from per-request data using linear interpolation (the
same convention as NumPy's default percentile method). Latency/TTFT values in
this benchmark JSON are assumed to be seconds and are displayed in milliseconds.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

PERCENTILES = (10, 20, 30, 40, 50, 60, 70, 80, 90, 99)
# Preferred display order for scheduling strategies; unknown strategies follow alphabetically.
STRATEGY_ORDER = ("fcfs", "sjf", "ljf", "bub")
FILENAME_RE = re.compile(
    r"^(?P<strategy>.+?)_rate(?P<rate>\d+(?:\.\d+)?)_req(?P<req>\d+)"
    r"(?:_run(?P<run>\d+))?$",
    re.IGNORECASE,
)


@dataclass
class Experiment:
    path: Path
    strategy: str
    rate: float | None
    configured_requests: int | None
    run_number: int | None
    summary: dict[str, Any]
    requests: list[dict[str, Any]]
    hit_max_total_tokens: int = 0
    warnings: list[str] = field(default_factory=list)


def as_float(value: Any) -> float | None:
    """Convert a JSON value to a finite float, or None if unavailable."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def as_int(value: Any) -> int | None:
    number = as_float(value)
    if number is None:
        return None
    return int(number)


def parse_filename(path: Path) -> tuple[str, float | None, int | None, int | None]:
    """Parse strategy/rate/request count/run number from a conventional filename."""
    stem = path.stem
    match = FILENAME_RE.match(stem)
    if match:
        return (
            match.group("strategy").lower(),
            as_float(match.group("rate")),
            as_int(match.group("req")),
            as_int(match.group("run")),
        )

    # Graceful fallback for filenames that do not follow the full convention.
    strategy = re.split(r"_rate", stem, maxsplit=1, flags=re.IGNORECASE)[0].lower()
    return strategy or stem.lower(), None, None, None


def load_experiment(path: Path) -> Experiment:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise ValueError("JSON 根节点不是 object")

    config = payload.get("config") or {}
    summary = payload.get("summary") or {}
    requests_raw = payload.get("requests") or []
    if not isinstance(config, dict) or not isinstance(summary, dict):
        raise ValueError("config 或 summary 不是 object")
    if not isinstance(requests_raw, list):
        raise ValueError("requests 不是 list")

    strategy, filename_rate, filename_count, run_number = parse_filename(path)
    config_rate = as_float(config.get("request_rate"))
    config_count = as_int(config.get("num_prompts"))

    # Prefer the values stored in config for the actual run conditions;
    # filename metadata is used as a fallback and checked for inconsistencies.
    rate = config_rate if config_rate is not None else filename_rate
    configured_requests = config_count if config_count is not None else filename_count

    requests = [item for item in requests_raw if isinstance(item, dict)]
    warnings: list[str] = []

    if filename_rate is not None and config_rate is not None and not math.isclose(
        filename_rate, config_rate, rel_tol=0, abs_tol=1e-9
    ):
        warnings.append(
            f"文件名 rate={filename_rate:g}，但 config.request_rate={config_rate:g}；表格采用 config 值"
        )
    if filename_count is not None and config_count is not None and filename_count != config_count:
        warnings.append(
            f"文件名 req={filename_count}，但 config.num_prompts={config_count}；表格采用 config 值"
        )

    expected_requests = as_int(summary.get("total_requests"))
    if expected_requests is not None and expected_requests != len(requests):
        warnings.append(
            f"summary.total_requests={expected_requests}，requests 明细条数={len(requests)}"
        )

    hit_count = as_int(summary.get("hit_max_total_tokens")) or 0
    return Experiment(
        path=path,
        strategy=strategy,
        rate=rate,
        configured_requests=configured_requests,
        run_number=run_number,
        summary=summary,
        requests=requests,
        hit_max_total_tokens=max(hit_count, 0),
        warnings=warnings,
    )


def linear_percentile(values: Iterable[float], percentile: int) -> float | None:
    """Return a percentile using linear interpolation between sorted values."""
    data = sorted(values)
    if not data:
        return None
    if len(data) == 1:
        return data[0]

    position = (len(data) - 1) * (percentile / 100.0)
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return data[lower_index]
    fraction = position - lower_index
    return data[lower_index] * (1.0 - fraction) + data[upper_index] * fraction


def request_values(experiment: Experiment, metric: str) -> list[float]:
    values: list[float] = []
    for request in experiment.requests:
        value = as_float(request.get(metric))
        if value is not None:
            values.append(value)
    return values


def check_summary_percentiles(experiment: Experiment) -> None:
    """Compare available summary latency percentiles with request-level values."""
    values = request_values(experiment, "latency")
    if not values:
        return

    for percentile in (10, 20, 50, 90, 99):
        key = f"p{percentile}_latency"
        summary_value = as_float(experiment.summary.get(key))
        calculated = linear_percentile(values, percentile)
        if summary_value is None or calculated is None:
            continue
        # A >1ms difference may indicate a different percentile convention,
        # incomplete request details, or an inconsistent JSON file.
        if abs(summary_value - calculated) > 0.001:
            experiment.warnings.append(
                f"summary.{key}={summary_value * 1000:.2f} ms，"
                f"requests 重算={calculated * 1000:.2f} ms（差异超过 1 ms）"
            )


def strategy_sort_key(strategy: str) -> tuple[int, str]:
    """Return the preferred scheduler order, then alphabetical for unknown strategies."""
    normalized = strategy.strip().lower()
    try:
        return STRATEGY_ORDER.index(normalized), normalized
    except ValueError:
        return len(STRATEGY_ORDER), normalized


def format_number(value: float | int | None, digits: int = 2) -> str:
    if value is None:
        return "—"
    if isinstance(value, float) and not math.isfinite(value):
        return "—"
    return f"{value:.{digits}f}"


def markdown_escape(text: str) -> str:
    return str(text).replace("|", r"\|").replace("\n", " ").replace("\r", " ")


def generate_markdown_table(experiments: list[Experiment], metric: str = "latency") -> str:
    """Generate Markdown tables grouped by request rate for easier scanning."""
    if metric not in {"latency", "ttft"}:
        raise ValueError("metric 必须是 'latency' 或 'ttft'")

    valid = [item for item in experiments if item.hit_max_total_tokens == 0]
    valid.sort(
        key=lambda item: (
            item.rate is None,
            item.rate if item.rate is not None else math.inf,
            strategy_sort_key(item.strategy),
            item.run_number if item.run_number is not None else 0,
            item.path.name.lower(),
        )
    )

    percentile_label = "Latency" if metric == "latency" else "TTFT"
    columns = [
        "Rate",
        "策略",
        "实际吞吐量 (req/s)",
        "平均延迟 (ms)",
        "平均 TTFT (ms)",
        "平均 ITL (ms)",
    ] + [f"{percentile_label} P{p} (ms)" for p in PERCENTILES]

    lines = [
        f"# LightLLM 实验对比（{percentile_label} 分位数）",
        "",
        "- 延迟单位：ms；吞吐量单位：req/s。",
        f"- P10～P99 根据每个请求的 `requests[].{metric}` 计算，使用线性插值；缺少数据时显示 `—`。",
        "- 不同 Rate 分成独立表格，方便比较同一负载下的不同策略。",
        "- `hit_max_total_tokens > 0` 的实验不纳入主表，会列在下方的排除清单。",
        "",
    ]

    # Keep rows for the same request rate in one table, with a heading between rates.
    groups: dict[float | None, list[Experiment]] = {}
    for item in valid:
        groups.setdefault(item.rate, []).append(item)

    for rate, group in groups.items():
        if rate is None:
            rate_heading = "未知"
            rate_text = "—"
        else:
            rate_heading = format_number(rate, 3).rstrip("0").rstrip(".")
            rate_text = rate_heading

        lines.extend([
            f"## Rate = **{rate_heading} req/s**",
            "",
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join(["---:", "---"] + ["---:"] * (len(columns) - 2)) + " |",
        ])

        for item in group:
            summary = item.summary
            # The benchmark's summary latency and TTFT are in seconds; ITL is already ms.
            throughput = as_float(summary.get("throughput"))
            avg_latency = as_float(summary.get("avg_latency"))
            avg_ttft = as_float(summary.get("avg_ttft"))
            avg_itl = as_float(summary.get("avg_inter_token_latency_ms"))

            values = request_values(item, metric)
            percentiles = [linear_percentile(values, p) for p in PERCENTILES]
            percentile_ms = [value * 1000 if value is not None else None for value in percentiles]

            strategy_text = item.strategy.upper()
            if item.run_number is not None:
                strategy_text += f" (run{item.run_number})"

            row = [
                f"**{rate_text}**",
                strategy_text,
                format_number(throughput, 3),
                format_number(avg_latency * 1000 if avg_latency is not None else None, 2),
                format_number(avg_ttft * 1000 if avg_ttft is not None else None, 2),
                format_number(avg_itl, 2),
            ] + [format_number(value, 2) for value in percentile_ms]
            lines.append("| " + " | ".join(markdown_escape(value) for value in row) + " |")

        lines.append("")

    if not valid:
        lines.extend(["## 正常实验", "", "没有可用的正常实验。", ""])

    excluded = [item for item in experiments if item.hit_max_total_tokens > 0]
    lines.extend(["## 排除的实验", ""])
    if excluded:
        lines.extend([
            "以下实验因 `hit_max_total_tokens > 0` 未纳入主表：",
            "",
            "| 文件 | Rate | 策略 | hit_max_total_tokens |",
            "| --- | ---: | --- | ---: |",
        ])
        for item in sorted(
            excluded,
            key=lambda x: (
                x.rate is None,
                x.rate if x.rate is not None else math.inf,
                strategy_sort_key(x.strategy),
                x.path.name.lower(),
            ),
        ):
            rate_text = format_number(item.rate, 3).rstrip("0").rstrip(".") if item.rate is not None else "—"
            lines.append(
                f"| {markdown_escape(item.path.name)} | {rate_text} | "
                f"{markdown_escape(item.strategy.upper())} | {item.hit_max_total_tokens} |"
            )
    else:
        lines.append("没有排除实验。")

    with_warnings = [item for item in experiments if item.warnings]
    if with_warnings:
        lines.extend(["", "## 数据检查提醒", ""])
        for item in sorted(with_warnings, key=lambda x: x.path.name.lower()):
            for warning in item.warnings:
                lines.append(f"- `{markdown_escape(item.path.name)}`：{markdown_escape(warning)}")

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="读取 LightLLM benchmark JSON，按 rate/策略排序并输出 Markdown 对比表。"
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default=".",
        help="包含 JSON 结果的文件夹（默认：当前文件夹）",
    )
    parser.add_argument(
        "--metric",
        choices=("latency", "ttft"),
        default="latency",
        help="分位数字段使用端到端 latency 或 TTFT（默认：latency）",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="输出 Markdown 文件路径（默认：文件夹内 comparison_<metric>.md）",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    directory = Path(args.directory).expanduser().resolve()
    if not directory.is_dir():
        print(f"错误：文件夹不存在：{directory}", file=sys.stderr)
        return 2

    json_files = sorted(directory.glob("*.json"))
    if not json_files:
        print(f"错误：文件夹中找不到 JSON：{directory}", file=sys.stderr)
        return 1

    experiments: list[Experiment] = []
    load_errors: list[tuple[str, str]] = []
    for path in json_files:
        try:
            experiment = load_experiment(path)
            check_summary_percentiles(experiment)
            experiments.append(experiment)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            load_errors.append((path.name, str(exc)))

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else directory / f"comparison_{args.metric}.md"
    )

    markdown = generate_markdown_table(experiments, metric=args.metric)
    if load_errors:
        markdown += "\n## 无法读取的 JSON\n\n"
        for filename, error in load_errors:
            markdown += f"- `{markdown_escape(filename)}`：{markdown_escape(error)}\n"

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(markdown, encoding="utf-8")
    except OSError as exc:
        print(f"错误：无法写入输出文件 {output_path}: {exc}", file=sys.stderr)
        return 2

    excluded_count = sum(item.hit_max_total_tokens > 0 for item in experiments)
    print(f"已扫描 JSON：{len(json_files)}")
    print(f"成功读取：{len(experiments)}；读取失败：{len(load_errors)}")
    print(f"纳入主表：{len(experiments) - excluded_count}；排除异常实验：{excluded_count}")
    print(f"分位数指标：{args.metric}")
    print(f"Markdown 已输出：{output_path}")

    if load_errors:
        print("\n读取失败的文件：", file=sys.stderr)
        for filename, error in load_errors:
            print(f"  {filename}: {error}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
