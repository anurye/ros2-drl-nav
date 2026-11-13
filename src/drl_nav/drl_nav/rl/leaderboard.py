# Copyright 2025 Ahmed Yesuf Nurye.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Build the results tables from ``results/*.json`` and write them into the README.

    ros2 run drl_nav leaderboard [--results-dir results] [--readme README.md]

One table per robot and tier; rows are algorithms (mean ± std over training seeds).
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import statistics
from typing import Dict, List, Tuple

START, END = "<!-- leaderboard:start -->", "<!-- leaderboard:end -->"
METRICS = [
    ("success_rate", "Success", 100),
    ("collision_rate", "Collision", 100),
    ("timeout_rate", "Timeout", 100),
    ("spl", "SPL", 100),
]
TIER_ORDER = ["easy", "medium", "hard"]

Reports = Dict[
    Tuple[str, str], Dict[str, List[Dict]]
]  # (robot, tier) -> algo -> reports


def load_results(results_dir: Path) -> Reports:
    groups: Reports = defaultdict(lambda: defaultdict(list))
    for path in sorted(results_dir.glob("*.json")):
        report = json.loads(path.read_text())
        key = (
            report.get("robot", {}).get("name", "unknown"),
            report["env"].get("name", "unknown"),
        )
        groups[key][report["algo"]].append(report)
    return groups


def fmt(values: List[float], scale: float) -> str:
    mean = statistics.mean(values) * scale
    if len(values) > 1:
        return f"{mean:.1f} ± {statistics.stdev(values) * scale:.1f}"
    return f"{mean:.1f}"


def table(by_algo: Dict[str, List[Dict]]) -> str:
    header = (
        "| Algorithm | Seeds | "
        + " | ".join(f"{label} %" for _, label, _ in METRICS)
        + " | Steps to goal | Model |"
    )
    rows = [header, "|" + "---|" * (len(METRICS) + 4)]
    ranked = sorted(
        by_algo.items(),
        key=lambda kv: -statistics.mean(r["summary"]["success_rate"] for r in kv[1]),
    )
    for algo, reports in ranked:
        cells = [algo, str(len(reports))]
        for key, _, scale in METRICS:
            cells.append(fmt([r["summary"][key] for r in reports], scale))
        steps = [
            r["summary"]["mean_steps_to_goal"]
            for r in reports
            if r["summary"]["mean_steps_to_goal"] is not None
        ]
        cells.append(f"{statistics.mean(steps):.0f}" if steps else "–")
        hub = next((r["hub_url"] for r in reports if r.get("hub_url")), None)
        cells.append(f"[HF]({hub})" if hub else "–")
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def render(groups: Reports) -> str:
    def order(key):
        robot, tier = key
        return robot, TIER_ORDER.index(tier) if tier in TIER_ORDER else 99, tier

    sections = []
    for robot, tier in sorted(groups, key=order):
        by_algo = groups[(robot, tier)]
        env_id = next(iter(next(iter(by_algo.values()))))["env"].get("id", "")
        sections.append(f"**{robot} / {tier}** (`{env_id}`)\n\n{table(by_algo)}")
    episodes = {
        r["eval"]["episodes"]
        for algos in groups.values()
        for rs in algos.values()
        for r in rs
    }
    sections.append(
        f"Evaluated on {', '.join(map(str, sorted(episodes)))} fixed scenarios per "
        "run (`evaluate`, seeds from 10000)."
    )
    return "\n\n".join(sections)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--readme", type=Path, default=Path("README.md"))
    args = parser.parse_args()

    groups = load_results(args.results_dir)
    if not groups:
        raise SystemExit(f"no results in {args.results_dir}")
    body = render(groups)
    print(body)

    text = args.readme.read_text()
    if START not in text or END not in text:
        raise SystemExit(f"{args.readme} lacks the {START} / {END} markers")
    text = re.sub(
        f"{re.escape(START)}.*?{re.escape(END)}",
        f"{START}\n{body}\n{END}",
        text,
        flags=re.S,
    )
    args.readme.write_text(text)


if __name__ == "__main__":
    main()
