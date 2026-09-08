"""Copy the metric tables from reports/*.json into README.md (between the markers).

Run after `make eval` (fixture, CI smoke test) and `make eval DATASET=full` (real code).
The README never contains a hand-typed number. Blocks:
  METRICS     <- reports/results_full.json  (real Labor Code)
  CI-METRICS  <- reports/results.json       (synthetic fixture, what CI checks)
"""

import json
from pathlib import Path

from tkrag.eval.report import update_readme

ROOT = Path(__file__).resolve().parents[1]
BLOCKS = {"METRICS": "results_full", "CI-METRICS": "results"}


def main() -> None:
    readme_path = ROOT / "README.md"
    text = readme_path.read_text(encoding="utf-8")
    for name, stem in BLOCKS.items():
        results_json = ROOT / "reports" / f"{stem}.json"
        if not results_json.exists():
            print(f"skip {name}: reports/{stem}.json is missing")
            continue
        results = json.loads(results_json.read_text(encoding="utf-8"))
        text = update_readme(text, results, f"reports/{stem}.md", name)
        print(f"README.md block {name} updated from reports/{stem}.json")
    readme_path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
