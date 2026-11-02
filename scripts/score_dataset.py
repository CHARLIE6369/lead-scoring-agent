"""Score a lead CSV from the command line.

Usage:
    python scripts/score_dataset.py data/lead_scoring_sample_dataset_500.csv outputs/scored_leads.csv
    add --llm to also generate LLM explanations (needs GROQ_API_KEY).
"""
import collections
import csv
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.llm import enrich_with_llm  # noqa: E402
from app.scoring import FLAT_FIELDS, score_lead, to_flat_row  # noqa: E402


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    use_llm = "--llm" in sys.argv
    src = pathlib.Path(args[0]) if args else ROOT / "data" / "lead_scoring_sample_dataset_500.csv"
    dst = pathlib.Path(args[1]) if len(args) > 1 else ROOT / "outputs" / "scored_leads.csv"
    dst.parent.mkdir(parents=True, exist_ok=True)

    with open(src, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    results = []
    for raw in rows:
        r = score_lead(raw)
        results.append(enrich_with_llm(r) if use_llm else r)

    with open(dst, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FLAT_FIELDS)
        writer.writeheader()
        for r in results:
            writer.writerow(to_flat_row(r))

    counts = collections.Counter(r["classification"] for r in results)
    avg = sum(r["score"] for r in results) / len(results)
    print(f"Scored {len(results)} leads -> {dst}")
    print(f"Hot: {counts['Hot']}  Warm: {counts['Warm']}  Cold: {counts['Cold']}  Average: {avg:.1f}")


if __name__ == "__main__":
    main()
