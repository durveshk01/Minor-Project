"""Verify server-side Gemini using only aggregates from the real UCI sample."""

import argparse
import json
import logging
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def main() -> None:
    from src.ai.gemini_client import GeminiClient
    from src.ai.insight_generator import build_analytics_context, generate_insights
    from src.data.loader import load_dataset
    from src.data.pipeline import process_dataset
    from src.data.schema import AggregationOptions, ColumnMapping
    from src.features.time_features import seasonal_decomposition

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/qa/gemini-live.json")
    parser.add_argument("--model", default=None)
    args = parser.parse_args()
    logging.getLogger("httpx").setLevel(logging.ERROR)
    logging.getLogger("google").setLevel(logging.ERROR)
    path = ROOT / "data/sample/uci_online_retail_daily.csv"
    loaded = load_dataset(path)
    metadata = json.loads(path.with_suffix(".metadata.json").read_text(encoding="utf-8"))
    dataset = process_dataset(
        loaded.frame,
        ColumnMapping("Date", "Sales"),
        aggregation=AggregationOptions(missing_periods="ffill", source_frequency="D"),
        metadata=metadata,
    )
    seasonal = seasonal_decomposition(dataset.series, 7)
    context = build_analytics_context(dataset, seasonal=seasonal)
    with GeminiClient(model=args.model, max_retries=0) as client:
        connectivity = client.check_connectivity()
        explanation = generate_insights(context, client=client)
    evidence = {
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "dataset": path.name,
        "dataset_fingerprint": dataset.fingerprint,
        "raw_records_sent": 0,
        "analytics_sections": list(context),
        "connectivity": asdict(connectivity),
        "explanation": asdict(explanation),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({
        "connectivity_status": connectivity.status,
        "explanation_status": explanation.status,
        "explanation_source": explanation.source,
        "model": explanation.model,
        "evidence": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
