"""Execute the reproducible real UCI experiment and save genuine result artifacts."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    from src.experiments.retail_experiment import run_uci_experiment

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/experiments/uci_daily")
    parser.add_argument("--holdout", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--backtest-folds", type=int, default=3)
    parser.add_argument("--backtest-horizon", type=int, default=14)
    parser.add_argument("--gemini", action="store_true", help="Request a live explanation from aggregate computed results.")
    args = parser.parse_args()
    result = run_uci_experiment(args.output, args.holdout, args.horizon, args.backtest_folds, args.backtest_horizon, args.gemini)
    print(result.evaluation.comparison.to_string(index=False))
    print(f"Selected by holdout RMSE: {result.evaluation.best_model}")
    print(f"Explanation source: {result.insights.source}; status: {result.insights.status}")
    print(f"Artifacts: {result.output_directory.resolve()}")


if __name__ == "__main__":
    main()
