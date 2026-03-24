import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from alphagen_embedding.dataset import build_embedding_dataset, load_alpha_source_file, save_embedding_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_root", default="kaggle/working/checkpoints")
    parser.add_argument("--runs_root", default="kaggle/working/runs")
    parser.add_argument("--alpha_source_file", default="")
    parser.add_argument("--provider_uri", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--market", default="csi300")
    parser.add_argument("--start_time", default="2021-01-01")
    parser.add_argument("--end_time", default="2022-12-31")
    parser.add_argument("--num_alpha", type=int, default=20)
    parser.add_argument("--stock_limit", type=int, default=100)
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    alpha_candidates = load_alpha_source_file(args.alpha_source_file) if args.alpha_source_file else None
    dataset = build_embedding_dataset(
        checkpoint_root=args.checkpoint_root,
        runs_root=args.runs_root,
        provider_uri=args.provider_uri,
        market=args.market,
        start_time=args.start_time,
        end_time=args.end_time,
        num_alpha=args.num_alpha,
        stock_limit=args.stock_limit,
        lookback=args.lookback,
        horizon=args.horizon,
        device=args.device,
        alpha_candidates=alpha_candidates,
    )
    paths = save_embedding_dataset(dataset, Path(args.output_dir))
    print(paths)


if __name__ == "__main__":
    main()
