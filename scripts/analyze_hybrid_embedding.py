import argparse
import json
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from alphagen_embedding.analysis import nearest_neighbor_analysis, save_embedding_visualizations
from alphagen_embedding.analysis import embedding_deduplicate, embedding_diverse_selection
from alphagen_embedding.analysis import compare_alpha_subsets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_summary", required=True)
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    parser.add_argument("--dedup_distance_threshold", type=float, default=None)
    parser.add_argument("--select_k", type=int, default=10)
    parser.add_argument("--lambda_diversity", type=float, default=0.35)
    args = parser.parse_args()

    summary_path = Path(args.train_summary)
    with summary_path.open("r", encoding="utf-8") as f:
        train_summary = json.load(f)

    output_dir = summary_path.parent / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(train_summary["best_checkpoint"], map_location="cpu")
    alpha_summary = checkpoint["alpha_summary"]
    embeddings = torch.load(summary_path.parent / f"mean_embeddings_{args.split}.pt", map_location="cpu")
    dataset = torch.load(train_summary["dataset_path"], map_location="cpu")

    figures = save_embedding_visualizations(embeddings, alpha_summary, output_dir, args.split)
    nn_summary = nearest_neighbor_analysis(embeddings, alpha_summary, output_dir, args.split)
    dedup_summary = embedding_deduplicate(
        embeddings,
        alpha_summary,
        output_dir,
        args.split,
        distance_threshold=args.dedup_distance_threshold,
    )
    diverse_summary = embedding_diverse_selection(
        embeddings,
        alpha_summary,
        output_dir,
        args.split,
        select_k=args.select_k,
        lambda_diversity=args.lambda_diversity,
    )
    result = {
        "split": args.split,
        "figures": figures,
        "nearest_neighbor": nn_summary,
        "deduplication": dedup_summary,
        "diverse_selection": diverse_summary,
        "subset_ic_comparison": compare_alpha_subsets(dataset, dedup_summary, diverse_summary, output_dir),
    }
    result_path = output_dir / f"analysis_{args.split}.json"
    with result_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
