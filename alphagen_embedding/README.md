Hybrid Alpha Embedding V1

This module is an independent experimental branch for validating whether hybrid alpha embeddings are useful.

Core entrypoints
1. Build dataset
   `python scripts/build_embedding_dataset.py --provider_uri <qlib_data_path> --output_dir embedding_outputs/dataset_v1`

2. Train model
   `python scripts/train_hybrid_embedding.py --provider_uri <qlib_data_path>`

3. Analyze embeddings
   `python scripts/analyze_hybrid_embedding.py --train_summary embedding_outputs/<run_ts>/train_summary.json --split test`

Outputs
- `embedding_dataset.pt`
- `embedding_dataset_meta.json`
- `best_model.pt`
- `train_log.csv`
- `train_summary.json`
- `mean_embeddings_{train|valid|test}.pt`
- `embedding_export_{train|valid|test}.csv`
- `analysis/` with PCA, t-SNE, and nearest-neighbor outputs
