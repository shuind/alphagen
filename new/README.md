# Clean Multi-Head AlphaGen Line

This directory is a clean rebuild path for the thesis final method:

- `single_transformer`: shared Transformer with only the base head.
- `multihead`: shared Transformer with `base/simple/ts/pv/rank/explore` heads, no intrinsic reward.
- `multihead_intrinsic`: multi-head policy plus AST count-based intrinsic reward on the explore head.
- Classic-factor behavior cloning pretrains multi-head policies by default. It uses built-in
  Alpha101/Alpha158-style RPN token templates and can be extended with `--classic-factor-csv`.

The old `train_maskable_ppo.py` and RI reward experiments are intentionally not reused as the main method.
Only the stable infrastructure is reused: expression grammar, action masks, Qlib calculator, AlphaPool scoring,
checkpoint format, and post-hoc evaluation scripts.

Smoke test:

```powershell
python -m new.train_multihead_ppo 0 tcsi300 10 `
  --step 2048 `
  --method multihead_intrinsic `
  --head-pretrain-epochs 1 `
  --intrinsic-beta 0.1 `
  --provider_uri "C:\Users\qdz\.qlib\qlib_data\cn_data_baostock_fwdadj\tcsi300_kaggle_subset_20260326_162656" `
  --logdir runs_smoke_new `
  --ckpt_dir ckpt_smoke_new `
  --tb_dir tb_smoke_new `
  --device cpu
```
