Kaggle 运行说明（中文）

1) 安装依赖（建议固定版本）
```bash
!pip install \
  stable-baselines3==2.0.0 \
  sb3_contrib==2.0.0 \
  protobuf==3.20.3 \
  tensorboard==2.15.2 \
  shimmy==1.1.0 \
  gym==0.26.2 \
  fire \
  loguru

!pip install "git+https://github.com/microsoft/qlib.git@v0.8.6"
```

2) 克隆仓库并切换分支
```bash
git clone https://github.com/shuind/alphagen.git
cd alphagen
git checkout exp/kaggle_qlib_fix

```

3) 可选环境变量
```bash
export QLIB_PROVIDER_URI=/kaggle/input/baostock/cn_data_baostock_fwdadj
export CKPT_DIR=/kaggle/working/checkpoints
export TB_DIR=/kaggle/working/tb_log
```

4) 推荐运行方式
```bash
python train_maskable_ppo.py 0 csi300 50 --step=2000
```

说明
- 全市场（如 csi300 以外更大范围）可能更容易 OOM，因为 rankIC 计算是 O(s^2) 级别，s 为股票数量。
- 建议优先使用 csi300 以降低显存/内存压力。
