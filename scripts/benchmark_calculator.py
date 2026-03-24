import argparse
import json
import os
import time
from pathlib import Path

from alphagen.data.expression import Expression
from alphagen_qlib.calculator import QLibStockDataCalculator
from alphagen_qlib.compat import patch_all


def load_pool_exprs(pool_json_path: Path):
    with pool_json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    exprs = [eval(expr_str, globals()) for expr_str in data["exprs"]]
    weights = data["weights"]
    return exprs, weights


def build_target():
    from alphagen.data.expression import Feature, Ref
    from alphagen_qlib.stock_data import FeatureType
    close = Feature(FeatureType.CLOSE)
    return Ref(close, -20) / close - 1


def time_once(fn):
    start = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - start
    return elapsed, result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider_uri", required=True)
    parser.add_argument("--pool_json", required=True)
    parser.add_argument("--market", default="csi300")
    parser.add_argument("--start_time", default="2021-01-01")
    parser.add_argument("--end_time", default="2022-12-31")
    parser.add_argument("--cache_size", type=int, default=128)
    args = parser.parse_args()

    os.environ["ALPHAGEN_ALPHA_CACHE_SIZE"] = str(args.cache_size)
    patch_all(args.provider_uri)

    import qlib
    from qlib.constant import REG_CN
    from alphagen_qlib.stock_data import StockData

    qlib.init(provider_uri=args.provider_uri, region=REG_CN)
    data = StockData(instrument=args.market, start_time=args.start_time, end_time=args.end_time)
    calculator = QLibStockDataCalculator(data, build_target())
    exprs, weights = load_pool_exprs(Path(args.pool_json))

    print(f"cache_size={args.cache_size}")
    print(f"expr_count={len(exprs)}")

    elapsed_single, _ = time_once(lambda: calculator.calc_single_IC_ret(exprs[0]))
    elapsed_single_cached, _ = time_once(lambda: calculator.calc_single_IC_ret(exprs[0]))
    print(f"single_ic_first={elapsed_single:.4f}s")
    print(f"single_ic_cached={elapsed_single_cached:.4f}s")

    elapsed_mutual, _ = time_once(lambda: calculator.calc_mutual_IC(exprs[0], exprs[1]))
    elapsed_mutual_cached, _ = time_once(lambda: calculator.calc_mutual_IC(exprs[0], exprs[1]))
    print(f"mutual_ic_first={elapsed_mutual:.4f}s")
    print(f"mutual_ic_cached={elapsed_mutual_cached:.4f}s")

    elapsed_batch, batch_values = time_once(lambda: calculator.calc_mutual_IC_batch(exprs[0], exprs[1:]))
    print(f"mutual_ic_batch={elapsed_batch:.4f}s")
    print(f"mutual_ic_batch_count={len(batch_values)}")

    elapsed_pool, pool_ic = time_once(lambda: calculator.calc_pool_IC_ret(exprs, weights))
    print(f"pool_ic={pool_ic:.6f}")
    print(f"pool_ic_time={elapsed_pool:.4f}s")


if __name__ == "__main__":
    from alphagen.data.expression import *  # noqa: F401,F403

    main()
