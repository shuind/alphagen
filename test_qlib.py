import os

import qlib
from qlib.config import REG_CN

from alphagen_qlib.stock_data import StockData
from alphagen_qlib.calculator import QLibStockDataCalculator
from alphagen.data.expression import Feature, FeatureType, Ref

import torch  # 👈 新增这一行


def main():
    provider_uri = os.path.join(
        os.path.expanduser("~"),
        ".qlib",
        "qlib_data",
        "cn_data_baostock_fwdadj",
    )
    print("Using provider_uri:", provider_uri)

    qlib.init(provider_uri=provider_uri, region=REG_CN)

    close = Feature(FeatureType.CLOSE)
    target = Ref(close, -20) / close - 1

    device = torch.device("cpu")  # 👈 强制用 CPU

    data = StockData(
        instrument="csi300",
        start_time="2018-01-01",
        end_time="2018-12-31",
        device=device,  # 👈 把 device 传进去
    )
    calc = QLibStockDataCalculator(data, target)

    print("n_days =", data.n_days)
    print("example IC =", calc.calc_single_IC_ret(target))


if __name__ == "__main__":
    main()
