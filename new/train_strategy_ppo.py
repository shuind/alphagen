"""Strategy-generation training entrypoint.

This module is a naming wrapper around the historical implementation module.
It keeps old imports stable while allowing commands and reports to use the
less ambiguous "strategy" terminology.
"""

import runpy


if __name__ == "__main__":
    runpy.run_module("new.train_multihead_ppo", run_name="__main__")
