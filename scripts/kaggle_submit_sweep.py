import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


LOCAL_REWARD_PATCH_MARKER = "# codex-local-reward-patch-20260428"

LOCAL_REWARD_PATCH_SOURCE = r"""# codex-local-reward-patch-20260428
from pathlib import Path


def _replace_once(path: str, old: str, new: str) -> None:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if old in text:
        p.write_text(text.replace(old, new, 1), encoding="utf-8")
        return
    if new in text:
        return
    raise RuntimeError(f"patch pattern not found in {path}: {old[:120]!r}")


def _replace_all(path: str, old: str, new: str) -> None:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if old in text:
        p.write_text(text.replace(old, new), encoding="utf-8")
        return
    if new in text:
        return
    raise RuntimeError(f"patch pattern not found in {path}: {old[:120]!r}")


_replace_once(
    "train_maskable_ppo.py",
    '''choices=["re", "re+func", "re+struct", "re+reg", "re+func+struct", "re+all",
                                 "re_v2", "re_v2+func", "re_v2+struct", "re_v2+reg", "re_v2+all"])''',
    '''choices=["re", "re+func", "re+struct", "re+reg", "re+func+struct", "re+all",
                                 "re+func_v2", "re+struct_v2", "re+reg_v2",
                                 "re+struct_v2+reg_v2", "re+all_v2"])''',
)

alpha_path = "alphagen/models/alpha_pool.py"
_replace_once(alpha_path, 'return "residual" if self.reward_mode.startswith("re_v2") else "mutual_ic"', 'return "residual" if self._use_func_v2 else "mutual_ic"')
_replace_once(alpha_path, "ri_reg = self._calc_ri_reg_v2(expr, token_seq) if self._use_v2 and self._use_ri_reg else (", "ri_reg = self._calc_ri_reg_v2(expr, token_seq) if self._use_reg_v2 and self._use_ri_reg else (")
_replace_once(alpha_path, '"admission_gate_enabled": bool(self.ri_admission_gate and self._use_v2),', '"admission_gate_enabled": bool(self.ri_admission_gate and self._use_any_v2),')
_replace_once(
    alpha_path,
    '''    @property
    def _use_v2(self) -> bool:
        return self.reward_mode.startswith("re_v2")''',
    '''    @property
    def _use_func_v2(self) -> bool:
        return "func_v2" in self.reward_mode or "all_v2" in self.reward_mode

    @property
    def _use_struct_v2(self) -> bool:
        return "struct_v2" in self.reward_mode or "all_v2" in self.reward_mode

    @property
    def _use_reg_v2(self) -> bool:
        return "reg_v2" in self.reward_mode or "all_v2" in self.reward_mode

    @property
    def _use_any_v2(self) -> bool:
        return self._use_func_v2 or self._use_struct_v2 or self._use_reg_v2''',
)
_replace_once(alpha_path, "if self._use_v2:\n            reward_lambda = self.lambda_ri / (1.0 + self.ri_schedule_decay * max(self.eval_cnt, 0))", "if self._use_any_v2:\n            reward_lambda = self.lambda_ri / (1.0 + self.ri_schedule_decay * max(self.eval_cnt, 0))")
_replace_once(alpha_path, '''        if self.reward_mode == "re_v2":
            return re, reward_lambda
''', "")
_replace_once(alpha_path, "        y = candidate_value.reshape(-1).float()\n", "        y = candidate_value.reshape(-1).float()\n        target = target_value.reshape(-1).float().to(y.device)\n")
_replace_once(alpha_path, "            y = y[sample_idx]\n", "            y = y[sample_idx]\n            target = target[sample_idx]\n")
_replace_once(alpha_path, "        y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)\n", "        y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)\n        target = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)\n")
_replace_once(alpha_path, "        # For sampled path, metric is computed on sampled vectors to reduce memory footprint.", "        # Score the candidate's pool-orthogonal residual against the return target.")
_replace_once(alpha_path, "                target_metric = y.detach().cpu().view(1, -1)", "                target_metric = target.detach().cpu().view(1, -1)")
_replace_all(alpha_path, "                target_metric = y.view(1, -1)", "                target_metric = target.view(1, -1)")
_replace_once(alpha_path, "            target_metric = y.view(1, -1)", "            target_metric = target.view(1, -1)")
_replace_once(alpha_path, "        if self._use_v2:\n            return self._calc_ri_struct_v2(token_seq, re_value)", "        if self._use_struct_v2:\n            return self._calc_ri_struct_v2(token_seq, re_value)")

print("Applied local reward patch for re+func_v2 and re+struct_v2+reg_v2.")
!python -m py_compile train_maskable_ppo.py alphagen/models/alpha_pool.py
"""


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _split_csv(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def _expand_reward_modes(
    reward_modes: Optional[str],
    reward_families: str,
    reward_suffixes: str,
) -> List[str]:
    if reward_modes:
        return _split_csv(reward_modes)

    suffix_map = {
        "base": "",
        "func": "+func",
        "struct": "+struct",
        "reg": "+reg",
        "all": "+all",
        "func_v2": "+func_v2",
        "struct_v2": "+struct_v2",
        "reg_v2": "+reg_v2",
        "struct_v2_reg_v2": "+struct_v2+reg_v2",
        "all_v2": "+all_v2",
    }
    families = _split_csv(reward_families)
    suffixes = _split_csv(reward_suffixes)
    expanded: List[str] = []
    for fam in families:
        if fam != "re":
            raise ValueError(f"unsupported reward family: {fam}")
        for s in suffixes:
            if s not in suffix_map:
                raise ValueError(f"unsupported reward suffix: {s}")
            expanded.append(f"{fam}{suffix_map[s]}")
    return expanded


def _run(cmd: List[str], cwd: Optional[Path] = None) -> str:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stdout}")
    return proc.stdout


def _run_with_retries(cmd: List[str], retries: int, backoff_sec: float, cwd: Optional[Path] = None) -> str:
    last_err: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            return _run(cmd, cwd=cwd)
        except Exception as e:
            last_err = e
            if attempt >= retries:
                break
            sleep_for = backoff_sec * attempt
            print(f"    [retry] attempt={attempt}/{retries} failed, sleep={sleep_for:.1f}s")
            time.sleep(sleep_for)
    raise RuntimeError(f"all retries failed ({retries}): {last_err}")


def _detect_notebook_file(kernel_dir: Path) -> Path:
    notebooks = sorted(kernel_dir.glob("*.ipynb"))
    if not notebooks:
        raise FileNotFoundError(f"no .ipynb found in kernel dir: {kernel_dir}")
    return notebooks[0]


def _replace_or_insert_flag(cmd: str, flag: str, value: str) -> str:
    pattern = re.compile(rf"({re.escape(flag)}\s+)([^\s\\]+)")
    if pattern.search(cmd):
        return pattern.sub(rf"\1{value}", cmd)
    cmd = cmd.rstrip()
    if not cmd.endswith("\\"):
        cmd += " \\"
    return f"{cmd}\n  {flag} {value} \\"


def _extract_first(pattern: str, text: str, default: str) -> str:
    m = re.search(pattern, text)
    return m.group(1) if m else default


def _choose_flag_value(source: str, flag: str, override: Optional[str], default: str) -> str:
    if override is not None and str(override).strip() != "":
        return str(override).strip()
    return _extract_first(rf"{re.escape(flag)}\s+([^\s\\]+)", source, default)


def _render_train_cell(
    source: str,
    seed: int,
    market_override: Optional[str],
    backbone: str,
    reward_mode: str,
    step: int,
    pool: Optional[int],
    run_name: str,
    ri_func_backend: Optional[str] = None,
    ri_struct_backend: Optional[str] = None,
    ri_corr_threshold: Optional[str] = None,
    ri_ast_similarity_threshold: Optional[str] = None,
    ri_corr_value_bonus: Optional[str] = None,
    ri_corr_underexplore_power: Optional[str] = None,
    ri_turnover_weight: Optional[str] = None,
    ri_turnover_topk: Optional[str] = None,
    ri_turnover_baseline: Optional[str] = None,
    ri_turnover_step_stride: Optional[str] = None,
    train_start_time: Optional[str] = None,
    train_end_time: Optional[str] = None,
    valid_start_time: Optional[str] = None,
    valid_end_time: Optional[str] = None,
    test_start_time: Optional[str] = None,
    test_end_time: Optional[str] = None,
) -> str:
    market = market_override or _extract_first(r"train_maskable_ppo\.py\s+\d+\s+([^\s\\]+)\s+\d+", source, "tcsi300")
    pool_value = str(pool) if pool is not None else _extract_first(
        r"train_maskable_ppo\.py\s+\d+\s+[^\s\\]+\s+(\d+)",
        source,
        "20",
    )
    lambda_ri = _extract_first(r"--lambda_ri\s+([^\s\\]+)", source, "0.3")
    ri_func_weight = _extract_first(r"--ri_func_weight\s+([^\s\\]+)", source, "1.0")
    ri_struct_weight = _extract_first(r"--ri_struct_weight\s+([^\s\\]+)", source, "0.3")
    ri_reg_weight = _extract_first(r"--ri_reg_weight\s+([^\s\\]+)", source, "0.3")
    ri_schedule_decay = _extract_first(r"--ri_schedule_decay\s+([^\s\\]+)", source, "1e-4")
    ri_struct_value_bonus = _extract_first(r"--ri_struct_value_bonus\s+([^\s\\]+)", source, "0.1")
    ri_struct_underexplore_power = _extract_first(r"--ri_struct_underexplore_power\s+([^\s\\]+)", source, "1.0")
    ri_func_metric = _extract_first(r"--ri_func_metric\s+([^\s\\]+)", source, "ic")
    ri_func_backend_value = _choose_flag_value(source, "--ri_func_backend", ri_func_backend, "auto")
    ri_struct_backend_value = _choose_flag_value(source, "--ri_struct_backend", ri_struct_backend, "token_bigram")
    ri_corr_threshold_value = _choose_flag_value(source, "--ri_corr_threshold", ri_corr_threshold, "0.8")
    ri_ast_similarity_threshold_value = _choose_flag_value(
        source,
        "--ri_ast_similarity_threshold",
        ri_ast_similarity_threshold,
        "0.9",
    )
    ri_corr_value_bonus_value = _choose_flag_value(source, "--ri_corr_value_bonus", ri_corr_value_bonus, "0.1")
    ri_corr_underexplore_power_value = _choose_flag_value(
        source,
        "--ri_corr_underexplore_power",
        ri_corr_underexplore_power,
        "1.0",
    )
    ri_turnover_weight_value = _choose_flag_value(source, "--ri_turnover_weight", ri_turnover_weight, "0.0")
    ri_turnover_topk_value = _choose_flag_value(source, "--ri_turnover_topk", ri_turnover_topk, "30")
    ri_turnover_baseline_value = _choose_flag_value(source, "--ri_turnover_baseline", ri_turnover_baseline, "0.5")
    ri_turnover_step_stride_value = _choose_flag_value(
        source,
        "--ri_turnover_step_stride",
        ri_turnover_step_stride,
        "5",
    )
    optimize_every = _extract_first(r"--optimize_every\s+([^\s\\]+)", source, "2")
    optimize_n_iter = _extract_first(r"--optimize_n_iter\s+([^\s\\]+)", source, "256")
    logdir = _extract_first(r"--logdir\s+([^\s\\]+)", source, "/kaggle/working/runs")
    ckpt_dir = _extract_first(r"--ckpt_dir\s+([^\s\\]+)", source, "/kaggle/working/checkpoints")
    tb_dir = _extract_first(r"--tb_dir\s+([^\s\\]+)", source, "/kaggle/working/tb_log")
    provider_uri = _extract_first(r"--provider_uri\s+([^\s\\]+)", source, "")

    lines = [
        f"!python train_maskable_ppo.py {seed} {market} {pool_value} --step {step} \\",
        f"  --backbone {backbone} \\",
        f"  --reward_mode {reward_mode} \\",
        f"  --lambda_ri {lambda_ri} \\",
        f"  --ri_func_weight {ri_func_weight} \\",
        f"  --ri_struct_weight {ri_struct_weight} \\",
        f"  --ri_reg_weight {ri_reg_weight} \\",
        f"  --ri_schedule_decay {ri_schedule_decay} \\",
        f"  --ri_struct_value_bonus {ri_struct_value_bonus} \\",
        f"  --ri_struct_underexplore_power {ri_struct_underexplore_power} \\",
        f"  --ri_func_metric {ri_func_metric} \\",
        f"  --ri_func_backend {ri_func_backend_value} \\",
        f"  --ri_struct_backend {ri_struct_backend_value} \\",
        f"  --ri_corr_threshold {ri_corr_threshold_value} \\",
        f"  --ri_ast_similarity_threshold {ri_ast_similarity_threshold_value} \\",
        f"  --ri_corr_value_bonus {ri_corr_value_bonus_value} \\",
        f"  --ri_corr_underexplore_power {ri_corr_underexplore_power_value} \\",
        f"  --ri_turnover_weight {ri_turnover_weight_value} \\",
        f"  --ri_turnover_topk {ri_turnover_topk_value} \\",
        f"  --ri_turnover_baseline {ri_turnover_baseline_value} \\",
        f"  --ri_turnover_step_stride {ri_turnover_step_stride_value} \\",
        "  --profile_timing \\",
        f"  --optimize_every {optimize_every} \\",
        f"  --optimize_n_iter {optimize_n_iter} \\",
        f"  --train-start-time {train_start_time or '2014-01-01'} \\",
        f"  --train-end-time {train_end_time or '2018-12-31'} \\",
        f"  --valid-start-time {valid_start_time or '2019-01-01'} \\",
        f"  --valid-end-time {valid_end_time or '2019-12-31'} \\",
        f"  --test-start-time {test_start_time or '2019-01-01'} \\",
        f"  --test-end-time {test_end_time or '2019-12-31'} \\",
        f"  --logdir {logdir} \\",
        f"  --ckpt_dir {ckpt_dir} \\",
        f"  --tb_dir {tb_dir} \\",
        f"  --run_name {run_name}",
    ]
    if provider_uri:
        lines[-1] = f"{lines[-1]} \\"
        lines.append(f"  --provider_uri {provider_uri}")
    return "\n".join(lines) + "\n"


def _update_train_cell_source(
    source: str,
    seed: int,
    market_override: Optional[str],
    backbone: str,
    reward_mode: str,
    step: int,
    pool: Optional[int],
    run_name: str,
    ri_func_backend: Optional[str] = None,
    ri_struct_backend: Optional[str] = None,
    ri_corr_threshold: Optional[str] = None,
    ri_ast_similarity_threshold: Optional[str] = None,
    ri_corr_value_bonus: Optional[str] = None,
    ri_corr_underexplore_power: Optional[str] = None,
    ri_turnover_weight: Optional[str] = None,
    ri_turnover_topk: Optional[str] = None,
    ri_turnover_baseline: Optional[str] = None,
    ri_turnover_step_stride: Optional[str] = None,
    train_start_time: Optional[str] = None,
    train_end_time: Optional[str] = None,
    valid_start_time: Optional[str] = None,
    valid_end_time: Optional[str] = None,
    test_start_time: Optional[str] = None,
    test_end_time: Optional[str] = None,
) -> str:
    if "!python train_maskable_ppo.py" not in source:
        return source
    return _render_train_cell(
        source=source,
        seed=seed,
        market_override=market_override,
        backbone=backbone,
        reward_mode=reward_mode,
        step=step,
        pool=pool,
        run_name=run_name,
        ri_func_backend=ri_func_backend,
        ri_struct_backend=ri_struct_backend,
        ri_corr_threshold=ri_corr_threshold,
        ri_ast_similarity_threshold=ri_ast_similarity_threshold,
        ri_corr_value_bonus=ri_corr_value_bonus,
        ri_corr_underexplore_power=ri_corr_underexplore_power,
        ri_turnover_weight=ri_turnover_weight,
        ri_turnover_topk=ri_turnover_topk,
        ri_turnover_baseline=ri_turnover_baseline,
        ri_turnover_step_stride=ri_turnover_step_stride,
        train_start_time=train_start_time,
        train_end_time=train_end_time,
        valid_start_time=valid_start_time,
        valid_end_time=valid_end_time,
        test_start_time=test_start_time,
        test_end_time=test_end_time,
    )


def _rewrite_notebook(
    notebook_path: Path,
    seed: int,
    market_override: Optional[str],
    backbone: str,
    reward_mode: str,
    step: int,
    pool: Optional[int],
    run_name: str,
    ri_func_backend: Optional[str] = None,
    ri_struct_backend: Optional[str] = None,
    ri_corr_threshold: Optional[str] = None,
    ri_ast_similarity_threshold: Optional[str] = None,
    ri_corr_value_bonus: Optional[str] = None,
    ri_corr_underexplore_power: Optional[str] = None,
    ri_turnover_weight: Optional[str] = None,
    ri_turnover_topk: Optional[str] = None,
    ri_turnover_baseline: Optional[str] = None,
    ri_turnover_step_stride: Optional[str] = None,
    train_start_time: Optional[str] = None,
    train_end_time: Optional[str] = None,
    valid_start_time: Optional[str] = None,
    valid_end_time: Optional[str] = None,
    test_start_time: Optional[str] = None,
    test_end_time: Optional[str] = None,
    inject_local_reward_patch: bool = False,
) -> None:
    payload = json.loads(notebook_path.read_text(encoding="utf-8"))
    changed = False
    if inject_local_reward_patch:
        _ensure_local_reward_patch_cell(payload)
    else:
        _remove_local_reward_patch_cell(payload)
    for cell in payload.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if "!python train_maskable_ppo.py" not in src:
            continue
        new_src = _update_train_cell_source(
            src,
            seed,
            market_override,
            backbone,
            reward_mode,
            step,
            pool,
            run_name,
            ri_func_backend=ri_func_backend,
            ri_struct_backend=ri_struct_backend,
            ri_corr_threshold=ri_corr_threshold,
            ri_ast_similarity_threshold=ri_ast_similarity_threshold,
            ri_corr_value_bonus=ri_corr_value_bonus,
            ri_corr_underexplore_power=ri_corr_underexplore_power,
            ri_turnover_weight=ri_turnover_weight,
            ri_turnover_topk=ri_turnover_topk,
            ri_turnover_baseline=ri_turnover_baseline,
            ri_turnover_step_stride=ri_turnover_step_stride,
            train_start_time=train_start_time,
            train_end_time=train_end_time,
            valid_start_time=valid_start_time,
            valid_end_time=valid_end_time,
            test_start_time=test_start_time,
            test_end_time=test_end_time,
        )
        if new_src != src:
            cell["source"] = new_src.splitlines(keepends=True)
            changed = True
        break
    if not changed:
        raise RuntimeError(f"no train cell updated in notebook: {notebook_path}")
    notebook_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")


def _ensure_local_reward_patch_cell(payload: Dict) -> None:
    cells = payload.setdefault("cells", [])
    patch_source = LOCAL_REWARD_PATCH_SOURCE.splitlines(keepends=True)
    for cell in cells:
        if cell.get("cell_type") == "code" and LOCAL_REWARD_PATCH_MARKER in "".join(cell.get("source", [])):
            cell["source"] = patch_source
            return

    insert_at = 1
    for idx, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if "git checkout exp/b_sweep_v1" in source:
            insert_at = idx + 1
            break
    cells.insert(
        insert_at,
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": patch_source,
        },
    )


def _remove_local_reward_patch_cell(payload: Dict) -> None:
    cells = payload.setdefault("cells", [])
    payload["cells"] = [
        cell
        for cell in cells
        if not (
            cell.get("cell_type") == "code"
            and LOCAL_REWARD_PATCH_MARKER in "".join(cell.get("source", []))
        )
    ]


@dataclass
class Job:
    job_id: str
    seed: int
    backbone: str
    reward_mode: str
    status: str = "pending"
    kernel_dir: str = ""
    pushed_at: str = ""
    output: str = ""
    error: str = ""


def _build_jobs(seeds: List[int], backbones: List[str], reward_modes: List[str]) -> List[Job]:
    jobs: List[Job] = []
    for seed in seeds:
        for backbone in backbones:
            for reward_mode in reward_modes:
                jobs.append(
                    Job(
                        job_id=f"seed{seed}_{backbone}_{reward_mode}".replace("+", "p"),
                        seed=seed,
                        backbone=backbone,
                        reward_mode=reward_mode,
                    )
                )
    return jobs


def _load_or_init_state(state_path: Path, jobs: List[Job], force_reset: bool) -> Dict:
    if state_path.exists() and not force_reset:
        return json.loads(state_path.read_text(encoding="utf-8"))
    return {
        "created_at": _now(),
        "updated_at": _now(),
        "jobs": [asdict(j) for j in jobs],
        "submit_history": [],
    }


def _save_state(state_path: Path, state: Dict) -> None:
    state["updated_at"] = _now()
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _pending_indices(state: Dict) -> List[int]:
    return [i for i, j in enumerate(state["jobs"]) if j["status"] == "pending"]


def _runnable_indices(state: Dict, retry_failed: bool) -> List[int]:
    if retry_failed:
        return [i for i, j in enumerate(state["jobs"]) if j["status"] in {"pending", "failed"}]
    return _pending_indices(state)


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit Kaggle training sweep in timed batches.")
    parser.add_argument("--kernel-dirs", type=str, required=True,
                        help="comma-separated local kernel dirs (each contains kernel-metadata.json and ipynb)")
    parser.add_argument("--notebook-file", type=str, default="",
                        help="notebook filename inside kernel dir; empty means auto-detect first *.ipynb")
    parser.add_argument("--seeds", type=str, default="0,1")
    parser.add_argument("--market", type=str, default="")
    parser.add_argument("--backbones", type=str, default="lstm,transformer")
    parser.add_argument("--reward-modes", type=str, default="",
                        help="explicit csv list, e.g. re,re+all,re+func_v2,re+struct_v2+reg_v2,re+all_v2")
    parser.add_argument("--reward-families", type=str, default="re",
                        help="used only when --reward-modes is empty")
    parser.add_argument("--reward-suffixes", type=str, default="base,func,struct,reg,all,func_v2,struct_v2,reg_v2,struct_v2_reg_v2,all_v2",
                        help="used only when --reward-modes is empty")
    parser.add_argument("--pool", type=int, default=None,
                        help="override train_maskable_ppo pool capacity; empty keeps the notebook value")
    parser.add_argument("--step", type=int, default=128000)
    parser.add_argument("--ri-func-backend", type=str, default="",
                        choices=["", "auto", "mutual_ic", "residual", "corr_cluster"])
    parser.add_argument("--ri-struct-backend", type=str, default="",
                        choices=["", "token_bigram", "ast_cluster"])
    parser.add_argument("--ri-corr-threshold", type=str, default="")
    parser.add_argument("--ri-ast-similarity-threshold", type=str, default="")
    parser.add_argument("--ri-corr-value-bonus", type=str, default="")
    parser.add_argument("--ri-corr-underexplore-power", type=str, default="")
    parser.add_argument("--ri-turnover-weight", type=str, default="")
    parser.add_argument("--ri-turnover-topk", type=str, default="")
    parser.add_argument("--ri-turnover-baseline", type=str, default="")
    parser.add_argument("--ri-turnover-step-stride", type=str, default="")
    parser.add_argument("--train-start-time", "--train-start", dest="train_start_time", type=str, default="2014-01-01")
    parser.add_argument("--train-end-time", "--train-end", dest="train_end_time", type=str, default="2018-12-31")
    parser.add_argument("--valid-start-time", "--valid-start", dest="valid_start_time", type=str, default="2019-01-01")
    parser.add_argument("--valid-end-time", "--valid-end", dest="valid_end_time", type=str, default="2019-12-31")
    parser.add_argument("--test-start-time", "--test-start", dest="test_start_time", type=str, default="2019-01-01")
    parser.add_argument("--test-end-time", "--test-end", dest="test_end_time", type=str, default="2019-12-31")
    parser.add_argument("--submit-batch-size", type=int, default=2)
    parser.add_argument("--interval-minutes", type=float, default=24.0)
    parser.add_argument("--one-batch", action="store_true",
                        help="submit only one batch and exit, useful when Kaggle queue capacity is limited")
    parser.add_argument("--state-path", type=str, default="platform_v2/runtime/kaggle_submit_state.json")
    parser.add_argument("--force-reset", action="store_true")
    parser.add_argument("--retry-failed", action="store_true",
                        help="include failed jobs in runnable queue when resuming")
    parser.add_argument("--max-push-retries", type=int, default=3)
    parser.add_argument("--push-retry-backoff-sec", type=float, default=20.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--inject-local-reward-patch", action="store_true",
                        help="insert a notebook cell that patches cloned GitHub code for local reward changes")
    args = parser.parse_args()

    kernel_dirs = [Path(x).resolve() for x in _split_csv(args.kernel_dirs)]
    if not kernel_dirs:
        raise ValueError("at least one kernel dir is required")
    for kd in kernel_dirs:
        if not (kd / "kernel-metadata.json").exists():
            raise FileNotFoundError(f"kernel-metadata.json not found: {kd}")

    seeds = [int(x) for x in _split_csv(args.seeds)]
    backbones = _split_csv(args.backbones)
    reward_modes = _expand_reward_modes(args.reward_modes, args.reward_families, args.reward_suffixes)
    jobs = _build_jobs(seeds, backbones, reward_modes)

    state_path = Path(args.state_path).resolve()
    state = _load_or_init_state(state_path, jobs, args.force_reset)
    _save_state(state_path, state)

    print(f"[init] total_jobs={len(state['jobs'])} pending={len(_pending_indices(state))}")
    if args.retry_failed:
        failed_count = sum(1 for j in state["jobs"] if j["status"] == "failed")
        print(f"[init] retry_failed=True failed={failed_count}")
    print(f"[init] interval={args.interval_minutes}min batch_size={args.submit_batch_size}")
    print(f"[init] state_file={state_path}")

    batch_round = 0
    try:
        while True:
            runnable = _runnable_indices(state, args.retry_failed)
            if not runnable:
                print("[done] all jobs submitted.")
                break
            batch_round += 1
            batch = runnable[: args.submit_batch_size]
            print(f"[round {batch_round}] submitting {len(batch)} job(s), remaining_after={len(runnable) - len(batch)}")
            for order, idx in enumerate(batch):
                job = state["jobs"][idx]
                kernel_dir = kernel_dirs[(batch_round + order - 1) % len(kernel_dirs)]
                notebook_path = (
                    (kernel_dir / args.notebook_file).resolve()
                    if args.notebook_file
                    else _detect_notebook_file(kernel_dir)
                )
                pool_tag = f"p{args.pool}_" if args.pool is not None else ""
                run_name = f"b_{pool_tag}seed{job['seed']}_{job['backbone']}_{job['reward_mode']}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                print(f"  [job] {job['job_id']} -> kernel_dir={kernel_dir.name}")
                try:
                    _rewrite_notebook(
                        notebook_path=notebook_path,
                        seed=int(job["seed"]),
                        market_override=(args.market or None),
                        backbone=str(job["backbone"]),
                        reward_mode=str(job["reward_mode"]),
                        step=int(args.step),
                        pool=args.pool,
                        run_name=run_name,
                        ri_func_backend=(args.ri_func_backend or None),
                        ri_struct_backend=(args.ri_struct_backend or None),
                        ri_corr_threshold=(args.ri_corr_threshold or None),
                        ri_ast_similarity_threshold=(args.ri_ast_similarity_threshold or None),
                        ri_corr_value_bonus=(args.ri_corr_value_bonus or None),
                        ri_corr_underexplore_power=(args.ri_corr_underexplore_power or None),
                        ri_turnover_weight=(args.ri_turnover_weight or None),
                        ri_turnover_topk=(args.ri_turnover_topk or None),
                        ri_turnover_baseline=(args.ri_turnover_baseline or None),
                        ri_turnover_step_stride=(args.ri_turnover_step_stride or None),
                        train_start_time=args.train_start_time,
                        train_end_time=args.train_end_time,
                        valid_start_time=args.valid_start_time,
                        valid_end_time=args.valid_end_time,
                        test_start_time=args.test_start_time,
                        test_end_time=args.test_end_time,
                        inject_local_reward_patch=bool(args.inject_local_reward_patch),
                    )
                    if args.dry_run:
                        output = "[dry-run] skipped kaggle kernels push"
                    else:
                        output = _run_with_retries(
                            ["kaggle", "kernels", "push", "-p", str(kernel_dir)],
                            retries=max(1, int(args.max_push_retries)),
                            backoff_sec=max(1.0, float(args.push_retry_backoff_sec)),
                        )
                    job["status"] = "submitted"
                    job["kernel_dir"] = str(kernel_dir)
                    job["pushed_at"] = _now()
                    job["output"] = output[-2000:]
                    state["submit_history"].append(
                        {
                            "job_id": job["job_id"],
                            "kernel_dir": str(kernel_dir),
                            "submitted_at": job["pushed_at"],
                        }
                    )
                    print(f"    [ok] {job['job_id']}")
                except Exception as e:
                    job["status"] = "failed"
                    job["error"] = str(e)
                    print(f"    [failed] {job['job_id']} -> {e}")
                _save_state(state_path, state)

            if args.one_batch:
                remaining = len(_runnable_indices(state, args.retry_failed))
                print(f"[stop] one_batch=True, remaining={remaining}, state saved: {state_path}")
                break

            if _runnable_indices(state, args.retry_failed):
                sleep_sec = max(1.0, args.interval_minutes * 60.0)
                print(f"[round {batch_round}] sleeping {sleep_sec:.0f}s ... (Ctrl+C to stop, state auto-saved)")
                time.sleep(sleep_sec)
    except KeyboardInterrupt:
        _save_state(state_path, state)
        print(f"\n[stop] interrupted by user, state saved: {state_path}")
        sys.exit(130)


if __name__ == "__main__":
    main()
