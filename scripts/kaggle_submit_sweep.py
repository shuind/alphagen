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
    }
    families = _split_csv(reward_families)
    suffixes = _split_csv(reward_suffixes)
    expanded: List[str] = []
    for fam in families:
        if fam not in {"re", "re_v2"}:
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


def _update_train_cell_source(
    source: str,
    seed: int,
    backbone: str,
    reward_mode: str,
    step: int,
    run_name: str,
) -> str:
    if "train_maskable_ppo.py" not in source:
        return source
    source = re.sub(
        r"(train_maskable_ppo\.py\s+)(\d+)",
        rf"\g<1>{seed}",
        source,
        count=1,
    )
    source = _replace_or_insert_flag(source, "--backbone", backbone)
    source = _replace_or_insert_flag(source, "--reward_mode", reward_mode)
    source = _replace_or_insert_flag(source, "--step", str(step))
    source = _replace_or_insert_flag(source, "--run_name", run_name)
    source = re.sub(r"--save_model_ckpt(\s+|\\\n)?", "", source)
    return source


def _rewrite_notebook(
    notebook_path: Path,
    seed: int,
    backbone: str,
    reward_mode: str,
    step: int,
    run_name: str,
) -> None:
    payload = json.loads(notebook_path.read_text(encoding="utf-8"))
    changed = False
    for cell in payload.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if "train_maskable_ppo.py" not in src:
            continue
        new_src = _update_train_cell_source(src, seed, backbone, reward_mode, step, run_name)
        if new_src != src:
            cell["source"] = new_src.splitlines(keepends=True)
            changed = True
        break
    if not changed:
        raise RuntimeError(f"no train cell updated in notebook: {notebook_path}")
    notebook_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit Kaggle training sweep in timed batches.")
    parser.add_argument("--kernel-dirs", type=str, required=True,
                        help="comma-separated local kernel dirs (each contains kernel-metadata.json and ipynb)")
    parser.add_argument("--notebook-file", type=str, default="",
                        help="notebook filename inside kernel dir; empty means auto-detect first *.ipynb")
    parser.add_argument("--seeds", type=str, default="0,1")
    parser.add_argument("--backbones", type=str, default="lstm,transformer")
    parser.add_argument("--reward-modes", type=str, default="",
                        help="explicit csv list, e.g. re,re+all,re_v2,re_v2+all")
    parser.add_argument("--reward-families", type=str, default="re,re_v2",
                        help="used only when --reward-modes is empty")
    parser.add_argument("--reward-suffixes", type=str, default="base,func,struct,reg,all",
                        help="used only when --reward-modes is empty")
    parser.add_argument("--step", type=int, default=64000)
    parser.add_argument("--submit-batch-size", type=int, default=2)
    parser.add_argument("--interval-minutes", type=float, default=24.0)
    parser.add_argument("--state-path", type=str, default="platform_v2/runtime/kaggle_submit_state.json")
    parser.add_argument("--force-reset", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
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
    print(f"[init] interval={args.interval_minutes}min batch_size={args.submit_batch_size}")
    print(f"[init] state_file={state_path}")

    batch_round = 0
    try:
        while True:
            pending = _pending_indices(state)
            if not pending:
                print("[done] all jobs submitted.")
                break
            batch_round += 1
            batch = pending[: args.submit_batch_size]
            print(f"[round {batch_round}] submitting {len(batch)} job(s), remaining_after={len(pending) - len(batch)}")
            for order, idx in enumerate(batch):
                job = state["jobs"][idx]
                kernel_dir = kernel_dirs[(batch_round + order - 1) % len(kernel_dirs)]
                notebook_path = (
                    (kernel_dir / args.notebook_file).resolve()
                    if args.notebook_file
                    else _detect_notebook_file(kernel_dir)
                )
                run_name = f"b_seed{job['seed']}_{job['backbone']}_{job['reward_mode']}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                print(f"  [job] {job['job_id']} -> kernel_dir={kernel_dir.name}")
                try:
                    _rewrite_notebook(
                        notebook_path=notebook_path,
                        seed=int(job["seed"]),
                        backbone=str(job["backbone"]),
                        reward_mode=str(job["reward_mode"]),
                        step=int(args.step),
                        run_name=run_name,
                    )
                    if args.dry_run:
                        output = "[dry-run] skipped kaggle kernels push"
                    else:
                        output = _run(["kaggle", "kernels", "push", "-p", str(kernel_dir)])
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

            if _pending_indices(state):
                sleep_sec = max(1.0, args.interval_minutes * 60.0)
                print(f"[round {batch_round}] sleeping {sleep_sec:.0f}s ... (Ctrl+C to stop, state auto-saved)")
                time.sleep(sleep_sec)
    except KeyboardInterrupt:
        _save_state(state_path, state)
        print(f"\n[stop] interrupted by user, state saved: {state_path}")
        sys.exit(130)


if __name__ == "__main__":
    main()
