import argparse
import json
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


NEW_PACKAGE_MARKER = "# codex-local-new-package-20260506"


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _split_csv(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def _run(cmd: List[str], cwd: Optional[Path] = None) -> str:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stdout}")
    return proc.stdout


def _run_with_retries(cmd: List[str], retries: int, backoff_sec: float, cwd: Optional[Path] = None) -> str:
    last_exc: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            if attempt > 1:
                print(f"    [retry] attempt={attempt}/{retries}")
            return _run(cmd, cwd=cwd)
        except Exception as exc:
            last_exc = exc
            if attempt >= retries:
                break
            sleep_sec = backoff_sec * attempt
            print(f"    [retry] failed: {exc}")
            print(f"    [retry] sleeping {sleep_sec:.1f}s")
            time.sleep(sleep_sec)
    assert last_exc is not None
    raise last_exc


def _detect_notebook_file(kernel_dir: Path) -> Path:
    notebooks = sorted(kernel_dir.glob("*.ipynb"))
    if not notebooks:
        raise FileNotFoundError(f"no ipynb found in {kernel_dir}")
    return notebooks[0]


def _load_new_package_files(new_dir: Path) -> Dict[str, str]:
    if not new_dir.exists():
        raise FileNotFoundError(f"new package dir not found: {new_dir}")
    files: Dict[str, str] = {}
    for path in sorted(new_dir.rglob("*.py")) + sorted(new_dir.rglob("*.md")):
        rel = path.relative_to(new_dir.parent).as_posix()
        files[rel] = path.read_text(encoding="utf-8")
    if "new/train_multihead_ppo.py" not in files:
        raise FileNotFoundError(f"new/train_multihead_ppo.py not found under {new_dir}")
    return files


def _render_new_package_cell(files: Dict[str, str]) -> str:
    files_json = json.dumps(files, ensure_ascii=False, indent=2)
    compile_targets = " ".join(path for path in files if path.endswith(".py"))
    return f"""{NEW_PACKAGE_MARKER}
from pathlib import Path

files = {files_json}

for rel_path, content in files.items():
    path = Path(rel_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")

print("Injected local new/ multihead package:", len(files), "files")
!python -m py_compile {compile_targets}
"""


def _ensure_new_package_cell(payload: Dict, files: Dict[str, str]) -> None:
    cells = payload.setdefault("cells", [])
    source = _render_new_package_cell(files).splitlines(keepends=True)
    for cell in cells:
        if cell.get("cell_type") == "code" and NEW_PACKAGE_MARKER in "".join(cell.get("source", [])):
            cell["source"] = source
            return

    insert_at = 1
    for idx, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if "git checkout" in src:
            insert_at = idx + 1
            break
    cells.insert(
        insert_at,
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": source,
        },
    )


def _render_train_cell(
    seed: int,
    market: str,
    pool: int,
    step: int,
    method: str,
    run_name: str,
    intrinsic_beta: float,
    simple_bias: float,
    ts_bias: float,
    optimize_every: int,
    optimize_n_iter: int,
) -> str:
    return "\n".join(
        [
            f"!python -m new.train_multihead_ppo {seed} {market} {pool} --step {step} \\",
            f"  --method {method} \\",
            f"  --intrinsic-beta {intrinsic_beta} \\",
            f"  --simple-bias {simple_bias} \\",
            f"  --ts-bias {ts_bias} \\",
            f"  --optimize_every {optimize_every} \\",
            f"  --optimize_n_iter {optimize_n_iter} \\",
            "  --logdir /kaggle/working/runs \\",
            "  --ckpt_dir /kaggle/working/checkpoints \\",
            "  --tb_dir /kaggle/working/tb_log \\",
            f"  --run_name {run_name}",
            "",
        ]
    )


def _rewrite_notebook(
    notebook_path: Path,
    files: Dict[str, str],
    seed: int,
    market: str,
    pool: int,
    step: int,
    method: str,
    run_name: str,
    intrinsic_beta: float,
    simple_bias: float,
    ts_bias: float,
    optimize_every: int,
    optimize_n_iter: int,
) -> None:
    payload = json.loads(notebook_path.read_text(encoding="utf-8"))
    _ensure_new_package_cell(payload, files)
    cells = payload.setdefault("cells", [])
    # Remove stale train cells from previous submitters. Keep the injected package cell even
    # though it contains the literal string new.train_multihead_ppo inside README content.
    payload["cells"] = [
        cell
        for cell in cells
        if not (
            cell.get("cell_type") == "code"
            and NEW_PACKAGE_MARKER not in "".join(cell.get("source", []))
            and (
                "train_maskable_ppo.py" in "".join(cell.get("source", []))
                or "new.train_multihead_ppo" in "".join(cell.get("source", []))
            )
        )
    ]
    train_source = _render_train_cell(
        seed=seed,
        market=market,
        pool=pool,
        step=step,
        method=method,
        run_name=run_name,
        intrinsic_beta=intrinsic_beta,
        simple_bias=simple_bias,
        ts_bias=ts_bias,
        optimize_every=optimize_every,
        optimize_n_iter=optimize_n_iter,
    )
    train_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": train_source.splitlines(keepends=True),
    }
    insert_at = len(payload["cells"])
    for idx, cell in enumerate(payload["cells"]):
        if cell.get("cell_type") == "code" and "zip -r /kaggle/working/alphagen_outputs.zip" in "".join(cell.get("source", [])):
            insert_at = idx
            break
    payload["cells"].insert(insert_at, train_cell)
    notebook_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")


@dataclass
class Job:
    job_id: str
    seed: int
    method: str


def _build_jobs(seeds: List[int], methods: List[str]) -> List[Job]:
    jobs: List[Job] = []
    for seed in seeds:
        for method in methods:
            jobs.append(Job(job_id=f"seed{seed}_{method}", seed=seed, method=method))
    return jobs


def _load_or_init_state(path: Path, jobs: List[Job], force_reset: bool) -> Dict:
    if path.exists() and not force_reset:
        return json.loads(path.read_text(encoding="utf-8"))
    state = {
        "created_at": _now(),
        "updated_at": _now(),
        "jobs": [
            {
                **asdict(job),
                "status": "pending",
                "attempts": 0,
                "submitted_at": None,
                "kernel_dir": None,
                "run_name": None,
                "error": None,
            }
            for job in jobs
        ],
    }
    return state


def _save_state(path: Path, state: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = _now()
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _runnable_indices(state: Dict, retry_failed: bool) -> List[int]:
    statuses = {"pending"}
    if retry_failed:
        statuses.add("failed")
    return [idx for idx, job in enumerate(state["jobs"]) if job["status"] in statuses]


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit clean multi-head AlphaGen Kaggle sweep.")
    parser.add_argument("--kernel-dirs", required=True, help="comma-separated kernel dirs")
    parser.add_argument("--notebook-file", default="", help="notebook filename inside kernel dir")
    parser.add_argument("--new-dir", default="new", help="local new package dir to inject into notebook")
    parser.add_argument("--seeds", default="0,1")
    parser.add_argument(
        "--methods",
        default="single_transformer,multihead,multihead_intrinsic",
        help="csv: single_transformer,multihead,multihead_intrinsic",
    )
    parser.add_argument("--market", default="tcsi300")
    parser.add_argument("--pool", type=int, default=10)
    parser.add_argument("--step", type=int, default=64000)
    parser.add_argument("--intrinsic-beta", type=float, default=0.1)
    parser.add_argument("--simple-bias", type=float, default=0.4)
    parser.add_argument("--ts-bias", type=float, default=0.5)
    parser.add_argument("--optimize-every", type=int, default=2)
    parser.add_argument("--optimize-n-iter", type=int, default=256)
    parser.add_argument("--submit-batch-size", type=int, default=2)
    parser.add_argument("--interval-minutes", type=float, default=24.0)
    parser.add_argument("--one-batch", action="store_true")
    parser.add_argument("--state-path", default="data/kaggle_submit_state_multihead.json")
    parser.add_argument("--force-reset", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--max-push-retries", type=int, default=3)
    parser.add_argument("--push-retry-backoff-sec", type=float, default=20.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    kernel_dirs = [Path(x).resolve() for x in _split_csv(args.kernel_dirs)]
    if not kernel_dirs:
        raise ValueError("at least one kernel dir is required")
    for kd in kernel_dirs:
        if not (kd / "kernel-metadata.json").exists():
            raise FileNotFoundError(f"kernel-metadata.json not found: {kd}")

    files = _load_new_package_files(Path(args.new_dir).resolve())
    seeds = [int(x) for x in _split_csv(args.seeds)]
    methods = _split_csv(args.methods)
    allowed_methods = {"single_transformer", "multihead", "multihead_intrinsic"}
    bad_methods = [m for m in methods if m not in allowed_methods]
    if bad_methods:
        raise ValueError(f"unsupported methods: {bad_methods}")

    jobs = _build_jobs(seeds, methods)
    state_path = Path(args.state_path).resolve()
    state = _load_or_init_state(state_path, jobs, args.force_reset)
    _save_state(state_path, state)

    print(f"[init] total_jobs={len(state['jobs'])} runnable={len(_runnable_indices(state, args.retry_failed))}")
    print(f"[init] methods={methods} seeds={seeds} pool={args.pool} step={args.step}")
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
                run_name = (
                    f"mh_p{args.pool}_seed{job['seed']}_{job['method']}_"
                    f"{datetime.now().strftime('%Y%m%d%H%M%S')}"
                )
                print(f"  [job] {job['job_id']} -> kernel_dir={kernel_dir.name}")
                try:
                    _rewrite_notebook(
                        notebook_path=notebook_path,
                        files=files,
                        seed=int(job["seed"]),
                        market=str(args.market),
                        pool=int(args.pool),
                        step=int(args.step),
                        method=str(job["method"]),
                        run_name=run_name,
                        intrinsic_beta=float(args.intrinsic_beta),
                        simple_bias=float(args.simple_bias),
                        ts_bias=float(args.ts_bias),
                        optimize_every=int(args.optimize_every),
                        optimize_n_iter=int(args.optimize_n_iter),
                    )
                    if args.dry_run:
                        output = "[dry-run] skipped kaggle kernels push"
                    else:
                        output = _run_with_retries(
                            ["kaggle", "kernels", "push", "-p", str(kernel_dir)],
                            retries=max(1, int(args.max_push_retries)),
                            backoff_sec=float(args.push_retry_backoff_sec),
                        )
                    job["status"] = "submitted"
                    job["attempts"] = int(job.get("attempts", 0)) + 1
                    job["submitted_at"] = _now()
                    job["kernel_dir"] = str(kernel_dir)
                    job["run_name"] = run_name
                    job["error"] = None
                    job["last_output"] = output[-4000:]
                    print(f"    [ok] {job['job_id']}")
                except Exception as exc:
                    job["status"] = "failed"
                    job["attempts"] = int(job.get("attempts", 0)) + 1
                    job["error"] = str(exc)
                    print(f"    [failed] {job['job_id']} -> {exc}")
                _save_state(state_path, state)
            if args.one_batch:
                print("[stop] one-batch requested.")
                break
            if _runnable_indices(state, args.retry_failed):
                sleep_sec = max(0.0, float(args.interval_minutes) * 60.0)
                print(f"[round {batch_round}] sleeping {sleep_sec:.0f}s ... (Ctrl+C to stop, state auto-saved)")
                time.sleep(sleep_sec)
    except KeyboardInterrupt:
        _save_state(state_path, state)
        print("\n[stop] interrupted; state saved.")


if __name__ == "__main__":
    main()
