#!/usr/bin/env python3
"""Launch a grid of SLURM eval jobs from a YAML config.

Usage:
    python scripts/cluster_evals/launch_evals.py --config scripts/cluster_evals/configs/example.yaml
    python scripts/cluster_evals/launch_evals.py --config my.yaml --dry-run
"""

from __future__ import annotations

import argparse
import datetime
import subprocess
import sys
from pathlib import Path

import yaml


def get_auto_experiment_name(task_ids: list[int], perturbation_ids: list[int], max_steps: int, repeats: int) -> str:
    t = "_".join(str(i) for i in task_ids)
    p = "_".join(str(i) for i in perturbation_ids)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"t{t}_p{p}_s{max_steps}_r{repeats}_{ts}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch REALM eval jobs from a YAML config.")
    parser.add_argument("--config", required=True, help="Path to the YAML eval config file.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the sbatch commands without submitting them.",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))

    eval_cfg = cfg["eval"]
    slurm = cfg.get("slurm", {})
    base_port: int = cfg.get("base_port", 8000)
    debug: bool = cfg.get("debug", False)

    task_ids = eval_cfg["task_ids"]
    perturbation_ids = eval_cfg["perturbation_ids"]
    repeats: int = eval_cfg["repeats"]
    max_steps: int = eval_cfg["max_steps"]
    experiment_name: str = eval_cfg.get("experiment_name") or get_auto_experiment_name(
        task_ids, perturbation_ids, max_steps, repeats
    )

    config_path = str(Path(args.config).resolve())
    script = Path(__file__).parent / "run_single_eval.sh"
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    total = len(task_ids) * len(perturbation_ids)
    print(f"Experiment : {experiment_name}")
    print(f"Tasks      : {task_ids}")
    print(f"Perturbs   : {perturbation_ids}")
    print(f"Jobs       : {total}")
    print(f"Config     : {config_path}")
    if debug:
        print("Mode       : DEBUG (no policy server)")
    print()

    submitted = 0
    for task_id in task_ids:
        for pert_id in perturbation_ids:
            port = base_port + pert_id + 100 * task_id
            cmd = ["sbatch"]
            if slurm:
                cmd += [
                    f"--job-name=omnigibson-eval-t{task_id}-p{pert_id}",
                    f"--partition={slurm['partition']}",
                    f"--gpus={slurm['gpus']}",
                    f"--mem={slurm['mem']}",
                    "--ntasks-per-node=1",
                    f"--cpus-per-gpu={slurm['cpus_per_gpu']}",
                    f"--time={slurm['time']}",
                ]
            cmd += [
                str(script),
                "--config", config_path,
                "--task_id", str(task_id),
                "--perturbation_id", str(pert_id),
                "--port", str(port),
                "--experiment_name", experiment_name,
                "--run_id", run_id,
            ]
            if debug:
                cmd.append("--debug")

            if args.dry_run:
                print(" ".join(cmd))
            else:
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"ERROR submitting task={task_id} pert={pert_id}: {result.stderr.strip()}", file=sys.stderr)
                else:
                    print(f"Submitted task={task_id} pert={pert_id} port={port}: {result.stdout.strip()}")
                submitted += 1

    if not args.dry_run:
        print(f"\nSubmitted {submitted}/{total} jobs.")


if __name__ == "__main__":
    main()
