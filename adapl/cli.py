"""Command-line interface for AdapL experiments."""

from __future__ import annotations

import argparse
from typing import Optional, Sequence

from adapl.constants import (
    CIFAR100_MAIN_BATCH_SIZE,
    CIFAR100_MAIN_CLIENT_FRACTION,
    CIFAR100_MAIN_DIRICHLET_ALPHA,
    CIFAR100_MAIN_GLOBAL_ROUNDS,
    CIFAR100_MAIN_LOCAL_STEPS,
    CIFAR100_MAIN_LR,
    CIFAR100_MAIN_MOMENTUM,
    CIFAR100_MAIN_NUM_CLIENTS,
    CIFAR100_MAIN_WEIGHT_DECAY,
)
from adapl.methods import canonicalize_method, get_method_info, method_choices
from adapl.paths import DEFAULT_CIFAR100_DIR, DEFAULT_RESULTS_DIR, resolve_project_path


def _default_output_csv(args: argparse.Namespace) -> str:
    partition_name = "noniid" if args.partition in {"dirichlet", "non-iid"} else "iid"
    local_tag = (
        f"steps{args.local_steps}"
        if args.local_update_mode == "random-batch"
        else f"epochs{args.local_epochs}"
    )
    privacy_tag = ""
    if args.method != "pf":
        if args.epsilon_min is not None:
            budget_tag = f"epsmin{args.epsilon_min}"
        elif args.noise_multiplier is not None:
            budget_tag = f"nm{args.noise_multiplier}"
        else:
            budget_tag = "dp"
        privacy_tag = (
            f"_{budget_tag}_delta{args.delta}_clip{args.clipping_norm}"
        )
    filename = (
        f"{args.method}_{args.dataset}_{args.model}_{partition_name}_"
        f"alpha{args.dirichlet_alpha}_k{args.num_clients}_sr{args.client_fraction}_"
        f"{local_tag}_b{args.batch_size}_lr{args.lr}{privacy_tag}_"
        f"r{args.global_rounds}.csv"
    )
    return str(DEFAULT_RESULTS_DIR / filename)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CIFAR-100 federated learning experiments for AdapL."
    )
    parser.add_argument("--dataset", default="cifar100", choices=["cifar100"])
    parser.add_argument("--model", default="resnet18", choices=["resnet18"])
    parser.add_argument(
        "--method",
        default="pf",
        type=str.lower,
        choices=method_choices(),
    )

    privacy_group = parser.add_mutually_exclusive_group()
    privacy_group.add_argument(
        "--no_dp",
        dest="no_dp",
        action="store_true",
        default=None,
        help="Disable DP. Required for PF / PrivacyFree.",
    )
    privacy_group.add_argument(
        "--use_dp",
        dest="no_dp",
        action="store_false",
        help="Enable DP-oriented methods once implemented.",
    )

    parser.add_argument("--epsilon_min", type=float, default=None)
    parser.add_argument("--epsilon_max", type=float, default=None)
    parser.add_argument("--delta", type=float, default=None)
    parser.add_argument("--clipping_norm", type=float, default=None)
    parser.add_argument("--noise_multiplier", type=float, default=None)
    parser.add_argument(
        "--privacy_budgets",
        default=None,
        help="Optional path or comma-separated client privacy budgets for DP methods.",
    )

    parser.add_argument(
        "--data_dir",
        default=str(DEFAULT_CIFAR100_DIR),
        help=(
            "Dataset cache root. Default: /root/autodl-tmp/data. "
            "torchvision stores CIFAR-100 files under this directory."
        ),
    )
    parser.add_argument(
        "--num_clients",
        type=int,
        default=CIFAR100_MAIN_NUM_CLIENTS,
        help=(
            "Total FL clients. Paper reports K in {20, 30, 40, 50}; "
            "default is K=20."
        ),
    )
    parser.add_argument(
        "--client_fraction",
        type=float,
        default=CIFAR100_MAIN_CLIENT_FRACTION,
        help="Client sample rate per round. Paper default is 0.8.",
    )
    parser.add_argument(
        "--partition",
        default="dirichlet",
        choices=["iid", "non-iid", "dirichlet"],
        help="Client data partition strategy.",
    )
    parser.add_argument(
        "--dirichlet_alpha",
        type=float,
        default=CIFAR100_MAIN_DIRICHLET_ALPHA,
        help=(
            "Dirichlet scaling/concentration parameter. CIFAR-100 main "
            "baseline default is 0.5; use 0.3 for the harder paper-style "
            "non-IID setting."
        ),
    )
    parser.add_argument(
        "--global_rounds",
        type=int,
        default=CIFAR100_MAIN_GLOBAL_ROUNDS,
        help="Global communication rounds. CIFAR-100 main baseline default is 300.",
    )
    parser.add_argument(
        "--local_steps",
        type=int,
        default=CIFAR100_MAIN_LOCAL_STEPS,
        help=(
            "Number of random mini-batch SGD steps per selected client per "
            "communication round. CIFAR-100 main baseline default is 20."
        ),
    )
    parser.add_argument(
        "--local_epochs",
        type=int,
        default=None,
        help=(
            "Backward-compatible alias for --local_steps in random-batch mode. "
            "In full-epoch mode, this is the number of full local epochs."
        ),
    )
    parser.add_argument(
        "--local_update_mode",
        default="random-batch",
        choices=["random-batch", "full-epoch"],
        help=(
            "random-batch performs --local_steps shuffled mini-batch updates "
            "per selected client per communication round. "
            "full-epoch iterates through the selected client's full DataLoader."
        ),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=CIFAR100_MAIN_BATCH_SIZE,
        help="Local random batch size. Default is 64 for CIFAR-100 stability.",
    )
    parser.add_argument("--test_batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=CIFAR100_MAIN_LR)
    parser.add_argument("--momentum", type=float, default=CIFAR100_MAIN_MOMENTUM)
    parser.add_argument("--weight_decay", type=float, default=CIFAR100_MAIN_WEIGHT_DECAY)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument(
        "--limit_train_samples",
        type=int,
        default=None,
        help="Optional smoke-test limit before partitioning.",
    )
    parser.add_argument(
        "--limit_test_samples",
        type=int,
        default=None,
        help="Optional smoke-test limit for evaluation.",
    )
    parser.add_argument(
        "--output_csv",
        default=None,
        help="Where to save per-round train loss and test accuracy.",
    )
    parser.add_argument(
        "--run_config_json",
        default=None,
        help="Optional path for saving run configuration as JSON.",
    )
    parser.add_argument(
        "--run_config_csv",
        default=None,
        help="Where to save run configuration and final summary as CSV.",
    )
    parser.add_argument(
        "--client_distribution_csv",
        default=None,
        help="Where to save per-client CIFAR-100 label counts as CSV.",
    )
    parser.add_argument(
        "--client_distribution_json",
        default=None,
        help="Optional path for saving per-client CIFAR-100 label counts as JSON.",
    )
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.method = canonicalize_method(args.method)

    if args.no_dp is None:
        args.no_dp = args.method == "pf"
    if args.method != "pf":
        if args.delta is None:
            args.delta = 1e-5
        if args.clipping_norm is None:
            args.clipping_norm = 1.0

    if args.local_update_mode == "random-batch" and args.local_epochs is not None:
        args.local_steps = args.local_epochs
    if args.local_update_mode == "full-epoch" and args.local_epochs is None:
        args.local_epochs = args.local_steps
    if args.output_csv is None:
        args.output_csv = _default_output_csv(args)
    args.data_dir = str(resolve_project_path(args.data_dir))
    args.output_csv = str(resolve_project_path(args.output_csv))
    if args.run_config_json:
        args.run_config_json = str(resolve_project_path(args.run_config_json))
    if args.run_config_csv:
        args.run_config_csv = str(resolve_project_path(args.run_config_csv))
    if args.client_distribution_csv:
        args.client_distribution_csv = str(
            resolve_project_path(args.client_distribution_csv)
        )
    if args.client_distribution_json:
        args.client_distribution_json = str(
            resolve_project_path(args.client_distribution_json)
        )
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    method_info = get_method_info(args.method)
    if not method_info.implemented:
        raise SystemExit(
            f"{method_info.display_name} is registered but not implemented yet. "
            "Implement it under adapl/methods/ before running this method."
        )

    try:
        from adapl.experiment import run_experiment
    except ModuleNotFoundError as exc:
        if exc.name in {"numpy", "torch", "torchvision"}:
            raise SystemExit(
                f"Missing dependency: {exc.name}. Install dependencies with "
                "`pip install -r requirements.txt`."
            ) from exc
        raise

    run_experiment(args)
