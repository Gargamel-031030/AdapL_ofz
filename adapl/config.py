"""Command-line parsing and experiment configuration normalization."""

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
from adapl.methods import canonicalize_method, method_choices
from adapl.paths import DEFAULT_CIFAR100_DIR, DEFAULT_RESULTS_DIR, resolve_project_path


def _parse_int_list(raw_value: str | Sequence[int], option_name: str) -> list[int]:
    if isinstance(raw_value, str):
        if not raw_value.strip():
            return []
        values = [item.strip() for item in raw_value.split(",")]
    else:
        values = list(raw_value)
    try:
        parsed = [int(value) for value in values if str(value).strip()]
    except ValueError as exc:
        raise ValueError(f"{option_name} must be a comma-separated integer list.") from exc
    return parsed


def _parse_float_list(raw_value: str | Sequence[float], option_name: str) -> list[float]:
    if isinstance(raw_value, str):
        if not raw_value.strip():
            return []
        values = [item.strip() for item in raw_value.split(",")]
    else:
        values = list(raw_value)
    try:
        parsed = [float(value) for value in values if str(value).strip()]
    except ValueError as exc:
        raise ValueError(f"{option_name} must be a comma-separated float list.") from exc
    return parsed


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
        elif args.privacy_scenario is not None:
            scenario = args.privacy_scenario.replace("scenario", "").strip("_- ")
            budget_tag = f"scenario{scenario}"
        elif args.privacy_budgets is not None:
            budget_tag = "budgets"
        else:
            budget_tag = "dp"
        privacy_tag = (
            f"_{budget_tag}"
            if args.no_dp
            else f"_dp_{budget_tag}_clip{args.clipping_norm}"
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
        help="Enable DP for methods that support or require it.",
    )

    parser.add_argument("--epsilon_min", type=float, default=None)
    parser.add_argument("--epsilon_max", type=float, default=None)
    parser.add_argument("--delta", "--target_delta", dest="delta", type=float, default=None)
    parser.add_argument(
        "--clipping_norm",
        "--clipping_bound",
        dest="clipping_norm",
        type=float,
        default=None,
    )
    parser.add_argument("--noise_multiplier", type=float, default=None)
    parser.add_argument("--noise_multiplier_override", type=float, default=None)
    parser.add_argument(
        "--privacy_budgets",
        "--epsilon_file",
        dest="privacy_budgets",
        default=None,
        help="Optional path or comma-separated client privacy budgets for DP methods.",
    )
    parser.add_argument(
        "--privacy_scenario",
        default=None,
        choices=["1", "2", "3", "scenario1", "scenario2", "scenario3"],
        type=str.lower,
        help=(
            "Paper privacy-level scenario. Scenario 1: 10/10/40/20/20; "
            "Scenario 2: 20/20/40/10/10; Scenario 3: 90/0/0/0/10 "
            "over levels with budgets 0.5/1/2/4/8."
        ),
    )
    privacy_accounting_group = parser.add_mutually_exclusive_group()
    privacy_accounting_group.add_argument(
        "--privacy_accounting",
        default="auto",
        choices=["auto", "on", "off"],
        help=(
            "Privacy budget consumption accounting mode. auto preserves each "
            "method's default behavior, on forces per-client accounting for "
            "any method using epsilon/privacy budgets, and off disables "
            "accounting while leaving DP noise behavior unchanged."
        ),
    )
    privacy_accounting_group.add_argument(
        "--use_privacy_accounting",
        dest="privacy_accounting",
        action="store_const",
        const="on",
        help="Alias for --privacy_accounting on.",
    )
    privacy_accounting_group.add_argument(
        "--no_privacy_accounting",
        dest="privacy_accounting",
        action="store_const",
        const="off",
        help="Alias for --privacy_accounting off.",
    )
    parser.add_argument(
        "--privacy_budget_seed",
        type=int,
        default=41,
        help="Seed used to assign scenario privacy budgets to clients.",
    )
    parser.add_argument(
        "--budget_growth_factor",
        type=float,
        default=1.0,
        help=(
            "Slowdown factor for per-client privacy budget accumulation. "
            "The RDP-computed epsilon is multiplied by this factor before "
            "comparison against the client's budget limit. Values < 1.0 "
            "make the budget grow slower, allowing more training rounds."
        ),
    )

    parser.add_argument(
        "--fisher_threshold",
        type=float,
        default=0.4,
        help="AdapL Fisher mask threshold: Im_i = 1 if F_i >= threshold.",
    )
    parser.add_argument(
        "--fisher_estimator",
        choices=["sample", "batch"],
        default="batch",
        help="AdapL Fisher diagonal estimator.",
    )
    parser.add_argument(
        "--fisher_batches",
        type=int,
        default=1,
        help="Number of local batches used to estimate AdapL Fisher masks. Use 0 for all batches.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=10.0,
        help="AdapL layer-wise noise multiplier strength.",
    )
    parser.add_argument(
        "--adapl_alpha",
        "--phi",
        dest="adapl_alpha",
        type=float,
        default=0.8,
        help="AdapL aggregation interpolation weight for epsilon-based weights.",
    )
    parser.add_argument(
        "--adapl_noise_decay_factor",
        "--decay_factor",
        dest="adapl_noise_decay_factor",
        type=float,
        default=0.99,
        help="AdapL base noise multiplier decay factor after sustained accuracy gains.",
    )
    parser.add_argument(
        "--adapl_accuracy_window",
        "--adapl_s",
        dest="adapl_accuracy_window",
        type=int,
        default=3,
        help=(
            "Number of previous evaluated accuracies that must form a "
            "non-decreasing run with the current new best before AdapL decays sigma."
        ),
    )
    parser.add_argument(
        "--max_clip_norm",
        type=float,
        default=4.0,
        help="Optional AdapL decay-stage per-layer clipping norm.",
    )
    parser.add_argument(
        "--prox_mu",
        type=float,
        default=0.0,
        help=(
            "Optional FedProx-style local proximal strength for AdapL. "
            "0 disables the proximal term."
        ),
    )
    parser.add_argument(
        "--adapl_disable_noise",
        "--disable_dp_noise",
        dest="adapl_disable_noise",
        action="store_true",
        help=(
            "Diagnostic AdapL ablation: keep the per-sample gradient path but "
            "skip Gaussian noise. The resulting run is not differentially private."
        ),
    )
    parser.add_argument(
        "--adapl_disable_clipping",
        "--disable_clipping",
        dest="adapl_disable_clipping",
        action="store_true",
        help=(
            "Diagnostic AdapL ablation: compute per-sample gradients without "
            "clipping them. The resulting run is not differentially private."
        ),
    )
    parser.add_argument(
        "--adapl_disable_fisher",
        "--disable_fisher",
        dest="adapl_disable_fisher",
        action="store_true",
        help=(
            "Diagnostic AdapL ablation: skip Fisher estimation, use all-true "
            "masks, and use uniform base layer noise multipliers."
        ),
    )
    parser.add_argument(
        "--adapl_noise_scope",
        "--noise_scope",
        dest="adapl_noise_scope",
        choices=["fisher", "all"],
        default="fisher",
        help=(
            "Where AdapL applies Gaussian noise: Fisher-important coordinates "
            "(legacy behavior) or every trainable coordinate."
        ),
    )
    parser.add_argument(
        "--adapl_freeze_bn",
        "--freeze_bn",
        dest="adapl_freeze_bn",
        action="store_true",
        help=(
            "Keep BatchNorm in evaluation mode during AdapL local training so "
            "running statistics are neither updated nor released."
        ),
    )
    nm_decay_group = parser.add_mutually_exclusive_group()
    nm_decay_group.add_argument(
        "--nm_decay",
        dest="nm_decay",
        action="store_true",
        default=False,
        help="Use moments-accountant binary search to initialize noise multipliers.",
    )
    nm_decay_group.add_argument(
        "--no_nm_decay",
        dest="nm_decay",
        action="store_false",
        help="Use the closed-form Gaussian noise multiplier initialization.",
    )

    parser.add_argument(
        "--feddpa_fisher_threshold",
        type=float,
        default=0.4,
        help=(
            "FedDPA Fisher mask threshold in [0, 1]. Parameters with "
            "normalized Fisher scores above this value remain personalized."
        ),
    )
    parser.add_argument(
        "--feddpa_fisher_batches",
        type=int,
        default=1,
        help=(
            "Number of local batches used to estimate FedDPA Fisher masks. "
            "Use 0 to scan the full client loader."
        ),
    )
    parser.add_argument(
        "--feddpa_lambda1",
        type=float,
        default=0.05,
        help="FedDPA regularization strength for personalized parameters.",
    )
    parser.add_argument(
        "--feddpa_lambda2",
        type=float,
        default=0.1,
        help="FedDPA regularization strength for shared parameters.",
    )
    parser.add_argument(
        "--pfa_public_fraction",
        type=float,
        default=0.1,
        help="Fraction of highest-epsilon clients treated as PFA public clients.",
    )
    parser.add_argument(
        "--pfa_projection_dim",
        type=int,
        default=1,
        help="Number of public-update principal directions used by PFA.",
    )
    parser.add_argument(
        "--pfa_weighted_projection",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use epsilon-weighted public/private projected aggregation in PFA.",
    )
    parser.add_argument(
        "--pfa_selection_attempts",
        type=int,
        default=50,
        help="Maximum client-sampling retries to include public and private clients.",
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
        "--user_sample_rate",
        dest="client_fraction",
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
    partition_flags = parser.add_mutually_exclusive_group()
    partition_flags.add_argument(
        "--iid",
        dest="partition_flag",
        action="store_const",
        const="iid",
        default=None,
        help="Alias for --partition iid.",
    )
    partition_flags.add_argument(
        "--no-iid",
        dest="partition_flag",
        action="store_const",
        const="dirichlet",
        help="Alias for --partition dirichlet.",
    )
    parser.add_argument(
        "--dirichlet_alpha",
        "--dir_alpha",
        dest="dirichlet_alpha",
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
        "--global_epoch",
        dest="global_rounds",
        type=int,
        default=CIFAR100_MAIN_GLOBAL_ROUNDS,
        help="Global communication rounds. CIFAR-100 main baseline default is 300.",
    )
    parser.add_argument(
        "--local_steps",
        "--local_epoch",
        dest="local_steps",
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
    parser.add_argument(
        "--lr_schedule",
        choices=["constant", "piecewise"],
        default="constant",
        help=(
            "Global-round learning-rate schedule. constant preserves the "
            "single --lr value. piecewise uses --lr_milestones and --lr_values."
        ),
    )
    parser.add_argument(
        "--lr_milestones",
        default="",
        help=(
            "Comma-separated completed global rounds where the piecewise "
            "learning rate switches, e.g. 20,40 means rounds 1-20 use the "
            "first value, 21-40 the second, and 41+ the third."
        ),
    )
    parser.add_argument(
        "--lr_values",
        default="",
        help=(
            "Comma-separated learning rates for --lr_schedule piecewise. "
            "Length must be len(--lr_milestones)+1, e.g. 0.03,0.01,0.005."
        ),
    )
    parser.add_argument("--momentum", type=float, default=CIFAR100_MAIN_MOMENTUM)
    parser.add_argument("--weight_decay", type=float, default=CIFAR100_MAIN_WEIGHT_DECAY)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=None,
        help=(
            "Optional number of evaluated rounds to wait without a new best "
            "test accuracy before stopping. Disabled by default."
        ),
    )
    parser.add_argument(
        "--early_stop_min_delta",
        type=float,
        default=0.0,
        help="Minimum test-accuracy improvement required to reset early stopping.",
    )
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


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    args.method = canonicalize_method(args.method)

    if args.partition_flag is not None:
        args.partition = args.partition_flag
    delattr(args, "partition_flag")

    if args.noise_multiplier_override is not None:
        if args.noise_multiplier_override <= 0:
            raise ValueError("--noise_multiplier_override must be positive.")
        args.noise_multiplier = args.noise_multiplier_override
    elif args.noise_multiplier is not None and args.noise_multiplier <= 0:
        raise ValueError("--noise_multiplier must be positive.")

    if args.no_dp is None:
        args.no_dp = args.method == "pf"
    if not args.no_dp:
        if args.delta is None:
            args.delta = 1e-5
        if args.clipping_norm is None:
            args.clipping_norm = 1.0

    if args.local_update_mode == "random-batch" and args.local_epochs is not None:
        args.local_steps = args.local_epochs
    if args.local_update_mode == "full-epoch" and args.local_epochs is None:
        args.local_epochs = args.local_steps
    if args.local_update_mode == "random-batch":
        args.local_epochs = None

    if args.fisher_batches < 0:
        raise ValueError("--fisher_batches must be non-negative.")
    if args.fisher_threshold < 0:
        raise ValueError("--fisher_threshold must be non-negative.")
    if args.gamma < 0:
        raise ValueError("--gamma must be non-negative.")
    if not 0 <= args.adapl_alpha <= 1:
        raise ValueError("--adapl_alpha/--phi must be in [0, 1].")
    if args.adapl_noise_decay_factor <= 0:
        raise ValueError("--adapl_noise_decay_factor/--decay_factor must be positive.")
    if args.adapl_accuracy_window <= 0:
        raise ValueError("--adapl_accuracy_window/--adapl_s must be positive.")
    if args.max_clip_norm is not None and args.max_clip_norm <= 0:
        raise ValueError("--max_clip_norm must be positive.")
    if args.prox_mu < 0:
        raise ValueError("--prox_mu must be non-negative.")
    adapl_ablation_requested = (
        args.adapl_disable_noise
        or args.adapl_disable_clipping
        or args.adapl_disable_fisher
        or args.adapl_noise_scope != "fisher"
        or args.adapl_freeze_bn
    )
    if adapl_ablation_requested and args.method != "adapl":
        raise ValueError("AdapL diagnostic switches require --method adapl.")
    if adapl_ablation_requested and args.no_dp:
        raise ValueError(
            "AdapL diagnostic switches exercise the DP trainer and require --use_dp. "
            "Use --no_dp alone for the ordinary SGD/FedAvg baseline."
        )
    if args.lr <= 0:
        raise ValueError("--lr must be positive.")
    args.lr_milestones = _parse_int_list(args.lr_milestones, "--lr_milestones")
    args.lr_values = _parse_float_list(args.lr_values, "--lr_values")
    if args.lr_milestones or args.lr_values:
        args.lr_schedule = "piecewise"
    if args.lr_schedule == "piecewise":
        if not args.lr_milestones:
            raise ValueError("--lr_schedule piecewise requires --lr_milestones.")
        if any(milestone <= 0 for milestone in args.lr_milestones):
            raise ValueError("--lr_milestones must contain positive round numbers.")
        if args.lr_milestones != sorted(set(args.lr_milestones)):
            raise ValueError("--lr_milestones must be strictly increasing.")
        if len(args.lr_values) != len(args.lr_milestones) + 1:
            raise ValueError(
                "--lr_values length must equal len(--lr_milestones)+1."
            )
        if any(value <= 0 for value in args.lr_values):
            raise ValueError("--lr_values must contain positive learning rates.")
        args.lr = args.lr_values[0]
    else:
        args.lr_milestones = []
        args.lr_values = []
    if args.early_stop_patience is not None and args.early_stop_patience <= 0:
        raise ValueError("--early_stop_patience must be positive.")
    if args.early_stop_min_delta < 0:
        raise ValueError("--early_stop_min_delta must be non-negative.")

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


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = build_parser()
    return normalize_args(parser.parse_args(argv))
