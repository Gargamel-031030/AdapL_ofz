"""CSV and JSON reporting helpers."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from typing import Dict, Iterable, Mapping, Optional, Sequence

from adapl.constants import (
    CIFAR100_HARD_DIRICHLET_ALPHA,
    CIFAR100_MAIN_BATCH_SIZE,
    CIFAR100_MAIN_CLIENT_FRACTION,
    CIFAR100_MAIN_DIRICHLET_ALPHA,
    CIFAR100_MAIN_GLOBAL_ROUNDS,
    CIFAR100_MAIN_LOCAL_STEPS,
    CIFAR100_MAIN_LR,
    PAPER_SOURCE_DIRICHLET_ALPHA,
)
from adapl.utils import ensure_parent_dir


def resolve_run_config_csv_path(
    output_csv: str,
    run_config_csv: Optional[str],
) -> str:
    if run_config_csv:
        return run_config_csv
    base_path, _ = os.path.splitext(output_csv)
    return f"{base_path}_config.csv"


def resolve_client_distribution_csv_path(
    output_csv: str,
    client_distribution_csv: Optional[str],
) -> str:
    if client_distribution_csv:
        return client_distribution_csv
    base_path, _ = os.path.splitext(output_csv)
    return f"{base_path}_client_distribution.csv"


def save_client_label_distribution_csv(
    path: str,
    distribution: Sequence[Dict[str, object]],
    num_classes: int,
) -> None:
    ensure_parent_dir(path)
    with open(path, "w", newline="") as csvfile:
        fieldnames = ["client_id", "total_samples", "num_classes"] + [
            f"class_{class_id}" for class_id in range(num_classes)
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for item in distribution:
            label_counts = item["label_counts"]
            row = {
                "client_id": item["client_id"],
                "total_samples": item["total_samples"],
                "num_classes": item["num_classes"],
            }
            row.update(
                {
                    f"class_{class_id}": label_counts[str(class_id)]
                    for class_id in range(num_classes)
                }
            )
            writer.writerow(row)


def save_client_label_distribution_json(
    path: str,
    distribution: Sequence[Dict[str, object]],
    partition: str,
    dirichlet_alpha: float,
    num_classes: int,
) -> None:
    ensure_parent_dir(path)
    payload = {
        "partition": partition,
        "dirichlet_alpha": dirichlet_alpha,
        "num_clients": len(distribution),
        "num_classes": num_classes,
        "clients": distribution,
    }
    with open(path, "w") as jsonfile:
        json.dump(payload, jsonfile, indent=2)


def save_run_config_json(
    path: str,
    args: argparse.Namespace,
    method_payload: Dict[str, object],
    client_distribution: Sequence[Dict[str, object]],
) -> None:
    ensure_parent_dir(path)
    payload = {
        "method": method_payload,
        "paper_migration": {
            "source_pdf": "TNSE.pdf",
            "source_dataset": "CIFAR10",
            "target_dataset": "CIFAR100",
            "model": "ResNet18",
            "partition": "Dirichlet non-IID",
            "paper_cifar10_dirichlet_alpha": PAPER_SOURCE_DIRICHLET_ALPHA,
            "cifar100_main_dirichlet_alpha": CIFAR100_MAIN_DIRICHLET_ALPHA,
            "cifar100_hard_dirichlet_alpha": CIFAR100_HARD_DIRICHLET_ALPHA,
            "cifar100_main_client_fraction": CIFAR100_MAIN_CLIENT_FRACTION,
            "cifar100_main_global_rounds": CIFAR100_MAIN_GLOBAL_ROUNDS,
            "cifar100_main_batch_size": CIFAR100_MAIN_BATCH_SIZE,
            "cifar100_main_local_steps": CIFAR100_MAIN_LOCAL_STEPS,
            "cifar100_main_lr": CIFAR100_MAIN_LR,
            "local_update": "random mini-batch SGD steps per selected client per round",
        },
        "args": vars(args),
        "effective_local_updates_per_round": (
            args.local_steps
            if args.local_update_mode == "random-batch"
            else args.local_epochs
        ),
        "client_summary": [
            {
                "client_id": item["client_id"],
                "total_samples": item["total_samples"],
                "num_classes": item["num_classes"],
            }
            for item in client_distribution
        ],
    }
    with open(path, "w") as jsonfile:
        json.dump(payload, jsonfile, indent=2)


def save_run_config_csv(
    path: str,
    args: argparse.Namespace,
    method_rows: Sequence[tuple[str, str, object]],
    client_distribution: Sequence[Dict[str, object]],
    final_test_accuracy: Optional[float] = None,
) -> None:
    ensure_parent_dir(path)

    total_samples = sum(int(item["total_samples"]) for item in client_distribution)
    min_client_samples = min(int(item["total_samples"]) for item in client_distribution)
    max_client_samples = max(int(item["total_samples"]) for item in client_distribution)
    min_client_classes = min(int(item["num_classes"]) for item in client_distribution)
    max_client_classes = max(int(item["num_classes"]) for item in client_distribution)

    rows = [
        *method_rows,
        ("paper_migration", "source_pdf", "TNSE.pdf"),
        ("paper_migration", "source_dataset", "CIFAR10"),
        ("paper_migration", "target_dataset", "CIFAR100"),
        ("paper_migration", "model", "ResNet18"),
        ("paper_migration", "partition", "Dirichlet non-IID"),
        ("paper_migration", "paper_cifar10_dirichlet_alpha", PAPER_SOURCE_DIRICHLET_ALPHA),
        ("cifar100_main", "dirichlet_alpha", CIFAR100_MAIN_DIRICHLET_ALPHA),
        ("cifar100_main", "hard_dirichlet_alpha", CIFAR100_HARD_DIRICHLET_ALPHA),
        ("cifar100_main", "client_fraction", CIFAR100_MAIN_CLIENT_FRACTION),
        ("cifar100_main", "global_rounds", CIFAR100_MAIN_GLOBAL_ROUNDS),
        ("cifar100_main", "batch_size", CIFAR100_MAIN_BATCH_SIZE),
        ("cifar100_main", "local_steps", CIFAR100_MAIN_LOCAL_STEPS),
        ("cifar100_main", "lr", CIFAR100_MAIN_LR),
        (
            "paper_migration",
            "local_update",
            "random mini-batch SGD steps per selected client per round",
        ),
        ("client_summary", "total_samples", total_samples),
        ("client_summary", "min_client_samples", min_client_samples),
        ("client_summary", "max_client_samples", max_client_samples),
        ("client_summary", "min_client_classes", min_client_classes),
        ("client_summary", "max_client_classes", max_client_classes),
    ]

    for key, value in sorted(vars(args).items()):
        rows.append(("args", key, value))

    effective_local_updates = (
        args.local_steps if args.local_update_mode == "random-batch" else args.local_epochs
    )
    rows.append(("effective", "local_updates_per_round", effective_local_updates))

    if final_test_accuracy is not None:
        rows.append(("result", "final_test_accuracy", f"{final_test_accuracy:.6f}"))

    with open(path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["section", "key", "value"])
        writer.writerows(rows)


def init_output_csv(path: str) -> None:
    ensure_parent_dir(path)
    with open(path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "round",
                "selected_clients",
                "train_loss",
                "test_loss",
                "test_accuracy",
                "selected_epsilons",
                "aggregation_weights",
                "dp_update_norm_mean",
                "dp_update_norm_min",
                "dp_update_norm_max",
                "dp_clipped_norm_mean",
                "dp_clip_factor_mean",
                "dp_clip_factor_min",
                "dp_noise_std_mean",
                "dp_epsilon_mean",
                "dp_epsilon_min",
                "aggregation_weight_mean",
                "aggregation_weight_min",
                "aggregation_weight_max",
                "feddpa_fisher_personalized_ratio_mean",
                "feddpa_fisher_personalized_ratio_min",
                "feddpa_fisher_personalized_ratio_max",
                "privacy_budget_accumulated_mean",
                "privacy_budget_accumulated_min",
                "privacy_budget_accumulated_max",
                "privacy_budget_current_steps_mean",
                "privacy_budget_active_clients",
                "privacy_budget_finished_clients",
            ]
        )


def _format_optional_metric(value: object) -> str:
    if value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number):
        return ""
    return f"{number:.6f}"


def append_output_csv(
    path: str,
    round_idx: int,
    selected_clients: Iterable[int],
    train_loss: float,
    test_loss: float,
    test_accuracy: float,
    round_metrics: Optional[Mapping[str, object]] = None,
) -> None:
    round_metrics = round_metrics or {}
    with open(path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                round_idx,
                " ".join(str(client_id) for client_id in selected_clients),
                f"{train_loss:.6f}",
                f"{test_loss:.6f}",
                f"{test_accuracy:.6f}",
                _format_optional_metric(round_metrics.get("selected_epsilons")),
                _format_optional_metric(round_metrics.get("aggregation_weights")),
                _format_optional_metric(round_metrics.get("dp_update_norm_mean")),
                _format_optional_metric(round_metrics.get("dp_update_norm_min")),
                _format_optional_metric(round_metrics.get("dp_update_norm_max")),
                _format_optional_metric(round_metrics.get("dp_clipped_norm_mean")),
                _format_optional_metric(round_metrics.get("dp_clip_factor_mean")),
                _format_optional_metric(round_metrics.get("dp_clip_factor_min")),
                _format_optional_metric(round_metrics.get("dp_noise_std_mean")),
                _format_optional_metric(round_metrics.get("dp_epsilon_mean")),
                _format_optional_metric(round_metrics.get("dp_epsilon_min")),
                _format_optional_metric(round_metrics.get("aggregation_weight_mean")),
                _format_optional_metric(round_metrics.get("aggregation_weight_min")),
                _format_optional_metric(round_metrics.get("aggregation_weight_max")),
                _format_optional_metric(
                    round_metrics.get("feddpa_fisher_personalized_ratio_mean")
                ),
                _format_optional_metric(
                    round_metrics.get("feddpa_fisher_personalized_ratio_min")
                ),
                _format_optional_metric(
                    round_metrics.get("feddpa_fisher_personalized_ratio_max")
                ),
                _format_optional_metric(
                    round_metrics.get("privacy_budget_accumulated_mean")
                ),
                _format_optional_metric(
                    round_metrics.get("privacy_budget_accumulated_min")
                ),
                _format_optional_metric(
                    round_metrics.get("privacy_budget_accumulated_max")
                ),
                _format_optional_metric(
                    round_metrics.get("privacy_budget_current_steps_mean")
                ),
                _format_optional_metric(
                    round_metrics.get("privacy_budget_active_clients")
                ),
                _format_optional_metric(
                    round_metrics.get("privacy_budget_finished_clients")
                ),
            ]
        )
