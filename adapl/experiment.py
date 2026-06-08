"""Shared experiment runner."""

from __future__ import annotations

import math
from argparse import Namespace
from typing import Sequence

from adapl.constants import CIFAR100_NUM_CLASSES
from adapl.data import (
    build_client_partitions,
    compute_client_label_distribution,
    load_cifar100,
    log_client_label_distribution,
)
from adapl.fl.evaluation import evaluate
from adapl.fl.loaders import make_test_loader, make_train_loaders
from adapl.fl.sampling import select_clients
from adapl.methods import build_method
from adapl.models.resnet import build_model_fn, validate_model_output
from adapl.reporting import (
    append_output_csv,
    init_output_csv,
    resolve_client_distribution_csv_path,
    resolve_run_config_csv_path,
    save_client_label_distribution_csv,
    save_client_label_distribution_json,
    save_run_config_csv,
    save_run_config_json,
)
from adapl.utils import clone_state_dict, resolve_device, set_random_seed


def _validate_args(args: Namespace) -> None:
    if args.global_rounds <= 0:
        raise ValueError("--global_rounds must be positive.")
    if args.eval_every <= 0:
        raise ValueError("--eval_every must be positive.")
    if args.num_clients <= 0:
        raise ValueError("--num_clients must be positive.")
    if not 0 < args.client_fraction <= 1:
        raise ValueError("--client_fraction must be in (0, 1].")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive.")
    if args.test_batch_size <= 0:
        raise ValueError("--test_batch_size must be positive.")


def _format_selected_clients(selected_clients: Sequence[int]) -> str:
    return "[" + ", ".join(str(client_id) for client_id in selected_clients) + "]"


def _float_metadata_values(client_updates, key: str) -> list[float]:
    values = []
    for update in client_updates:
        if key not in update.metadata:
            continue
        value = update.metadata[key]
        if value is None:
            continue
        values.append(float(value))
    return values


def _format_float_sequence(values: list[float]) -> str:
    return " ".join(f"{value:.6g}" for value in values)


def _summarize_round_metadata(client_updates) -> dict[str, object]:
    update_norms = _float_metadata_values(client_updates, "update_norm")
    clipped_norms = _float_metadata_values(client_updates, "clipped_norm")
    clip_factors = _float_metadata_values(client_updates, "clip_factor")
    noise_stds = _float_metadata_values(client_updates, "noise_std")
    client_epsilons = _float_metadata_values(client_updates, "epsilon")
    epsilons = _float_metadata_values(client_updates, "epsilon_min")
    aggregation_weights = _float_metadata_values(
        client_updates,
        "aggregation_weight",
    )

    metrics = {}
    if update_norms:
        metrics["dp_update_norm_mean"] = sum(update_norms) / len(update_norms)
        metrics["dp_update_norm_min"] = min(update_norms)
        metrics["dp_update_norm_max"] = max(update_norms)
    if clipped_norms:
        metrics["dp_clipped_norm_mean"] = sum(clipped_norms) / len(clipped_norms)
    if clip_factors:
        metrics["dp_clip_factor_mean"] = sum(clip_factors) / len(clip_factors)
        metrics["dp_clip_factor_min"] = min(clip_factors)
    if noise_stds:
        metrics["dp_noise_std_mean"] = sum(noise_stds) / len(noise_stds)
    if client_epsilons:
        metrics["selected_epsilons"] = _format_float_sequence(client_epsilons)
        metrics["dp_epsilon_mean"] = sum(client_epsilons) / len(client_epsilons)
    if epsilons:
        metrics["dp_epsilon_min"] = min(epsilons)
    if aggregation_weights:
        metrics["aggregation_weights"] = _format_float_sequence(aggregation_weights)
        metrics["aggregation_weight_mean"] = (
            sum(aggregation_weights) / len(aggregation_weights)
        )
        metrics["aggregation_weight_min"] = min(aggregation_weights)
        metrics["aggregation_weight_max"] = max(aggregation_weights)
    return metrics


def _format_round_metadata(metrics: dict[str, object]) -> str:
    if not metrics:
        return ""
    text = (
        " | "
        f"upd_norm={metrics.get('dp_update_norm_mean', math.nan):.4f} "
        f"clip={metrics.get('dp_clip_factor_mean', math.nan):.4f} "
        f"noise_std={metrics.get('dp_noise_std_mean', math.nan):.4f} "
        f"eps_min={metrics.get('dp_epsilon_min', math.nan):.4f}"
    )
    if "aggregation_weight_mean" in metrics:
        text += (
            f" eps_mean={metrics.get('dp_epsilon_mean', math.nan):.4f} "
            f"w_range=[{metrics.get('aggregation_weight_min', math.nan):.4f},"
            f"{metrics.get('aggregation_weight_max', math.nan):.4f}]"
        )
    return text


def run_experiment(args: Namespace) -> None:
    _validate_args(args)
    method = build_method(args.method, args)

    set_random_seed(args.seed)
    device = resolve_device(args.device)
    model_fn = build_model_fn(args.model)
    validate_model_output(model_fn, CIFAR100_NUM_CLASSES)

    train_dataset, test_dataset = load_cifar100(
        data_dir=args.data_dir,
        seed=args.seed,
        limit_train=args.limit_train_samples,
        limit_test=args.limit_test_samples,
    )
    client_datasets = build_client_partitions(
        train_dataset,
        num_clients=args.num_clients,
        partition=args.partition,
        dirichlet_alpha=args.dirichlet_alpha,
        seed=args.seed,
    )
    client_label_distribution = compute_client_label_distribution(
        client_datasets,
        num_classes=CIFAR100_NUM_CLASSES,
    )

    args.run_config_csv = resolve_run_config_csv_path(
        args.output_csv,
        args.run_config_csv,
    )
    args.client_distribution_csv = resolve_client_distribution_csv_path(
        args.output_csv,
        args.client_distribution_csv,
    )

    save_client_label_distribution_csv(
        args.client_distribution_csv,
        client_label_distribution,
        num_classes=CIFAR100_NUM_CLASSES,
    )
    if args.client_distribution_json:
        save_client_label_distribution_json(
            args.client_distribution_json,
            client_label_distribution,
            partition=args.partition,
            dirichlet_alpha=args.dirichlet_alpha,
            num_classes=CIFAR100_NUM_CLASSES,
        )
    if args.run_config_json:
        save_run_config_json(
            args.run_config_json,
            args,
            method.config_payload(),
            client_label_distribution,
        )
    save_run_config_csv(
        args.run_config_csv,
        args,
        method.config_rows(),
        client_label_distribution,
    )

    train_loaders = make_train_loaders(
        client_datasets,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    test_loader = make_test_loader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
    )

    global_model = model_fn().to(device)
    global_state = clone_state_dict(global_model.state_dict())
    init_output_csv(args.output_csv)

    print(f"Method: {method.display_name}")
    print(f"Dataset: {args.dataset}, model: {args.model}")
    print(f"Data directory: {args.data_dir}")
    print(f"Device: {device}")
    print(f"Train samples: {len(train_dataset)}, test samples: {len(test_dataset)}")
    print(f"Clients: {args.num_clients}, client_fraction: {args.client_fraction}")
    print(f"Global rounds: {args.global_rounds}")
    print(f"Local update mode: {args.local_update_mode}")
    if args.local_update_mode == "random-batch":
        print(
            f"Local update: {args.local_steps} random mini-batch SGD steps "
            "per selected client per round"
        )
    else:
        print(f"Local update: {args.local_epochs} full local epochs per selected client")
    print(f"Batch size: {args.batch_size}, test_batch_size: {args.test_batch_size}")
    print(
        f"Learning rate: {args.lr}, momentum: {args.momentum}, "
        f"weight_decay: {args.weight_decay}"
    )
    print(f"Partition: {args.partition}")
    print(f"Dirichlet alpha: {args.dirichlet_alpha}")
    log_client_label_distribution(client_label_distribution)
    print(f"Client distribution CSV saved to: {args.client_distribution_csv}")
    if args.client_distribution_json:
        print(f"Client distribution JSON saved to: {args.client_distribution_json}")
    if args.run_config_json:
        print(f"Run config JSON saved to: {args.run_config_json}")
    print(f"Run config CSV saved to: {args.run_config_csv}")
    for line in method.startup_lines():
        print(line)

    final_test_acc = math.nan
    for round_idx in range(1, args.global_rounds + 1):
        selected_clients = select_clients(
            args.num_clients,
            args.client_fraction,
            round_idx,
            args.seed,
        )
        method.begin_round(round_idx, selected_clients)

        client_updates = []
        weighted_loss_sum = 0.0
        for client_id in selected_clients:
            update = method.train_client(
                client_id=client_id,
                model_fn=model_fn,
                global_state=global_state,
                train_loader=train_loaders[client_id],
                device=device,
            )
            client_updates.append(update)
            weighted_loss_sum += update.train_loss * update.num_examples

        total_examples = sum(update.num_examples for update in client_updates)
        train_loss = weighted_loss_sum / float(total_examples)
        round_metrics = _summarize_round_metadata(client_updates)
        global_state = method.aggregate(client_updates)
        global_model.load_state_dict(global_state)

        if round_idx % args.eval_every == 0 or round_idx == args.global_rounds:
            test_loss, test_accuracy = evaluate(global_model, test_loader, device)
            final_test_acc = test_accuracy
        else:
            test_loss, test_accuracy = math.nan, math.nan

        append_output_csv(
            args.output_csv,
            round_idx=round_idx,
            selected_clients=selected_clients,
            train_loss=train_loss,
            test_loss=test_loss,
            test_accuracy=test_accuracy,
            round_metrics=round_metrics,
        )
        print(
            f"Round {round_idx:03d}/{args.global_rounds} | "
            f"clients={_format_selected_clients(selected_clients)} | "
            f"train_loss={train_loss:.4f} | "
            f"test_loss={test_loss:.4f} | "
            f"test_acc={test_accuracy:.4f}"
            f"{_format_round_metadata(round_metrics)}"
        )

    print(f"Final test accuracy: {final_test_acc:.4f}")
    print(f"Metrics saved to: {args.output_csv}")
    save_run_config_csv(
        args.run_config_csv,
        args,
        method.config_rows(),
        client_label_distribution,
        final_test_accuracy=final_test_acc,
    )
    print(f"Run summary CSV saved to: {args.run_config_csv}")
