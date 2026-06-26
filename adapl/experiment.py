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
from adapl.methods import build_method
from adapl.models.resnet import build_model_fn, validate_model_output
from adapl.privacy.accounting import build_privacy_budget_manager_from_args
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
    if not hasattr(args, "early_stop_patience"):
        args.early_stop_patience = None
    if not hasattr(args, "early_stop_min_delta"):
        args.early_stop_min_delta = 0.0
    if not hasattr(args, "lr_schedule"):
        args.lr_schedule = "constant"
    if not hasattr(args, "lr_milestones"):
        args.lr_milestones = []
    if not hasattr(args, "lr_values"):
        args.lr_values = []
    if not hasattr(args, "prox_mu"):
        args.prox_mu = 0.0
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
    if args.prox_mu < 0:
        raise ValueError("--prox_mu must be non-negative.")
    if args.lr <= 0:
        raise ValueError("--lr must be positive.")
    if args.lr_schedule not in {"constant", "piecewise"}:
        raise ValueError("--lr_schedule must be constant or piecewise.")
    if args.lr_schedule == "piecewise":
        if len(args.lr_values) != len(args.lr_milestones) + 1:
            raise ValueError(
                "--lr_values length must equal len(--lr_milestones)+1."
            )
    if args.early_stop_patience is not None and args.early_stop_patience <= 0:
        raise ValueError("--early_stop_patience must be positive.")
    if args.early_stop_min_delta < 0:
        raise ValueError("--early_stop_min_delta must be non-negative.")


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


def _learning_rate_for_round(args: Namespace, round_idx: int) -> float:
    if getattr(args, "lr_schedule", "constant") != "piecewise":
        return float(args.lr)

    milestones = list(getattr(args, "lr_milestones", []))
    values = list(getattr(args, "lr_values", []))
    if not milestones or not values:
        return float(args.lr)

    value_index = 0
    for milestone in milestones:
        if round_idx > int(milestone):
            value_index += 1
        else:
            break
    return float(values[min(value_index, len(values) - 1)])


def _summarize_round_metadata(client_updates) -> dict[str, object]:
    update_norms = _float_metadata_values(client_updates, "update_norm")
    clipped_norms = _float_metadata_values(client_updates, "clipped_norm")
    clip_factors = _float_metadata_values(client_updates, "clip_factor")
    clip_fractions = _float_metadata_values(client_updates, "clip_fraction")
    layer_clip_factors = _float_metadata_values(client_updates, "layer_clip_factor")
    coordinate_clip_fractions = _float_metadata_values(
        client_updates,
        "coordinate_clip_fraction",
    )
    coordinate_clip_radii = _float_metadata_values(
        client_updates,
        "coordinate_clip_radius",
    )
    privacy_clip_scales = _float_metadata_values(
        client_updates,
        "privacy_clip_scale",
    )
    sample_grad_norms = _float_metadata_values(client_updates, "sample_grad_norm")
    sample_grad_norm_p50s = _float_metadata_values(
        client_updates, "sample_grad_norm_p50"
    )
    sample_grad_norm_p90s = _float_metadata_values(
        client_updates, "sample_grad_norm_p90"
    )
    sample_grad_norm_p99s = _float_metadata_values(
        client_updates, "sample_grad_norm_p99"
    )
    proximal_norms = _float_metadata_values(client_updates, "proximal_norm")
    noise_stds = _float_metadata_values(client_updates, "noise_std")
    signal_l2s = _float_metadata_values(client_updates, "signal_l2")
    noise_l2s = _float_metadata_values(client_updates, "noise_l2")
    noise_to_signal_ratios = _float_metadata_values(
        client_updates, "noise_to_signal_ratio"
    )
    fisher_important_ratios = _float_metadata_values(
        client_updates, "fisher_important_ratio"
    )
    actual_minibatch_steps = _float_metadata_values(
        client_updates,
        "actual_minibatch_steps",
    )
    noise_multipliers = _float_metadata_values(client_updates, "noise_multiplier")
    noise_multiplier_mins = _float_metadata_values(
        client_updates,
        "noise_multiplier_min",
    )
    noise_multiplier_maxs = _float_metadata_values(
        client_updates,
        "noise_multiplier_max",
    )
    client_epsilons = _float_metadata_values(client_updates, "epsilon")
    epsilon_targets = _float_metadata_values(client_updates, "epsilon_target")
    aggregation_weights = _float_metadata_values(
        client_updates,
        "aggregation_weight",
    )
    fisher_personalized_ratios = _float_metadata_values(
        client_updates,
        "fisher_personalized_ratio",
    )
    pfa_public_flags = [
        bool(update.metadata["pfa_is_public"])
        for update in client_updates
        if "pfa_is_public" in update.metadata
    ]
    privacy_budget_accumulated = _float_metadata_values(
        client_updates,
        "privacy_budget_accumulated",
    )
    privacy_budget_current_steps = _float_metadata_values(
        client_updates,
        "privacy_budget_current_steps",
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
    if layer_clip_factors:
        metrics["adapl_layer_clip_factor_mean"] = (
            sum(layer_clip_factors) / len(layer_clip_factors)
        )
    if clip_fractions:
        metrics["adapl_clip_fraction_mean"] = sum(clip_fractions) / len(
            clip_fractions
        )
    if coordinate_clip_fractions:
        metrics["adapl_coordinate_clip_fraction_mean"] = (
            sum(coordinate_clip_fractions) / len(coordinate_clip_fractions)
        )
    if coordinate_clip_radii:
        metrics["adapl_coordinate_clip_radius_mean"] = (
            sum(coordinate_clip_radii) / len(coordinate_clip_radii)
        )
    if privacy_clip_scales:
        metrics["adapl_privacy_clip_scale_mean"] = (
            sum(privacy_clip_scales) / len(privacy_clip_scales)
        )
    if sample_grad_norms:
        metrics["adapl_sample_grad_norm_mean"] = sum(sample_grad_norms) / len(
            sample_grad_norms
        )
    if sample_grad_norm_p50s:
        metrics["adapl_sample_grad_norm_p50_mean"] = sum(
            sample_grad_norm_p50s
        ) / len(sample_grad_norm_p50s)
    if sample_grad_norm_p90s:
        metrics["adapl_sample_grad_norm_p90_mean"] = sum(
            sample_grad_norm_p90s
        ) / len(sample_grad_norm_p90s)
    if sample_grad_norm_p99s:
        metrics["adapl_sample_grad_norm_p99_mean"] = sum(
            sample_grad_norm_p99s
        ) / len(sample_grad_norm_p99s)
    if proximal_norms:
        metrics["dp_proximal_norm_mean"] = sum(proximal_norms) / len(proximal_norms)
    if noise_stds:
        metrics["dp_noise_std_mean"] = sum(noise_stds) / len(noise_stds)
    if signal_l2s:
        metrics["adapl_signal_l2_mean"] = sum(signal_l2s) / len(signal_l2s)
    if noise_l2s:
        metrics["adapl_noise_l2_mean"] = sum(noise_l2s) / len(noise_l2s)
    if noise_to_signal_ratios:
        metrics["adapl_noise_to_signal_ratio_mean"] = sum(
            noise_to_signal_ratios
        ) / len(noise_to_signal_ratios)
    if fisher_important_ratios:
        metrics["adapl_fisher_important_ratio_mean"] = sum(
            fisher_important_ratios
        ) / len(fisher_important_ratios)
        metrics["adapl_fisher_important_ratio_min"] = min(fisher_important_ratios)
        metrics["adapl_fisher_important_ratio_max"] = max(fisher_important_ratios)
    if actual_minibatch_steps:
        metrics["actual_minibatch_steps_min"] = min(actual_minibatch_steps)
        metrics["actual_minibatch_steps_max"] = max(actual_minibatch_steps)
        metrics["actual_minibatch_steps_mean"] = (
            sum(actual_minibatch_steps) / len(actual_minibatch_steps)
        )
    if noise_multipliers:
        metrics["noise_multiplier_min"] = (
            min(noise_multiplier_mins) if noise_multiplier_mins else min(noise_multipliers)
        )
        metrics["noise_multiplier_max"] = (
            max(noise_multiplier_maxs) if noise_multiplier_maxs else max(noise_multipliers)
        )
        metrics["noise_multiplier_mean"] = (
            sum(noise_multipliers) / len(noise_multipliers)
        )
    if client_epsilons:
        metrics["selected_epsilons"] = _format_float_sequence(client_epsilons)
        metrics["epsilon_distribution"] = _format_float_sequence(client_epsilons)
        metrics["dp_epsilon_mean"] = sum(client_epsilons) / len(client_epsilons)
        metrics["dp_epsilon_min"] = min(client_epsilons)
        metrics["dp_epsilon_max"] = max(client_epsilons)
    if epsilon_targets:
        metrics["epsilon_targets"] = _format_float_sequence(epsilon_targets)
        metrics["epsilon_target_min"] = min(epsilon_targets)
        metrics["epsilon_target_max"] = max(epsilon_targets)
    if aggregation_weights:
        metrics["aggregation_weights"] = _format_float_sequence(aggregation_weights)
        metrics["aggregation_weight_mean"] = (
            sum(aggregation_weights) / len(aggregation_weights)
        )
        metrics["aggregation_weight_min"] = min(aggregation_weights)
        metrics["aggregation_weight_max"] = max(aggregation_weights)
    if fisher_personalized_ratios:
        metrics["feddpa_fisher_personalized_ratio_mean"] = (
            sum(fisher_personalized_ratios) / len(fisher_personalized_ratios)
        )
        metrics["feddpa_fisher_personalized_ratio_min"] = min(
            fisher_personalized_ratios
        )
        metrics["feddpa_fisher_personalized_ratio_max"] = max(
            fisher_personalized_ratios
        )
    if pfa_public_flags:
        pfa_public_clients = sum(1 for is_public in pfa_public_flags if is_public)
        metrics["pfa_public_clients"] = pfa_public_clients
        metrics["pfa_private_clients"] = len(pfa_public_flags) - pfa_public_clients
    if privacy_budget_accumulated:
        metrics["privacy_budget_accumulated_mean"] = (
            sum(privacy_budget_accumulated) / len(privacy_budget_accumulated)
        )
        metrics["privacy_budget_accumulated_min"] = min(privacy_budget_accumulated)
        metrics["privacy_budget_accumulated_max"] = max(privacy_budget_accumulated)
    if privacy_budget_current_steps:
        metrics["privacy_budget_current_steps_mean"] = (
            sum(privacy_budget_current_steps) / len(privacy_budget_current_steps)
        )
    privacy_scenarios = sorted(
        {
            str(update.metadata["privacy_scenario"])
            for update in client_updates
            if update.metadata.get("privacy_scenario")
        }
    )
    if privacy_scenarios:
        metrics["privacy_scenario"] = " ".join(privacy_scenarios)
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
    if "feddpa_fisher_personalized_ratio_mean" in metrics:
        text += (
            " fisher_personal="
            f"{metrics.get('feddpa_fisher_personalized_ratio_mean', math.nan):.4f}"
        )
    if "pfa_public_clients" in metrics:
        text += (
            " pfa_public/private="
            f"{int(metrics.get('pfa_public_clients', 0))}/"
            f"{int(metrics.get('pfa_private_clients', 0))}"
        )
    if "privacy_budget_accumulated_mean" in metrics:
        text += (
            " budget="
            f"{metrics.get('privacy_budget_accumulated_mean', math.nan):.4f}"
        )
    if "actual_minibatch_steps_mean" in metrics:
        text += (
            " actual_steps="
            f"{metrics.get('actual_minibatch_steps_mean', math.nan):.2f}"
        )
    if "dp_proximal_norm_mean" in metrics:
        text += f" prox={metrics.get('dp_proximal_norm_mean', math.nan):.4f}"
    if "noise_multiplier_mean" in metrics:
        text += (
            " nm="
            f"{metrics.get('noise_multiplier_mean', math.nan):.4f}"
        )
    if "adapl_sample_grad_norm_p90_mean" in metrics:
        text += (
            " grad_p90="
            f"{metrics.get('adapl_sample_grad_norm_p90_mean', math.nan):.3f}"
            " clip_frac="
            f"{metrics.get('adapl_clip_fraction_mean', math.nan):.3f}"
            " noise_l2="
            f"{metrics.get('adapl_noise_l2_mean', math.nan):.3f}"
            " noise/signal="
            f"{metrics.get('adapl_noise_to_signal_ratio_mean', math.nan):.3f}"
            " fisher_ratio="
            f"{metrics.get('adapl_fisher_important_ratio_mean', math.nan):.3f}"
        )
    if "privacy_budget_active_clients" in metrics:
        text += (
            " active_clients="
            f"{int(metrics.get('privacy_budget_active_clients', 0))}"
        )
    if "learning_rate" in metrics:
        text += f" lr={metrics.get('learning_rate', math.nan):.6g}"
    return text


def run_experiment(args: Namespace) -> None:
    _validate_args(args)
    method = build_method(args.method, args)
    method_uses_internal_accountant = bool(
        getattr(method, "uses_internal_privacy_accountant", False)
    )
    privacy_accounting_mode = getattr(args, "privacy_accounting", "auto")
    privacy_budget_manager = None
    if privacy_accounting_mode != "off" and not method_uses_internal_accountant:
        privacy_budget_manager = method.build_privacy_budget_manager()
        if privacy_budget_manager is None and privacy_accounting_mode == "on":
            privacy_budget_manager = build_privacy_budget_manager_from_args(args)

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
    if hasattr(method, "prepare_privacy_accountants"):
        method.prepare_privacy_accountants(train_loaders)

    global_model = model_fn().to(device)
    global_state = clone_state_dict(global_model.state_dict())
    init_output_csv(args.output_csv)
    privacy_dataset_sizes = {
        client_id: len(train_loaders[client_id].dataset)
        for client_id in range(args.num_clients)
    }
    privacy_local_steps = {
        client_id: method.privacy_budget_local_steps(train_loaders[client_id])
        for client_id in range(args.num_clients)
    }

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
    if getattr(args, "lr_schedule", "constant") == "piecewise":
        print(
            "Learning-rate schedule: "
            f"piecewise milestones={args.lr_milestones}, values={args.lr_values}"
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
    print(f"Privacy accounting mode: {privacy_accounting_mode}")
    if method_uses_internal_accountant:
        print(
            "Privacy budget accountant: "
            f"enabled internally for {method.num_accountants} clients, "
            "precheck_filter=before_sampling"
        )
    elif privacy_budget_manager is not None:
        print(
            "Privacy budget accountant: "
            f"enabled for {privacy_budget_manager.num_clients} clients, "
            "precheck_filter=before_sampling"
        )
    else:
        print("Privacy budget accountant: disabled")
    for line in method.startup_lines():
        print(line)

    final_test_acc = math.nan
    last_test_acc = math.nan
    best_test_acc = -math.inf
    best_test_loss = math.nan
    best_test_round: int | None = None
    best_global_state = None
    evaluations_since_best = 0
    initial_lr = float(args.lr)
    for round_idx in range(1, args.global_rounds + 1):
        current_lr = _learning_rate_for_round(args, round_idx)
        args.lr = current_lr
        candidate_client_ids = None
        if method_uses_internal_accountant:
            candidate_client_ids = method.eligible_client_ids(
                list(range(args.num_clients))
            )
            if not candidate_client_ids:
                print(
                    f"Round {round_idx:03d}/{args.global_rounds} skipped: "
                    "all client privacy budgets are exhausted."
                )
                break
        elif privacy_budget_manager is not None:
            candidate_client_ids = privacy_budget_manager.eligible_client_ids(
                client_ids=list(range(args.num_clients)),
                dataset_sizes=privacy_dataset_sizes,
                batch_size=args.batch_size,
                local_steps_by_client=privacy_local_steps,
            )
            if not candidate_client_ids:
                print(
                    f"Round {round_idx:03d}/{args.global_rounds} skipped: "
                    "all client privacy budgets are exhausted."
                )
                break

        selected_clients = method.select_clients(
            num_clients=args.num_clients,
            client_fraction=args.client_fraction,
            round_idx=round_idx,
            seed=args.seed,
            candidate_client_ids=candidate_client_ids,
        )
        if not selected_clients:
            print(
                f"Round {round_idx:03d}/{args.global_rounds} skipped: "
                "no eligible clients were selected."
            )
            break
        if privacy_budget_manager is not None:
            method.set_privacy_budget_context(
                privacy_budget_manager.context_for_clients(selected_clients)
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
            if privacy_budget_manager is not None:
                privacy_budget_manager.update_client(
                    client_id,
                    privacy_local_steps[client_id],
                )
                update.metadata.update(
                    privacy_budget_manager.metadata_for_client(client_id)
                )
            client_updates.append(update)
            weighted_loss_sum += update.train_loss * update.num_examples

        total_examples = sum(update.num_examples for update in client_updates)
        train_loss = weighted_loss_sum / float(total_examples)
        global_state = method.aggregate(client_updates)
        global_model.load_state_dict(global_state)
        round_metrics = _summarize_round_metadata(client_updates)
        round_metrics["learning_rate"] = current_lr
        if method_uses_internal_accountant:
            round_metrics["privacy_budget_finished_clients"] = (
                method.num_finished_accountants
            )
            round_metrics["privacy_budget_active_clients"] = (
                method.num_active_accountants
            )
        elif privacy_budget_manager is not None:
            round_metrics["privacy_budget_finished_clients"] = (
                privacy_budget_manager.num_finished
            )
            round_metrics["privacy_budget_active_clients"] = (
                privacy_budget_manager.num_clients
                - privacy_budget_manager.num_finished
            )

        if round_idx % args.eval_every == 0 or round_idx == args.global_rounds:
            test_loss, test_accuracy = evaluate(global_model, test_loader, device)
            last_test_acc = test_accuracy
            final_test_acc = test_accuracy
            if hasattr(method, "observe_global_accuracy"):
                decayed = method.observe_global_accuracy(global_state, test_accuracy)
                round_metrics["adapl_sigma_decayed"] = 1.0 if decayed else 0.0
            previous_best_acc = best_test_acc
            has_evaluation = math.isfinite(test_accuracy)
            is_best_round = has_evaluation and (
                best_test_round is None or test_accuracy > previous_best_acc
            )
            resets_early_stop = has_evaluation and (
                best_test_round is None
                or test_accuracy > previous_best_acc + args.early_stop_min_delta
            )
            if is_best_round:
                best_test_acc = test_accuracy
                best_test_loss = test_loss
                best_test_round = round_idx
                best_global_state = clone_state_dict(global_state)
            if resets_early_stop:
                evaluations_since_best = 0
            elif has_evaluation:
                evaluations_since_best += 1
            round_metrics["is_best_round"] = 1.0 if is_best_round else 0.0
        else:
            test_loss, test_accuracy = math.nan, math.nan
            round_metrics["is_best_round"] = 0.0

        if best_test_round is not None:
            round_metrics["best_test_accuracy"] = best_test_acc
            round_metrics["best_test_round"] = best_test_round

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
        if (
            args.early_stop_patience is not None
            and evaluations_since_best >= args.early_stop_patience
        ):
            print(
                f"Early stopping at round {round_idx:03d}: "
                "no accuracy improvement of at least "
                f"{args.early_stop_min_delta:g} for "
                f"{evaluations_since_best} evaluations."
            )
            break

    args.lr = initial_lr

    if best_global_state is not None and best_test_round is not None:
        global_state = clone_state_dict(best_global_state)
        global_model.load_state_dict(global_state)
        final_test_acc = best_test_acc
        print(
            f"Best test accuracy: {best_test_acc:.4f} "
            f"(round {best_test_round}, test_loss={best_test_loss:.4f})"
        )
        print(f"Last evaluated test accuracy: {last_test_acc:.4f}")

    print(f"Final test accuracy: {final_test_acc:.4f}")
    print(f"Metrics saved to: {args.output_csv}")
    save_run_config_csv(
        args.run_config_csv,
        args,
        method.config_rows(),
        client_label_distribution,
        final_test_accuracy=final_test_acc,
        best_test_accuracy=best_test_acc if best_test_round is not None else None,
        best_test_round=best_test_round,
        last_test_accuracy=last_test_acc if math.isfinite(last_test_acc) else None,
    )
    print(f"Run summary CSV saved to: {args.run_config_csv}")
