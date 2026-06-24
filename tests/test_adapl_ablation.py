from __future__ import annotations

from collections import OrderedDict
import csv
import tempfile
import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from adapl.config import build_parser, normalize_args
from adapl.methods.adapl import AdapL
from adapl.noise_strategy import LayerNoiseStats
from adapl.reporting import append_output_csv, init_output_csv
from adapl.trainers.adapl_trainer import _apply_masked_noise, local_update_first


class _TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.fc(inputs)


class _TinyBatchNormClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(4)
        self.fc = nn.Linear(4, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.fc(self.bn(inputs))


def _loader() -> DataLoader:
    inputs = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 3.0, 4.0, 5.0],
            [3.0, 4.0, 5.0, 6.0],
            [4.0, 5.0, 6.0, 7.0],
        ]
    )
    targets = torch.tensor([0, 0, 1, 1])
    return DataLoader(TensorDataset(inputs, targets), batch_size=4, shuffle=False)


def _state(model: nn.Module) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(
        (name, tensor.detach().clone()) for name, tensor in model.state_dict().items()
    )


class AdapLAblationTest(unittest.TestCase):
    def test_cli_ablation_disables_accounting_without_disabling_dp_trainer(self) -> None:
        args = normalize_args(
            build_parser().parse_args(
                [
                    "--method",
                    "adapl",
                    "--use_dp",
                    "--epsilon_min",
                    "8",
                    "--adapl_disable_noise",
                    "--adapl_disable_fisher",
                    "--adapl_noise_scope",
                    "all",
                    "--adapl_freeze_bn",
                ]
            )
        )
        method = AdapL(args)

        self.assertTrue(method.dp_enabled)
        self.assertFalse(method.accounting_enabled)
        self.assertTrue(method.disable_noise)
        self.assertTrue(method.disable_fisher)
        self.assertEqual(method.noise_scope, "all")
        self.assertTrue(method.freeze_batch_norm)
        self.assertTrue(any("not DP" in line for line in method.startup_lines()))

    def test_two_round_method_ablation_runs_without_privacy_accounting(self) -> None:
        args = normalize_args(
            build_parser().parse_args(
                [
                    "--method",
                    "adapl",
                    "--use_dp",
                    "--epsilon_min",
                    "8",
                    "--num_clients",
                    "1",
                    "--client_fraction",
                    "1",
                    "--global_rounds",
                    "2",
                    "--local_steps",
                    "1",
                    "--batch_size",
                    "4",
                    "--adapl_disable_noise",
                    "--adapl_disable_clipping",
                    "--adapl_disable_fisher",
                ]
            )
        )
        method = AdapL(args)
        loader = _loader()
        method.prepare_privacy_accountants([loader])
        self.assertIn(0, method.client_privacy)
        self.assertEqual(method.accountants, {})

        state = _state(_TinyClassifier())
        method.begin_round(1, [0])
        first = method.train_client(
            client_id=0,
            model_fn=_TinyClassifier,
            global_state=state,
            train_loader=loader,
            device=torch.device("cpu"),
        )
        self.assertIsNone(first.metadata["epsilon"])
        self.assertEqual(first.metadata["accountant_committed_steps"], 0)
        state = method.aggregate([first])
        method.observe_global_accuracy(state, 0.1)

        method.begin_round(2, [0])
        second = method.train_client(
            client_id=0,
            model_fn=_TinyClassifier,
            global_state=state,
            train_loader=loader,
            device=torch.device("cpu"),
        )
        self.assertEqual(second.metadata["adapl_update_phase"], "decay")
        self.assertEqual(second.metadata["fisher_important_ratio"], 1.0)
        self.assertEqual(second.metadata["noise_l2"], 0.0)

    def test_unclipped_noiseless_path_reports_diagnostics(self) -> None:
        torch.manual_seed(7)
        global_state = _state(_TinyClassifier())
        result = local_update_first(
            model_fn=_TinyClassifier,
            global_state=global_state,
            train_loader=_loader(),
            local_steps=1,
            local_epochs=None,
            local_update_mode="random-batch",
            lr=0.05,
            momentum=0.0,
            weight_decay=0.0,
            device=torch.device("cpu"),
            clipping_bound=1.0,
            base_noise_multiplier=1.0,
            gamma=0.0,
            prox_mu=0.0,
            enable_clipping=False,
            enable_noise=False,
        )

        self.assertEqual(result.actual_minibatch_steps, 1)
        self.assertEqual(result.sample_clip_factor_mean, 1.0)
        self.assertEqual(result.sample_clip_fraction, 0.0)
        self.assertGreater(result.sample_grad_norm_p90, 0.0)
        self.assertEqual(result.noise_std_mean, 0.0)
        self.assertEqual(result.noise_l2_mean, 0.0)
        self.assertEqual(result.noise_to_signal_ratio_mean, 0.0)

    def test_all_noise_scope_ignores_zero_fisher_mask(self) -> None:
        stats = {"weight": LayerNoiseStats(sigma=1.0, std=0.25)}
        masks = {"weight": torch.zeros(4, dtype=torch.bool)}

        fisher_grads = {"weight": torch.zeros(4)}
        _, _, _, _, fisher_noise_l2 = _apply_masked_noise(
            fisher_grads,
            masks,
            stats,
            "fisher",
        )
        self.assertEqual(fisher_noise_l2, 0.0)
        self.assertTrue(torch.equal(fisher_grads["weight"], torch.zeros(4)))

        torch.manual_seed(11)
        all_grads = {"weight": torch.zeros(4)}
        _, _, _, _, all_noise_l2 = _apply_masked_noise(
            all_grads,
            masks,
            stats,
            "all",
        )
        self.assertGreater(all_noise_l2, 0.0)
        self.assertFalse(torch.equal(all_grads["weight"], torch.zeros(4)))

    def test_freeze_bn_preserves_running_statistics(self) -> None:
        torch.manual_seed(13)
        global_state = _state(_TinyBatchNormClassifier())
        common = dict(
            model_fn=_TinyBatchNormClassifier,
            global_state=global_state,
            train_loader=_loader(),
            local_steps=1,
            local_epochs=None,
            local_update_mode="random-batch",
            lr=0.05,
            momentum=0.0,
            weight_decay=0.0,
            device=torch.device("cpu"),
            clipping_bound=100.0,
            base_noise_multiplier=1.0,
            gamma=0.0,
            prox_mu=0.0,
            enable_noise=False,
        )

        updating = local_update_first(**common, freeze_batch_norm=False)
        frozen = local_update_first(**common, freeze_batch_norm=True)

        initial_mean = global_state["bn.running_mean"]
        self.assertFalse(torch.equal(updating.state_dict["bn.running_mean"], initial_mean))
        self.assertTrue(torch.equal(frozen.state_dict["bn.running_mean"], initial_mean))
        self.assertEqual(
            int(frozen.state_dict["bn.num_batches_tracked"].item()),
            int(global_state["bn.num_batches_tracked"].item()),
        )

    def test_reporting_keeps_header_and_row_aligned(self) -> None:
        metrics = {
            "adapl_sample_grad_norm_mean": 4.0,
            "adapl_sample_grad_norm_p50_mean": 3.0,
            "adapl_sample_grad_norm_p90_mean": 7.0,
            "adapl_sample_grad_norm_p99_mean": 9.0,
            "adapl_clip_fraction_mean": 0.75,
            "adapl_signal_l2_mean": 2.0,
            "adapl_noise_l2_mean": 8.0,
            "adapl_noise_to_signal_ratio_mean": 4.0,
            "adapl_fisher_important_ratio_mean": 0.2,
            "adapl_fisher_important_ratio_min": 0.1,
            "adapl_fisher_important_ratio_max": 0.3,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/metrics.csv"
            init_output_csv(path)
            append_output_csv(
                path=path,
                round_idx=1,
                selected_clients=[0],
                train_loss=1.0,
                test_loss=1.0,
                test_accuracy=0.1,
                round_metrics=metrics,
            )
            with open(path, newline="") as csvfile:
                rows = list(csv.reader(csvfile))

        self.assertEqual(len(rows), 2)
        self.assertEqual(len(rows[0]), len(rows[1]))
        self.assertEqual(
            rows[1][rows[0].index("adapl_noise_to_signal_ratio_mean")],
            "4.000000",
        )


if __name__ == "__main__":
    unittest.main()
