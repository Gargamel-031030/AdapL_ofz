# AdapL Federated Learning Experiments

This project is structured for CIFAR-100 federated learning experiments.
The current implemented method is `PF` / `PrivacyFree`, a no-DP FedAvg
baseline with ResNet18.

## Project Layout

- `main.py`: unified experiment entry point.
- `baseline_fedavg.py`: compatibility wrapper for the old script name.
- `adapl/cli.py`: command-line arguments and `main()` dispatch.
- `adapl/experiment.py`: shared experiment runner.
- `adapl/data/`: dataset loading, partitioning, and client label summaries.
- `adapl/models/`: model builders.
- `adapl/fl/`: common FL utilities such as client training, aggregation,
  sampling, and evaluation.
- `adapl/methods/`: method registry and algorithm-specific logic.

## Data And Output Paths

Datasets are not tracked by Git. The repository ignores `data/`, so CIFAR-100
will be downloaded on the server after you pull the project.

Default dataset cache path:

```text
<project_root>/data/cifar100/
```

`torchvision.datasets.CIFAR100` uses this directory as its `root`, so the
downloaded/extracted CIFAR-100 files will live under:

```text
<project_root>/data/cifar100/cifar-100-python/
```

The code reference is `--data_dir`, whose default is the absolute project-root
path above. You can override it when needed:

```bash
python main.py --method PF --data_dir /path/to/server/datasets/cifar100
```

Experiment outputs are also ignored by Git and default to:

```text
<project_root>/results/
```

## Implemented Methods

- `pf`, `privacyfree`, `fedavg`: privacy-free FedAvg baseline.
- `min`, `minimum`: DP-FedAvg baseline where all clients use the strictest
  privacy budget `epsilon_min`.

The following methods are registered as planned extension points but are not
implemented yet: `feddpa`, `ppfed`, `weiavg`, `pfa`, `efl`, `adapl`.

## Run A Smoke Test

```bash
python main.py \
  --method pf \
  --global_rounds 1 \
  --limit_train_samples 200 \
  --limit_test_samples 100 \
  --num_workers 0 \
  --output_csv results/smoke_pf.csv
```

## Run The Default PF Baseline

```bash
python main.py --method pf
```

Outputs are written under `results/` by default.

## Run The Minimum DP Baseline

```bash
python main.py \
  --method Min \
  --epsilon_min 2.0 \
  --delta 1e-5 \
  --clipping_norm 1.0
```

The current DP module clips each client's model update, adds Gaussian noise,
and then aggregates privatized client updates. The default noise multiplier is
computed from the classical single-release Gaussian mechanism bound when
`--noise_multiplier` is not provided. Tighter multi-round accounting should be
added under `adapl/privacy/accounting.py` when the experimental protocol needs
formal composed privacy reports.
