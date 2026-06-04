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

Datasets are not tracked by Git. The repository ignores `data/`, so local
dataset caches will not be pushed to GitHub.

Default server dataset cache path:

```text
/root/autodl-tmp/data/
```

`torchvision.datasets.CIFAR100` uses this directory as its `root`, so the
downloaded/extracted CIFAR-100 files will live under:

```text
/root/autodl-tmp/data/cifar-100-python/
```

The code reference is `--data_dir`, whose default is the AutoDL path above.
You can override it with a command-line argument or with `ADAPL_DATA_DIR`:

```bash
python main.py --method PF --data_dir /path/to/server/datasets/cifar100
```

Experiment outputs are also ignored by Git and default to:

```text
/root/autodl-tmp/results/
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
  --privacy_scenario 3 \
  --delta 1e-5 \
  --clipping_norm 1.0
```

The current DP module clips each client's model update, adds Gaussian noise,
and then aggregates privatized client updates. The default noise multiplier is
computed from the classical single-release Gaussian mechanism bound when
`--noise_multiplier` is not provided. Tighter multi-round accounting should be
added under `adapl/privacy/accounting.py` when the experimental protocol needs
formal composed privacy reports.
For ResNet models with BatchNorm, DP noise is applied to trainable parameters
only; non-trainable BatchNorm buffers such as `running_var` are not noised to
avoid invalid negative variances during evaluation.

For the paper privacy-level setup, levels map to maximum budgets
`0.5/1.0/2.0/4.0/8.0`. In `Min`, the strict uniform budget is
`epsilon_min = min_{k in K_t} epsilon_k`, where `K_t` is the set of clients
sampled in the current round. With 20 clients and sample rate 0.8, each round
trains 16 clients.

## Run The Min Stability Grid

The script below runs this grid sequentially. It defaults to 50 rounds per
combination for stability screening; once a configuration is promising, rerun
that configuration for 300 rounds.

```text
epsilon: 16, 8, 4
lr: 0.01, 0.005
local_steps: 5, 10
clipping_norm: 0.1, 0.5, 1.0
```

Start it in a screen session on the server:

```bash
screen -S min_grid
bash scripts/run_min_grid.sh
```

Results are written under `/root/autodl-tmp/results/min_grid/`. A rolling
summary is saved to `/root/autodl-tmp/results/min_grid/summary.csv`.

To change the screening length:

```bash
GLOBAL_ROUNDS=100 bash scripts/run_min_grid.sh
```

To run a smaller subset:

```bash
EPSILONS="8" LRS="0.005" LOCAL_STEPS_GRID="5" CLIPPING_NORMS="0.5" \
  bash scripts/run_min_grid.sh
```
