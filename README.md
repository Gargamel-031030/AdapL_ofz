# AdapL Federated Learning Experiments

This project is structured for CIFAR-100 federated learning experiments.
Implemented methods include privacy-free FedAvg, DP-FedAvg baselines, and
FedDPA-style personalized DP-FL with ResNet18.

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
- `weiavg`, `weightedavg`, `weighted-avg`: budget-weighted FedAvg baseline
  where selected client deltas are aggregated with
  `epsilon_i / sum(epsilon_selected)`.
- `feddpa`: FedDPA-style personalized DP-FL. Each selected client estimates a
  diagonal Fisher mask, keeps high-importance parameters as persistent local
  personalized state, trains personalized/shared parameter subsets separately,
  and applies WeiAvg-style per-minibatch gradient clipping and Gaussian noise
  before uploading the locally trained state for FedAvg aggregation.
- `pfa`: Projected Federated Averaging with heterogeneous privacy budgets.
  High-epsilon clients define the public update subspace, and private-client
  updates are projected through that subspace before aggregation. The default
  aggregation is epsilon-weighted, matching the PFA+WeiAvg setting.

The following methods are registered as planned extension points but are not
implemented yet: `ppfed`, `efl`, `adapl`.

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
`--noise_multiplier` is not provided.
For ResNet models with BatchNorm, DP noise is applied to trainable parameters
only; non-trainable BatchNorm buffers such as `running_var` are not noised to
avoid invalid negative variances during evaluation.

For the paper privacy-level setup, levels map to maximum budgets
`0.5/1.0/2.0/4.0/8.0`. In `Min`, the strict uniform budget is
`epsilon_min = min_{k in K_t} epsilon_k`, where `K_t` is the set of clients
sampled in the current round. With 20 clients and sample rate 0.8, each round
trains 16 clients.

## Privacy Budget Accounting

Privacy consumption is maintained by a shared per-client accountant in
`adapl/privacy/accounting.py`, not inside WeiAvg or PFA-specific logic. Before
each round, the runner prechecks every client budget and samples only clients
whose next local training block still fits within their maximum epsilon. After
local training completes, the runner updates each selected client's accumulated
budget and step count. WeiAvg uses epsilon for server aggregation weights; PFA
uses epsilon to classify public/private clients and, by default, to weight
projected aggregation.

Use `--privacy_accounting auto|on|off` to control this process. The boolean
aliases `--use_privacy_accounting` and `--no_privacy_accounting` map to `on`
and `off`.

- `auto` keeps each method's default behavior.
- `on` forces per-client accounting for any method, including PF or `--no_dp`
  runs, using `--privacy_budgets`, `--privacy_scenario`, `--epsilon_max`, or
  `--epsilon_min` as the client maximum budgets.
- `off` disables budget precheck/filter/update while leaving DP clipping/noise
  behavior unchanged.

## Run The WeiAvg Baseline

`WeiAvg` uses heterogeneous client privacy budgets. Pass either a paper
privacy-level scenario or an explicit budget list/file:

```bash
python main.py \
  --method weiavg \
  --privacy_scenario 3
```

For a selected client set `K_t`, the server computes
`weight_i = epsilon_i / sum_{j in K_t} epsilon_j` and applies the weighted
client update. Use `--method pf` with the same FL/data hyperparameters for the
ordinary FedAvg comparison.

## Run The FedDPA Baseline

`FedDPA` requires DP calibration through `--epsilon_min`,
`--privacy_scenario`, `--privacy_budgets`, or `--noise_multiplier`:

```bash
python main.py \
  --method feddpa \
  --privacy_scenario 3 \
  --feddpa_fisher_threshold 0.4 \
  --feddpa_fisher_batches 1
```

Use `--feddpa_fisher_batches 0` to estimate Fisher masks over each selected
client's full loader. The default `1` is a practical approximation for faster
CIFAR-100 screening. FedDPA applies DP noise during local training after each
minibatch backward pass, using
`noise_std = noise_multiplier * clipping_norm / batch_size`, matching the
current WeiAvg DP path.

## Run The PFA Baseline

`PFA` requires heterogeneous client privacy budgets. It uses the highest-budget
clients as public clients, retries sampling to include both public and private
clients when possible, and projects private updates through the public update
subspace:

```bash
python main.py \
  --method pfa \
  --privacy_scenario 3 \
  --pfa_projection_dim 1 \
  --pfa_public_fraction 0.1
```

Use `--no-pfa_weighted_projection` for count-weighted projected averaging.

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
