# Motion Plausibility Project Architecture

This project has two related goals:

1. Build trajectory datasets from raw location logs.
2. Train plausibility models and visualize their anomaly scores on a map.

The architecture should keep one clear source of truth for data and treat model inputs,
predictions, and the visualization database as derived artifacts.

## Directory Layout

```text
data/
  raw/
    iNaturalist/
      frequent-poster.log
    gowalla/
      gowalla_frequent_poster.log

  processed_parquet/
    inat/
      observations.parquet
      transitions.parquet
      trajectories.parquet
      trajectory_transitions.parquet
    gowalla/
      observations.parquet
      transitions.parquet
      trajectories.parquet
      trajectory_transitions.parquet
    synthetic/
      observations.parquet
      transitions.parquet
      trajectories.parquet
      trajectory_transitions.parquet
      synthetic_metadata.parquet
      generation_config.json

  features/
    motion_v1/
      inat/
      gowalla/
      synthetic/

artifacts/
  scalers/
  stats/
  models/
  predictions/
  runs/
  app/
    plausibility.db
```

The current repository still contains legacy folders such as `Preprocessed_Dataset/`,
`Models_weights_and_results/`, JSON exports, and dataset PKLs. Those should be
gradually replaced by the structure above.

## Data Flow

```text
raw logs
  -> canonical preprocessing
  -> processed Parquet tables
  -> model-specific filtering
  -> feature extraction and scaling
  -> model training
  -> continuous model scores and explanations
  -> SQLite app database
  -> interactive visualization
```

## Canonical Tables

Canonical Parquet tables store physical, unscaled data. They should not store
normalized model inputs.

Preprocessing retains every structurally valid transition and every trajectory with at
least one transition. Requirements such as a minimum sequence length belong to the
cleaning/model-selection stage, not canonical preprocessing.

### observations.parquet

One row per location point.

```text
dataset
user_id
observation_id
timestamp
lat
lon
is_obscured
```

### transitions.parquet

One row per movement between two observations.

```text
dataset
transition_id
observation_id1
observation_id2
speed_kmh
elapsed_time_s
distance_m
acceleration_m_s2
bearing_change_rad
transition_plausibility
plausibility_reason
```

`date`, `user_id`, and `trajectory_id` are intentionally not stored here. They are
derivable from observations and `trajectory_transitions`.

### trajectories.parquet

One row per trajectory.

```text
dataset
trajectory_id
user_id
n_transitions
start_observation_id
end_observation_id
```

The trajectory date is derivable from `start_observation_id -> observations.timestamp`.
The app database may later denormalize it for faster timeline queries.

### trajectory_transitions.parquet

Join table preserving transition order inside each trajectory.

```text
dataset
trajectory_id
transition_order
transition_id
```

This avoids storing transition lists inside trajectory rows and keeps the data easy
to query from Parquet or SQL.

## Model Input Pipeline

This project currently uses unsupervised anomaly detection. There is no ordinary
supervised train/test split for the real datasets. The real datasets provide the
normal/reference population after rule-based filtering. Synthetic anomalies are kept
separate and used only for testing and evaluation.

Cleaning and scaling are separate decisions, but they do not require storing a second
full copy of cleaned raw data. The model pipeline reads canonical Parquet, applies its
filtering policy, fits the scaler on valid real data only, and transforms the selected
rows:

```text
processed Parquet
  -> filter valid rows/trajectories
  -> fit scaler on valid real data
  -> transform valid real data
  -> transform synthetic data with the same scaler
  -> train or evaluate model
```

Feature datasets may be saved when they are the direct input to several model runs.
They are still derived artifacts, not the source of truth. They should be versioned by
feature set:

```text
data/features/<feature_set>/<dataset>/
  transition_features.parquet
  trajectories.parquet
```

For reproducibility, store lightweight metadata about filtering instead of duplicating
the cleaned raw dataset:

```text
artifacts/runs/<feature_set>/
  config.json
  exclusions.parquet
```

`exclusions.parquet` records rejected trajectory identifiers plus the filtering reason.
It does not contain duplicated physical feature values.

The scaler is a model artifact. If a model is trained on both iNaturalist and Gowalla,
its scaler must be fitted on the combined valid training population and then applied
consistently to both datasets and to synthetic anomalies.

## Synthetic Anomaly Generation

Synthetic anomalies are generated from clean canonical real trajectories before
normalization:

```text
clean processed Gowalla trajectories
  -> deterministic sampling and physical anomaly injection
  -> canonical synthetic Parquet
  -> motion_v1 transformation with real-data scalers
  -> model evaluation and benchmark metrics
```

The generator has no model, scaler, JSON, or visualization dependency:

```bash
python -m dataset_preprocessing.synthetic_anomalies
```

By default it creates 50 trajectories for each profile:

```text
back_and_forth
abnormal_speed
initial_fix_jump
```

Generated rows follow the canonical schemas. Speed and acceleration are recomputed
after injection, locations are propagated from distance and bearing, and timestamps
advance by elapsed time.

Synthetic-only ground truth and provenance stay in a separate lightweight table:

```text
data/processed_parquet/synthetic/synthetic_metadata.parquet

dataset
trajectory_id
source_dataset
source_trajectory_id
anomaly_profile
anomaly_start_order
anomaly_end_order
generation_seed
```

This avoids adding synthetic-only columns to canonical trajectories. Generation is
deterministic for a fixed seed and replaces the output only after all new files have
been written successfully.

## Model Artifacts

Models should be stored by `model_id`, not by dataset.

```text
artifacts/models/<model_id>/
  config.json
  model.pth                 # PyTorch models
  model.pkl                 # sklearn/classical models, if needed
  training_history.parquet
```

Predictions should be stored by model and dataset.

```text
artifacts/predictions/<model_id>/<dataset>/
  trajectory_scores.parquet
  transition_scores.parquet
```

Models do not need to expose identical internal outputs. They need a small common
scoring contract for comparison and optional explanation tables for model-specific
details.

## Model Framework

Model code lives outside data preprocessing:

```text
modeling/
  base.py
  registry.py
  config.py
  data.py
  losses.py
  torch_training.py
  evaluation.py
  cli.py
  benchmark.py
  models/
    lstm_autoencoder.py
  configs/
    lstm_benchmark.json
```

Each model family is a `ModelPlugin` registered by a stable `model_type`. A plugin owns
model-specific training, serialization, loading, and prediction adaptation. Shared
data loading, masking, output paths, and score conventions remain outside the model.

This boundary supports future attention-based networks and classical models without
putting their logic into one training script. A classical model can implement the same
plugin interface while saving `model.pkl`; it does not need to use PyTorch helpers.

Custom PyTorch reconstruction losses are registered by name in `modeling/losses.py`
and selected in the run config. Architecture parameters are model-specific and live
under the config's `architecture` key.

## Score Contract

The visualization must be able to change the anomaly percentage dynamically. Therefore
predictions store continuous scores, not a permanent `is_unplausible` label or one
fixed threshold.

Trajectory scores:

```text
model_id
dataset
trajectory_id
score
```

Transition scores, when supported:

```text
model_id
dataset
trajectory_id
transition_order
transition_id
score
```

Higher scores must consistently mean "more anomalous". If a model naturally uses the
opposite direction, its prediction adapter must reverse or transform the score.

When the user selects an anomaly percentage or percentile, the app computes the
threshold from the relevant trajectory-score distribution and classifies trajectories
at query time:

```text
threshold = percentile(model scores, selected percentile)
is_unplausible = score >= threshold
```

The UI-derived label and threshold may be returned by the API, but they are not stored
as the canonical model prediction.

The threshold population must be explicit in the model/run configuration, for example:

```text
combined valid training population
iNaturalist evaluation population
Gowalla evaluation population
synthetic evaluation population
```

This prevents the meaning of a selected percentile from changing silently between
datasets.

## Model Explanations

Explanation output is optional and capability-based because different model families
produce different diagnostics.

The current LSTM implementation stores reconstruction explanations as wide columns in
the score tables:

```text
error_speed
error_elapsed_time
error_distance
error_acceleration
error_bearing_sin
error_bearing_cos
```

This is more compact than repeating identifiers in one row per entity and feature.
Both trajectory and transition tables contain these columns. A trajectory feature
error is the mean of that feature's transition errors over the configured scoring
window. The trajectory `score` is the mean of its feature errors.

Other models may expose different explanation columns or no explanations. Their plugin
must declare capabilities so the database importer and visualization know which
outputs exist. A separate long-form details table remains available when explanations
are sparse or cannot be represented efficiently as one fixed set of columns:

```text
artifacts/predictions/<model_id>/<dataset>/details.parquet
```

The model configuration declares capabilities such as:

```text
trajectory_scores: true
transition_scores: true
feature_explanations: true
explanation_type: reconstruction_error
```

This gives the visualization a stable common interface without forcing every model
into an LSTM-specific schema.

Prediction writing is batch-streamed to Parquet. Evaluation memory therefore scales
with the configured batch size rather than the total number of Gowalla transitions.

## Visualization Database

`plausibility.db` is an app-serving cache. It can duplicate some fields from Parquet
when that makes the frontend simpler or faster.

The database should be rebuildable from:

```text
processed Parquet
model run metadata
prediction Parquet
```

This keeps the website fast without making SQLite the research data source of truth.

The current visualization cache intentionally stores only:

```text
dataset counts
searchable user summaries
trajectory timeline metadata
model configurations
model/dataset threshold curves
```

Canonical observations, transitions, and trajectory mappings remain in processed
Parquet. Prediction scores remain in prediction Parquet. The API filters those files
only for the trajectories currently displayed. This keeps `plausibility.db` small
enough for a laptop and avoids duplicating millions of transition and score rows.

Build the cache:

```bash
python build_database.py
```

Run the visualization:

```bash
python open_visualization.py
```

Then open `http://127.0.0.1:8001`. The map uses online OpenStreetMap tiles; geographic
tiles are not downloaded or stored by the project.

The frontend depends exclusively on the canonical architecture and exposes:

```text
dataset and searchable user selection
activity timeline grouped by date
model selection with training and architecture metadata
continuous anomaly-percentage threshold control
trajectory and transition reconstruction scores
feature-wise reconstruction errors when supported
Motion Plausibility Profile view synchronized with the map
speed-based colors on map paths and profile points
observation and transition hover diagnostics
day/month/year/smart timeline modes
progressive path animation and color-deficient palette
```

## Storage Rules

Keep:

```text
data/raw/
data/processed_parquet/
data/features/
artifacts/models/
artifacts/scalers/
artifacts/predictions/
artifacts/runs/
```

Treat as rebuildable:

```text
data/features/
artifacts/app/plausibility.db
```

Remove from the core pipeline:

```text
dataset PKLs
visualization JSON exports
ad hoc generated plots
```

PKL remains acceptable for Python-specific artifacts such as sklearn scalers and
classical models. PyTorch weights should stay as `.pth`. Datasets and predictions
should use Parquet.

## Current Preprocessing Entry Point

The merged preprocessing script is:

```bash
python Dataset_Preprocessing/preprocess_logs_to_parquet.py
```

By default it reads:

```text
data/raw/iNaturalist/frequent-poster.log
data/raw/gowalla/gowalla_frequent_poster.log
```

If those files do not exist yet, it falls back to the current legacy log locations:

```text
data/processed/iNaturalist/frequent-poster.log
data/processed/gowalla/gowalla_frequent_poster.log
```

Explicit inputs can be passed with:

```bash
python Dataset_Preprocessing/preprocess_logs_to_parquet.py \
  --dataset inat=data/raw/iNaturalist/frequent-poster.log \
  --dataset gowalla=data/raw/gowalla/gowalla_frequent_poster.log
```

## Current Feature Entry Point

The current feature set is `motion_v1`.

Generate the canonical synthetic anomaly dataset:

```bash
python -m dataset_preprocessing.synthetic_anomalies
```

Then build normalized features for iNaturalist, Gowalla, and synthetic anomalies:

```bash
python -m dataset_preprocessing.build_motion_features
```

This writes:

```text
data/features/motion_v1/<dataset>/transition_features.parquet
data/features/motion_v1/<dataset>/trajectories.parquet
artifacts/scalers/motion_v1.pkl
artifacts/runs/motion_v1/config.json
artifacts/runs/motion_v1/exclusions.parquet
```

`motion_v1` uses these features:

```text
speed
elapsed_time
distance
acceleration
bearing_sin
bearing_cos
```

The scaler is fitted only on valid real trajectories from `inat` and `gowalla`.
Synthetic anomalies are transformed with that fitted scaler and are marked
`use_for_training = false`.

## Current Stats Entry Point

Dataset statistics can be extracted from canonical Parquet with:

```bash
python Dataset_Preprocessing/parquet_dataset_stats.py
```

This writes:

```text
artifacts/stats/processed_parquet/summary.json
artifacts/stats/processed_parquet/summary.txt
artifacts/stats/processed_parquet/histograms/*.csv
```

The report includes transition distributions for:

```text
speed
acceleration
distance
elapsed_time
bearing_change
```

It also includes the distribution and average of `n_transitions` per trajectory.
Each dataset is reported twice:

```text
all
clean_real
```

`clean_real` uses the same filtering policy as `motion_v1`: trajectories must have at
least two transitions and every transition must be rule-plausible.

## Current Model Entry Points

List registered model families and their capabilities:

```bash
python -m modeling.cli list-models
```

Train the default LSTM autoencoder on valid iNaturalist and Gowalla trajectories:

```bash
python -m modeling.cli train --model-id lstm_motion_v1
```

Evaluate the saved model on both real datasets and synthetic anomalies:

```bash
python -m modeling.cli evaluate --model-id lstm_motion_v1
```

Train and evaluate in one command:

```bash
python -m modeling.cli run --model-id lstm_motion_v1
```

The defaults can be overridden by a JSON config passed with `--config`. Every resolved
run config is copied to `artifacts/models/<model_id>/config.json`, so evaluation uses
the exact feature list and architecture used for training. The training seed is also
stored there and controls Python, NumPy, PyTorch, and DataLoader shuffling.

Run the replacement for the old hard-coded LSTM benchmark:

```bash
python -m modeling.benchmark modeling/configs/lstm_benchmark.json
```

The benchmark JSON defines feature groups, epoch counts, and batch sizes. Each
combination receives a unique `model_id` and the same standard artifact structure.
It also computes ROC AUC and synthetic recall at configured anomaly percentages,
including recall per anomaly profile. Thresholds are derived only from clean real
reference scores and are written to:

```text
artifacts/predictions/<model_id>/synthetic/metrics.json
```

Metrics can be regenerated from existing predictions:

```bash
python -m modeling.synthetic_evaluation --model-id lstm_motion_v1
```

The initial registered model is `lstm_autoencoder`. It trains only on trajectories
marked `use_for_training = true`, which excludes synthetic anomalies. Evaluation
includes `inat`, `gowalla`, and `synthetic` by default.

The following scripts are no longer model-training entry points:

```text
Dataset_Preprocessing/trajectory_prediction_LSTM.py
Dataset_Preprocessing/benchmark_LSTM.py
```

They are legacy code. Synthetic generation no longer imports either script. New
training and benchmark work must use the `modeling` package.

`Dataset_Preprocessing/generate_synthetic_anomalies.py` is now only a thin entry point
for `dataset_preprocessing.synthetic_anomalies`; it contains no model or generation
logic.
