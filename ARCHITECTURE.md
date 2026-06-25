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
    motion_v2/
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
acceleration_valid
bearing_change_rad
bearing_change_valid
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

### Trajectory Sessionization

A calendar day is not automatically one coherent movement trajectory. Canonical
preprocessing starts a new trajectory when two consecutive observations are separated
by more than six hours:

```text
--max-inactivity-hours 6
```

The boundary is configurable, and `0` disables it. The transition spanning the gap is
not stored as movement because the path taken during that interval is unknown. The
observation after the gap becomes the first observation of the next session, so its
first modeled transition has undefined acceleration and bearing change.

Synthetic generation imports the same sessionization policy. If an injected sequence
contains an inactivity boundary, only the session containing the injected anomaly is
kept as that synthetic benchmark trajectory. This preserves one anomaly label per
synthetic trajectory and prevents a profile label from being attached to unrelated
session fragments.

## Model Input Pipeline

This project uses unsupervised anomaly detection, but unsupervised learning still
requires separate fitting and calibration populations. Split real trajectories by
user (or by time within user) before fitting scalers or models:

```text
real fit population         fit scaler and model
real calibration population choose thresholds and measure score drift
synthetic benchmark         measure controlled anomaly detection
trusted reviewed sample     estimate real false positives
```

Synthetic anomalies must not select hyperparameters and then also serve as the only
reported final benchmark. Keep either repeated benchmark seeds or a final untouched
synthetic generation seed.

The fit/calibration separation is unsupervised: neither real subset needs anomaly
labels. The split prevents threshold calibration on reconstruction errors from the
same trajectories used to optimize the model. Prefer assigning complete users to one
subset so trajectories from one person's repeated behavior do not leak across both.
If user separation is impractical, use an earlier time period for fitting and a later
period for calibration.

The current implementation assigns complete users deterministically:

```text
80% of users -> fit
20% of users -> calibration
```

The split seed and fraction are stored in the feature run configuration. iNaturalist
user `98904` and Gowalla user `16936` are explicitly reserved for calibration because
they contain manually reviewed plausible trajectories. Scalers are fitted only on fit
transitions, models train only on fit trajectories, and threshold curves use only
calibration trajectories.

Cleaning and scaling are separate decisions, but they do not require storing a second
full copy of cleaned raw data. The model pipeline reads canonical Parquet, applies its
filtering policy, fits the scaler on valid real data only, and transforms the selected
rows:

```text
processed Parquet
  -> filter valid rows/trajectories
  -> save one unscaled transformed feature set
  -> select each model's real fit population
  -> fit that model's scaler
  -> scale real and synthetic model inputs in memory
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
clean held-out iNaturalist and Gowalla calibration trajectories
  -> deterministic sampling and physical anomaly injection
  -> canonical synthetic Parquet
  -> motion_v2 unscaled feature transformation
  -> model-specific robust scaling fitted on real fit trajectories
  -> model evaluation and benchmark metrics
```

Regenerating the synthetic dataset invalidates every existing synthetic prediction
and metrics file, even when trajectory identifiers are reused. The visualization
database excludes synthetic predictions older than the current synthetic feature
tables; each model must be reevaluated before its scores reappear for that dataset.

The generator has no model, scaler, JSON, or visualization dependency:

```bash
python -m dataset_preprocessing.synthetic_anomalies
```

By default it creates 50 trajectories per source dataset for each profile:

```text
back_and_forth
abnormal_speed
initial_fix_jump
```

`back_and_forth` creates several ordinary-speed traversals between two nearby
locations, producing repeated direction reversals rather than one isolated U-turn.
`initial_fix_jump` models a camera GPS receiver starting from a stale saved position:
the first transition remains near that stale fix, followed by a fast correction to
the current position (reproduction of hardware bugs that can happen on some devices).

Generated rows follow the canonical schemas. Speed and acceleration are recomputed
after injection, locations are propagated from distance and bearing, and timestamps
advance by elapsed time.

After injection and session trimming, every synthetic transition is checked again
with the same deterministic plausibility policy used for real preprocessing. A
generated trajectory with any rule violation is retried and ultimately rejected
rather than written to the synthetic benchmark. This keeps synthetic profiles as ML
benchmark anomalies, not deterministic-rule positives.

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

The current feature set is `motion_v2`.

Generate the canonical synthetic anomaly dataset:

```bash
python -m dataset_preprocessing.synthetic_anomalies
```

Then build unscaled transformed features for iNaturalist, Gowalla, and synthetic
anomalies:

```bash
python -m dataset_preprocessing.build_motion_features
```

This writes:

```text
data/features/motion_v2/<dataset>/transition_features.parquet
data/features/motion_v2/<dataset>/trajectories.parquet
artifacts/runs/motion_v2/config.json
artifacts/runs/motion_v2/exclusions.parquet
```

`motion_v2` stores these transformed features:

```text
speed                     log1p(km/h)
speed_stationary          speed == 0
speed_0_5                 0 < speed < 5
speed_5_10                5 <= speed < 10
speed_10_25               10 <= speed < 25
speed_25_80               25 <= speed < 80
speed_80_140              80 <= speed < 140
speed_140_200             140 <= speed < 200
speed_200_plus            speed >= 200
elapsed_time              log1p(seconds)
distance                  log1p(metres)
acceleration              signed log1p(m/s^2)
acceleration_valid
bearing_sin
bearing_cos
bearing_valid
```

These Parquet files are deliberately not normalized. At training time each model fits
a `RobustScaler` only on the fit trajectories selected by that model's dataset and
sampling configuration. The artifact is saved as:

```text
artifacts/models/<model_id>/scaler.pkl
```

The same scaler is loaded for real calibration trajectories and synthetic benchmark
trajectories. This gives iNaturalist-only, Gowalla-only, combined, and balanced models
population-specific preprocessing without duplicating normalized Parquet datasets.
Only continuous log features are scaled; MPP indicators and bearing validity/angle
representations are left in their natural ranges.

Current comparison groups train this MPP-based input combination:

```text
MPP speed ranges + speed + acceleration + distance + elapsed_time + bearing_sin + bearing_cos + bearing_valid
```

The previous `elapsed_time + distance + bearing` combination is intentionally removed
from new matrices because it underuses speed-regime information. Bearing-free and
acceleration-free comparisons are handled by score modes instead of extra training
runs.

The forecasting experiment adds a richer motion combination:

```text
MPP speed ranges
speed                     log1p(km/h)
acceleration              signed log1p(m/s^2)
acceleration_valid
bearing_sin
bearing_cos
bearing_valid
distance                  log1p(metres)
elapsed_time              log1p(seconds)
```

This keeps the categorical MPP speed regimes while also exposing continuous speed and
local dynamics. It is intended for local next-step prediction rather than whole
trajectory reconstruction.

For models that already expose feature-wise errors, some feature ablations can be
performed without retraining by changing the anomaly-score formula. These modes reuse
the stored `error_*` columns:

```text
mean
weighted
mean_no_bearing
weighted_no_bearing
mean_no_acceleration
weighted_no_acceleration
```

This is not equivalent to training a model without that input feature. The model still
saw the feature during training and prediction. It only removes that feature group
from the final anomaly score used for thresholds, metrics, and visualization.

## Configurable Model Matrix

The current model matrix is defined in:

```text
modeling/configs/configurable_model_matrix.json
```

It trains 8 model artifacts from:

```text
4 training populations
  iNaturalist
  Gowalla
  combined
  combined weighted (balanced trajectory sampling)

1 feature combination
  MPP speed ranges + speed + acceleration + distance + elapsed time + bearing

2 architectures
  LSTM autoencoder
  LSTM next-step forecaster
```

Run the matrix with:

```bash
python -m modeling.experiments modeling/configs/configurable_model_matrix.json
```

The following score choices do not require additional trained models:

```text
transition error: feature mean, concept weighted, and variants excluding bearing or acceleration
trajectory error: mean, top 3 mean, top 1, or maximum 5-step window
transition 0: included or excluded
reference tail: 0.1% to 25%
```

Concept-weighted transition scoring gives equal weight to physical concepts rather
than columns. All MPP speed-bin errors form one speed concept; bearing sine, cosine,
and validity form one bearing concept; distance and elapsed time each form one
concept. Feature-wise errors remain stored once and all score combinations are
computed from them.

Each scaler is fitted only on the valid real fit trajectories selected by that model.
Training loaders fail before fitting if any selected trajectory overlaps:

```text
deterministic-rule exclusions
calibration/benchmark split rows
manually reviewed plausible trajectories
source trajectories used to generate synthetic anomalies
synthetic anomaly rows
```

Synthetic anomalies are transformed with the fitted model scaler and are marked
`use_for_training = false`.

The leakage boundary is enforced in three places:

```text
Dataset_Preprocessing/preprocess_logs_to_parquet.py
  computes deterministic transition_plausibility

Dataset_Preprocessing/build_motion_features.py
  drops every trajectory with at least one implausible transition before feature files are written
  assigns user-level fit/calibration split
  writes use_for_training/use_for_calibration flags

modeling/data.py
  training loaders read only use_for_training rows
  training loaders fail if selected rows overlap deterministic exclusions,
  calibration rows, trusted plausible labels, synthetic source controls, or synthetic rows
```

Scaler fitting uses the same training-only dataset object as model fitting. The order
inside each model plugin is:

```text
load training_only=True dataset
fit RobustScaler on that dataset only
apply scaler to that dataset
fit model
load evaluation datasets separately
apply saved scaler to evaluation datasets
```

Run this audit before reporting results or pushing a major pipeline change:

```bash
python -m modeling.data_leakage_audit
```

The audit checks the materialized feature files, deterministic-rule exclusions,
synthetic source controls, manually reviewed plausible labels, and active experiment
configs. It does not train models.

### LSTM autoencoder with Gaussian mixture scoring

The optional LAGMM experiment matrix is defined in:

```text
modeling/configs/lagmm_experiments.json
```

Run it with:

```bash
python -m modeling.experiments modeling/configs/lagmm_experiments.json
```

This implementation follows Wang, Li, and Khattak's LAGMM topology, adapted from
fixed 20-step CAV windows to this project's variable-length trajectories:

1. an LSTM encoder produces a 128-dimensional trajectory state;
2. fully connected layers compress it through 64, 32, and 16 dimensions to one
   latent dimension, with a symmetric decoder;
3. relative Euclidean reconstruction distance and cosine similarity are appended,
   producing the paper's three-dimensional estimation input;
4. a `3 -> 10 -> dropout(0.5) -> 4 softmax` estimation network predicts mixture
   membership;
5. reconstruction loss, sample energy, and covariance regularization are optimized
   jointly.

The model directory additionally contains:

```text
gmm.pkl
gmm_diagnostics.json
```

Feature-wise and transition-wise reconstruction errors are still exported to explain
which inputs were poorly reconstructed. They do not define the LAGMM trajectory score.
The exported GMM component probability describes cluster assignment certainty; it is
not an anomaly probability.
Consequently, transition weighting, mean/top-k trajectory aggregation, and first-step
inclusion controls are disabled for LAGMM in the visualization. The reference-tail
threshold remains dynamic and is applied directly to GMM energy.

### LSTM next-step forecaster

The forecasting experiment matrix is defined in:

```text
modeling/configs/lstm_forecaster_experiments.json
```

Run it with:

```bash
python -m modeling.experiments modeling/configs/lstm_forecaster_experiments.json
```

The forecaster is not an autoencoder. For a trajectory with transitions:

```text
t0, t1, t2, ..., tn
```

it trains on shifted pairs:

```text
input:  t0, t1, ..., t(n-1)
target: t1, t2, ..., tn
```

At each timestep the LSTM hidden state predicts the next transition. The anomaly score
is the squared prediction error for the transition that actually happened next. This
keeps the standard prediction contract:

```text
transition_scores.parquet   one forecast residual per predicted transition
trajectory_scores.parquet   mean forecast residual per trajectory
```

Trajectory-level mean, max, top-k, and sliding-window aggregations are computed from
those transition forecast errors exactly like reconstruction models. Transition order
zero is not scored because it has no previous transition context; the first scored
target is transition order one. This objective is better aligned with local GPS jumps,
unexpected reversals, and sudden motion changes than reconstructing the whole
trajectory from itself.

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
Undefined first-transition acceleration and undefined turn angles are excluded from
their numeric distributions and reported separately as data-quality counts. The
interactive visualization exposes the same summaries and histograms through the
Statistics panel, with JSON, CSV, and PNG export.
Each dataset is reported twice:

```text
all
clean_real
```

`clean_real` uses the same filtering policy as `motion_v2`: trajectories must have at
least two transitions and every transition must be rule-plausible.

## Current Model Entry Points

List registered model families and their capabilities:

```bash
python -m modeling.cli list-models
```

Train the default LSTM autoencoder on valid iNaturalist and Gowalla trajectories:

```bash
python -m modeling.cli train --model-id lstm_motion_v2
```

Evaluate the saved model on both real datasets and synthetic anomalies:

```bash
python -m modeling.cli evaluate --model-id lstm_motion_v2
```

Train and evaluate in one command:

```bash
python -m modeling.cli run --model-id lstm_motion_v2
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
It also computes ROC AUC plus precision, recall, F1, TP, FP, and FN counts at
configured anomaly percentages, overall and per anomaly profile. Thresholds are
derived only from clean real reference scores and are written to:

```text
artifacts/predictions/<model_id>/synthetic/metrics.json
```

Profile precision treats that profile as the positive population and the complete
clean real reference population as negatives. It is therefore intentionally sensitive
to the relative number of real and synthetic trajectories.

Synthetic benchmark metrics keep three populations with different roles:

```text
held-out real calibration scores   set the unsupervised percentile threshold only
manually reviewed plausible data   labeled negatives for FP, FPR, precision, F1, AUC
synthetic injected anomalies       labeled positives for TP, FN, recall, precision, F1, AUC
```

Manually reviewed plausible trajectories are excluded from the threshold-calibration
distribution before their false-positive behavior is measured. This keeps threshold
selection and labeled-negative evaluation disjoint.

The full iNaturalist and Gowalla calibration populations are not treated as plausible
labels because they contain unknown anomalies. Their flagged count is descriptive and
must never be called false positives. Likewise, unmodified source trajectories are
used only for paired score-delta diagnostics; they are not a negative class and no
source-dataset precision, recall, F1, or AUC is reported.

The machine-readable trusted cohort is stored in:

```text
data/labels/plausible_trajectories.csv
```

It includes all eligible trajectories from iNaturalist user `98904` and the reviewed
dates from Gowalla user `16936`. It is a negative-only diagnostic cohort, so it can
estimate false-positive behavior but cannot establish precision on real data.

The interactive visualization can append exact trajectory-level reviewed labels to
the same CSV. Those rows keep `dataset`, `user_id`, and `start_date` for readability
and also store `trajectory_id`, `source`, `created_at`, and optional `notes`. On the
next feature rebuild, reviewed plausible trajectories are forced into calibration
and `use_for_training = false`, even if their user split would otherwise put them in
the fit population. The UI endpoint rejects synthetic trajectories, zero-transition
trajectories, and trajectories with deterministic-rule violations.

One threshold from the selected reference is applied to every synthetic anomaly
profile. No profile receives its own threshold. ROC AUC is calculated from the complete
score ranking and therefore does not change with the percentile slider. Precision,
recall, F1, TP, FP, and FN are threshold-dependent and do change with the slider.
Per-profile precision and F1 define a separate binary task between that one profile
and the complete trusted-negative cohort. They depend on that constructed class ratio;
profile recall, FPR, balanced accuracy, and paired source-score diagnostics are more
appropriate for comparing profile sensitivity across models.

Synthetic data used repeatedly while choosing features, architecture, or thresholds is
a validation set, not a final test set. A final reported result requires a newly
generated source-user-disjoint benchmark whose seed and source users were not inspected
during model development.

Trajectory anomaly scores are derived from transition reconstruction errors and can
be compared without retraining:

```text
mean    average eligible transition error
max     largest eligible transition error
top_k   average of the three largest eligible transition errors
```

The interactive threshold slider is a reference-distribution percentile, labeled
`Reference tail`. A value of 5% selects the score threshold above which approximately
5% of the chosen reference population falls. It is not a calibrated probability that
an individual trajectory is anomalous. Producing anomaly probabilities would require
a representative labeled calibration set containing both normal and anomalous cases.

The dataset statistics panel can also derive distributions for trajectories flagged or
accepted by the selected model. These diagnostics use the active transition scoring,
trajectory aggregation, first-transition policy, and percentile. They compare the
selected population with all model-eligible trajectories and expose model score,
trajectory length, maximum speed, total distance, total elapsed time, transition
features, and feature-wise reconstruction-error lift.

Only models from `configurable_pipeline_v2` are indexed in the live visualization
database. Older model weights and predictions remain on disk, but their trajectory IDs
refer to the pre-sessionization dataset and must not be mixed with current canonical
trajectories.

Transition order zero is included by default. Its acceleration and bearing change are
explicitly marked invalid, so models can use its valid speed, time, and distance
without learning artificial acceleration or turn values. Each aggregation has its own percentile
threshold curve and synthetic metrics because their score distributions differ.
Maximum and top-k values are derived from transition-score Parquet when metrics and
the visualization cache are built; they are not duplicated in prediction storage.
Prediction errors are stored as float32 and transition order as int32. This is enough
precision for ranking and visualization while avoiding float64 duplication. Existing
prediction artifacts can be compacted with:

```bash
python -m modeling.compact_predictions
```

Real trajectories that fail the deterministic transition plausibility policy are
excluded before feature generation, so they cannot enter model training or the clean
negative evaluation population. They remain in canonical Parquet and are displayed as
deterministic baseline anomalies. Injected synthetic trajectories remain in model
evaluation even when the baseline also catches them; otherwise the benchmark would
silently remove the most obvious anomaly profiles instead of measuring model recall.

Metrics can be regenerated from existing predictions:

```bash
python -m modeling.synthetic_evaluation --model-id lstm_motion_v2
```

The initial registered model is `lstm_autoencoder`. It trains only on trajectories
marked `use_for_training = true`, which excludes synthetic anomalies. Evaluation
includes `inat`, `gowalla`, and `synthetic` by default.

`lstm_seq2seq_autoencoder` is also registered. Its encoder reads the complete
trajectory. During training, the decoder receives a learned start token followed by
the previous real transition (teacher forcing); during evaluation, it receives its
previous reconstruction. It emits the same trajectory, transition, and feature-wise
error artifacts as the original autoencoder, so both models are interchangeable in
the visualization.

`t_lstm_autoencoder` implements a time-aware LSTM encoder and decoder based on the
short-term memory decay proposed by Baytas et al. in
[Patient Subtyping via Time-Aware LSTM Networks](https://doi.org/10.1145/3097983.3097997).
Before each recurrent update, the previous cell state is decomposed into short- and
long-term components. Only the short-term component is decayed:

```text
short_memory = tanh(W_d * previous_cell + b_d)
decay         = 1 / log(e + elapsed_hours)
adjusted_cell = previous_cell - short_memory + short_memory * decay
```

The normal input, forget, candidate, and output gates then use `adjusted_cell`.
Elapsed time has two deliberately different roles:

```text
scaled elapsed_time feature    reconstructed like distance and bearing
unscaled time context          controls recurrent memory decay
```

The auxiliary context is read from the stored nonnegative `log1p(seconds)` column,
copied before model-specific scaling, converted back to seconds, then expressed in
hours. This avoids passing centered RobustScaler values to a decay function that
requires a nonnegative physical duration. No duplicate Parquet column is stored.

Its decoder receives the encoded latent vector at every transition. It is time-aware,
but it is not autoregressive sequence-to-sequence decoding.

`t_lstm_seq2seq_autoencoder` uses the same time-aware encoder and cell-state decay,
but changes the decoder:

```text
training     learned start token, then previous real transition (teacher forcing)
evaluation   learned start token, then previous reconstructed transition
time decay   target transition's elapsed interval at every decoder step
```

This is the direct time-aware counterpart of `lstm_seq2seq_autoencoder`. The separate
model type allows the effect of time-aware recurrence and the effect of autoregressive
decoding to be compared independently.

Train and evaluate the eight repeated-latent T-LSTM comparisons:

```bash
python -m modeling.experiments modeling/configs/t_lstm_experiments.json
```

Train only the eight T-LSTM seq2seq comparisons without retraining the first set:

```bash
python -m modeling.experiments modeling/configs/t_lstm_seq2seq_experiments.json
python build_database.py --comparison-group configurable_pipeline_v2
```

Each file covers the same four training populations and two feature combinations as
the current LSTM comparison matrix. Both T-LSTM variants use the same prediction
Parquet schema, synthetic metrics, model database records, threshold controls, and
feature-wise error visualization as other reconstruction models.

Model experiments use explicit configuration dimensions rather than model-name
conventions:

```text
model_type                         decoder/algorithm plugin
training.datasets                 iNaturalist, Gowalla, or both
training.sampling_strategy        all or balanced
experiment.feature_representation continuous_speed or mpp_speed_ranges
evaluation.ignore_first_transition transition-zero score policy
features                           exact model input columns
```

Balanced sampling deterministically draws the same number of trajectories from each
training dataset. With the current feature tables this is 13,900 iNaturalist and
13,900 Gowalla trajectories.

The MPP speed representation uses eight mutually exclusive one-hot ranges:

```text
0, (0,5), [5,10), [10,25), [25,80), [80,140), [140,200), [200,+inf) km/h
```

Training uses `masked_concept_mse`. Every physical concept has equal total loss
weight: all speed-range columns together count as one speed concept, and the bearing
columns together count as one bearing concept. This prevents a feature family from
dominating merely because it uses more columns.

## Reproducible Pipeline Command

The complete dependency order is encoded in `pipeline.py`:

```text
preprocess -> synthetic -> features -> stats -> models -> database
```

Preview commands:

```bash
python pipeline.py --dry-run
```

Rebuild everything:

```bash
python pipeline.py
```

Run or resume a subset:

```bash
python pipeline.py --to-stage features
python pipeline.py --from-stage models
python pipeline.py --skip models
```

No stage deletes previous model directories automatically. Model IDs and comparison
groups distinguish experiment generations. The current matrix prefixes model IDs with
`v2_`, so it does not overwrite the previous model generation.

## Contextual Feature Policy

Context features should be separated by whether they are available at prediction time:

```text
causal transition context
  previous-speed change
  previous elapsed-time ratio
  previous distance ratio
  turn magnitude and turn rate
  recent rolling stop/reversal frequency

whole-trajectory diagnostics
  total duration and distance
  direct-distance/path-distance efficiency
  stop fraction
  maximum and high-percentile speed
  revisit or alternating-location frequency
```

Causal features can be inputs to next-transition models because they use only the
current and earlier observations. Whole-trajectory values are useful for classical
trajectory-level models and diagnostics, but feeding them into a transition predictor
would leak future information. They should be stored as one row per trajectory, not
repeated on every transition.

## Evaluation Population Contract

Canonical trajectories rejected by deterministic plausibility or minimum-length
cleaning never enter feature Parquet, model predictions, thresholds, or ML metrics.
They remain visible as deterministic baseline anomalies in the application.

Eligible real trajectories have user-disjoint roles:

```text
fit          used to fit the model-specific scaler and model weights
calibration  used for threshold distributions and real reference metrics
```

Predictions may be exported for both roles so fit trajectories can still be inspected
in the visualization. Benchmark code explicitly joins real scores to
`use_for_calibration = true`; fit trajectories therefore do not enter metric
denominators. Synthetic trajectories are benchmark-only and are never used for a
scaler, model fit, or real threshold calibration.

Manually reviewed plausible trajectories are a protected subset of calibration.
They are excluded from training/scaler fitting, excluded from threshold selection
when benchmark metrics are computed, and used as labeled negatives for false-positive
diagnostics.

## Sequence Length

Length-group calibration means estimating score thresholds separately for predefined
trajectory-length buckets, for example 2-3, 4-7, 8-15, and 16+ transitions. It does
not modify canonical trajectories. It compensates for score distributions that vary
with sequence length, but each bucket needs enough calibration trajectories and the
bucket boundaries must be fixed before final evaluation.

Fixed or overlapping model windows are a separate option. A long trajectory can be
viewed as windows such as 8 transitions with stride 4, then window scores can be
aggregated back to the original trajectory. This reduces padding and localizes short
anomalies, but loses context at boundaries. Canonical trajectory storage should remain
unchanged; windowing belongs in the model input loader.

## Forecasting Alternatives

Next-transition prediction consumes transitions up to time `t-1` and predicts
transition `t`. Its error asks whether the next movement is surprising given the
observed past, and the model cannot reconstruct the same anomalous transition it is
being scored against. Transition zero has no previous context and is therefore
unscored or handled by a separate initial-state model.

Probabilistic forecasting predicts a conditional distribution instead of one point,
for example a mean and scale or mixture components. Negative log-likelihood then
becomes the anomaly score. It can represent several plausible movement modes and
different uncertainty levels, but requires carefully constrained distributions and
more calibration work than point prediction.

The explicit experiment runner supports independent runs and checkpoint reuse:

```bash
python -m modeling.experiments \
  modeling/configs/population_and_speed_experiments.json
```

The visualization's Model setup panel filters artifacts by each of these pipeline
choices. It selects already trained, reproducible artifacts; training remains a CLI
operation.

Train and evaluate the full-feature sequence-to-sequence model:

```bash
python -m modeling.cli run \
  --config modeling/configs/lstm_seq2seq_autoencoder.json \
  --model-id lstm_seq2seq_autoencoder_full_e15_b256
```

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
