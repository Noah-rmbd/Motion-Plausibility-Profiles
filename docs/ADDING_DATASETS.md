# Adding New Datasets

This project is built around canonical Parquet schemas. If you want to use a new dataset, you do not need to format it as a `.log` text file. Instead, you can use the abstract dataset loader interface to process your dataset directly into the canonical Parquet format.

## The Canonical Schema

The ML pipeline and visualization tools rely on three main tables:

### 1. `observations.parquet`
Stores raw GPS fixes.
- `dataset`: string
- `user_id`: string
- `observation_id`: string
- `timestamp`: string (e.g. "YYYY-MM-DD HH:MM:SS")
- `lat`: float64
- `lon`: float64
- `is_obscured`: bool

### 2. `transitions.parquet`
Stores movement features between consecutive observations.
- `dataset`: string
- `transition_id`: int64
- `observation_id1`: string
- `observation_id2`: string
- `speed_kmh`: float64
- `elapsed_time_s`: float64
- `distance_m`: float64
- `acceleration_m_s2`: float64
- `acceleration_valid`: bool
- `bearing_change_rad`: float64
- `bearing_change_valid`: bool
- `transition_plausibility`: int8
- `plausibility_reason`: string

### 3. `trajectories.parquet`
Groups transitions into continuous trajectories.
- `dataset`: string
- `trajectory_id`: int64
- `user_id`: string
- `n_transitions`: int64
- `start_observation_id`: string
- `end_observation_id`: string

## AbstractDatasetLoader

To easily ingest a new dataset format, implement the `AbstractDatasetLoader` provided in `dataset_preprocessing/ingestion/loaders/base.py`. This base class handles the complexities of sessionization, physical calculations (distance, speed, acceleration), and writing the output to Parquet. 

Your implementation only needs to parse your specific format (e.g. JSON, CSV, GeoJSON) and yield observation dictionaries.

### Example:

```python
from dataset_preprocessing.ingestion.loaders.base import AbstractDatasetLoader

class MyCustomLoader(AbstractDatasetLoader):
    
    def iter_observations(self, input_path):
        # 1. Parse your dataset file (input_path)
        # 2. Yield observations in the following format:
        yield {
            "user_id": "user123",
            "observation_id": "obs_1",
            "timestamp": "2023-01-01 12:00:00",
            "lat": 48.8566,
            "lon": 2.3522,
            "is_obscured": False
        }

# Using your loader:
loader = MyCustomLoader("my_custom_dataset")
loader.process("path/to/raw/data.csv", "data/processed_parquet")
```

Once processed, you can reference your dataset in the ML pipeline!
