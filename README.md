# Motion Plausibility Profiles: ML Pipeline & Interactive Visualization

Welcome to the Motion Plausibility Profiles (MPP) advanced pipeline. This repository extends the original 2022 project (see `original_mpp/` folder) by introducing a powerful Machine Learning pipeline for automated trajectory plausibility analysis and a rich interactive web visualization.

## What is this?
This project aims to detect implausible GPS trajectories (e.g., GPS jumps, unrealistic speeds) through unsupervised anomaly detection models (Autoencoders, Seq2Seq, etc.) and visualizes the results on an interactive map.

## Quickstart

### 1. Requirements

Install the required Python dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run the ML Pipeline

The pipeline preprocesses raw data, generates features, trains several ML models, and evaluates them.

```bash
# Preview the pipeline steps without running them
python pipeline.py --dry-run

# Run the complete canonical pipeline
python pipeline.py
```
*Note: Depending on the size of your datasets, the full pipeline can take some time.*

### 3. Start the Interactive Visualization

Once the pipeline has successfully built the model predictions, you can launch the visualization server to explore the results interactively:

```bash
# Build the SQLite cache database for fast serving
python build_database.py

# Start the web visualization server
python open_visualization.py
```

Open your browser at `http://127.0.0.1:8001` to view the map and analysis tools.

## Documentation

For a deeper dive into the architecture and concepts, please refer to our detailed documentation:

- [Machine Learning Pipeline](docs/ML_PIPELINE.md): Detailed explanation of preprocessing, model architectures, and evaluation.
- [Interactive Visualization](docs/VISUALIZATION.md): Guide on how to use the web application and its features.
- [Adding New Datasets](docs/ADDING_DATASETS.md): Learn how to use the `AbstractDatasetLoader` to plug your own data formats into the canonical schema.
- [Architecture](ARCHITECTURE.md): The core structural design of this repository.

## Original 2022 Project
This project is an extension of the work by Tobias Isenberg et al. (2022). 
The original files and README for that paper can be found in the `original_mpp/` folder.