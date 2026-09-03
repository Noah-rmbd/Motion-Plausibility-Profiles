# Interactive Visualization

The visualization platform in this project allows you to visually explore the trajectory datasets, the model results, and the corresponding anomaly scores on an interactive map.

## Overview

The interactive visualization is built on a web application stack:
- **Backend**: Python HTTP server serving the SQLite cache.
- **Frontend**: HTML, JS, CSS, Leaflet maps, and Plotly charts.

## Running the Visualization

After you have generated the database using the pipeline:

```bash
# 1. Build the visualization database (if you haven't yet)
python build_database.py

# 2. Start the visualization server
python open_visualization.py
```

Then, open your web browser and navigate to:
```
http://127.0.0.1:8001
```

## Features

### Map Display
Explore geographic clusters, view paths with speed-based color gradients, and hover over specific observations to see diagnostics (speed, distance, time). There is also a progressive path animation mode.

### Model Configuration & Stats
Filter by specific architectures, choose different threshold levels using a dynamic slider, and view real-time metrics (AUC, Precision, Recall, F1). The stats panel displays global statistics of the selected population.

### Motion Plausibility Profile (MPP)
The MPP chart is a synchronized Plotly chart displaying per-transition reconstruction errors alongside the map. Points and paths are colored by their speed to provide an intuitive understanding of the movement.
