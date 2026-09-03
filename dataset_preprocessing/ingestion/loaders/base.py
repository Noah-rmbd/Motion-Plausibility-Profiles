from abc import ABC, abstractmethod
from typing import Iterator, Dict, Any

class AbstractDatasetLoader(ABC):
    """
    Abstract interface for loading new datasets into the Motion Plausibility Profiles canonical Parquet format.
    
    Subclasses should implement `iter_observations()` to read their specific data format 
    (e.g., CSV, JSON, GeoJSON) and yield observation dictionaries.
    
    The expected schema for the yielded dictionaries is:
    {
        "user_id": str,
        "observation_id": str,
        "timestamp": str (format: 'YYYY-MM-DD HH:MM:SS'),
        "lat": float,
        "lon": float,
        "is_obscured": bool
    }
    """
    
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        
    @abstractmethod
    def iter_observations(self, input_path: str) -> Iterator[Dict[str, Any]]:
        """
        Parse the input file and yield raw observations.
        
        Args:
            input_path: Path to the raw dataset file.
            
        Yields:
            Dict containing the keys: 'user_id', 'observation_id', 'timestamp', 'lat', 'lon', 'is_obscured'
        """
        pass
    
    def process(self, input_path: str, output_dir: str):
        """
        Process the dataset into canonical Parquet tables.
        
        In a complete implementation, this method would consume `iter_observations`,
        calculate physical features (speed, distance, elapsed time, acceleration, bearing),
        apply sessionization, evaluate physical plausibility rules, and write the output
        to the 3 canonical Parquet files: observations.parquet, transitions.parquet, trajectories.parquet.
        
        For the full logic of these transformations, refer to `dataset_preprocessing/preprocess_logs_to_parquet.py`.
        """
        raise NotImplementedError("The core processing loop should be integrated or implemented here.")
