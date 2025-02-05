from dataclasses import dataclass
from enum import Enum
from typing import Optional

class DecayType(Enum):
    LINEAR = "linear"
    EXPONENTIAL = "exponential"

@dataclass
class DynamicDecodingConfig:
    # Temperature settings
    min_temperature: float = 0.1
    max_temperature: float = 2.0
    initial_temperature: float = 1.0
    
    # Decay settings
    decay_type: DecayType = DecayType.LINEAR
    decay_rate: float = 0.1  # For exponential decay
    linear_step: float = 0.1  # For linear decay
    
    # Similarity settings
    similarity_threshold: float = 0.85
    top_k: int = 5
    
    # Processing settings
    batch_size: int = 32
    
    # Cache settings
    cache_embeddings: bool = True
    cache_dir: Optional[str] = ".cache" 