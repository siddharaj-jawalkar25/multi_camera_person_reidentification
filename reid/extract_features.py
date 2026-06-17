"""
Step 2: ReID Feature Extraction (Person B's task)
---------------------------------------------------
Loads a pretrained OSNet model and turns a cropped person image into
an embedding vector that can be compared via cosine similarity.

STATUS: stub — fill in once torchreid is installed and tested.
Claude will build this out fully once detect_and_track.py is confirmed working.
"""

import numpy as np


def load_model():
    """Load pretrained OSNet model. TODO: implement with torchreid."""
    raise NotImplementedError


def get_embedding(crop_image: np.ndarray) -> np.ndarray:
    """
    Args:
        crop_image: BGR numpy array (a single cropped person image from OpenCV)
    Returns:
        1D numpy array embedding (e.g. 512-dim)
    """
    raise NotImplementedError


def average_embeddings(embeddings: list) -> np.ndarray:
    """Temporal aggregation: average a list of embeddings into one track-level embedding."""
    return np.mean(np.stack(embeddings), axis=0)
