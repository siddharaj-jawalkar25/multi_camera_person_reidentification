"""
Step 3: Global ID Matching + TTL (Person B's task)
------------------------------------------------------
In-memory store of active global IDs. Given a new (camera_id, track_id, embedding,
timestamp), decides whether it matches an existing global ID or needs a new one.
Also handles releasing IDs that have been idle past the TTL.

STATUS: stub — Claude will build this out fully next.
"""

import time
import numpy as np
from config import SIMILARITY_THRESHOLD, TTL_SECONDS


class IdentityManager:
    def __init__(self):
        self.active = {}   # global_id -> {"embedding": np.array, "last_seen": float, "camera_id": str}
        self.next_id = 1

    def cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    def update(self, camera_id: str, track_id: int, embedding: np.ndarray, timestamp: float = None):
        """TODO: implement matching against self.active, threshold decision, TTL cleanup."""
        raise NotImplementedError

    def cleanup_expired(self, now: float = None):
        """TODO: remove entries from self.active whose last_seen is older than TTL_SECONDS."""
        raise NotImplementedError
