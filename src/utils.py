import re
import numpy as np
import pandas as pd
from typing import List

def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s

def join_fields(row: pd.Series) -> str:
    parts = [
        clean_text(row.get("title", "")),
        clean_text(row.get("author", "")),
        clean_text(row.get("genres", "")),
        clean_text(row.get("description", "")),
    ]
    return " | ".join([p for p in parts if p])

def l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / norms
