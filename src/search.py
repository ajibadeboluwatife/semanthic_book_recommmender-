import os
import pickle
import numpy as np
import pandas as pd
import faiss
from typing import List, Dict, Any, Tuple

from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

from utils import l2_normalize, join_fields

class SemanticSearcher:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.df = pd.read_parquet(os.path.join(data_dir, "books.parquet"))
        with open(os.path.join(data_dir, "meta.pkl"), "rb") as f:
            self.meta = pickle.load(f)
        self.model = SentenceTransformer(self.meta.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"))
        self.index = faiss.read_index(os.path.join(data_dir, "books.index"))

        # Prepare corpus for BM25 (optional)
        self.corpus = (self.df["title"].fillna("") + " " + 
                       self.df["author"].fillna("") + " " +
                       self.df["genres"].fillna("") + " " + 
                       self.df["description"].fillna("")).tolist()
        self.tokenized_corpus = [doc.lower().split() for doc in self.corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def query(self, text: str, top_k: int = 10, filters: Dict[str, Any] = None,
              hybrid_alpha: float = 0.7) -> pd.DataFrame:
        filters = filters or {}
        # Apply filters to candidate mask
        mask = pd.Series([True] * len(self.df))
        if "genres" in filters and filters["genres"]:
            genres = set([g.strip().lower() for g in filters["genres"]])
            mask &= self.df["genres"].fillna("").str.lower().apply(
                lambda s: any(g in s for g in genres)
            )
        if "year_min" in filters:
            mask &= self.df["year"] >= int(filters["year_min"])
        if "year_max" in filters:
            mask &= self.df["year"] <= int(filters["year_max"])
        if "min_rating" in filters:
            mask &= self.df["rating"].fillna(0) >= float(filters["min_rating"])

        cand_idx = np.where(mask.values)[0]
        if len(cand_idx) == 0:
            return self.df.head(0)

        # Semantic scores
        q = self.model.encode([text], convert_to_numpy=True)
        q = l2_normalize(q)
        D, I = self.index.search(q, min(top_k*5, len(self.df)))  # oversample for re-filtering
        I = I[0]
        D = D[0]

        # Build score dict for semantic
        sem_scores = {int(i): float(s) for i, s in zip(I, D)}

        # BM25 scores
        tokenized_q = text.lower().split()
        bm25_scores_all = self.bm25.get_scores(tokenized_q)
        bm25_scores = {i: float(bm25_scores_all[i]) for i in cand_idx}

        # Normalize scores to [0,1]
        def norm_scores(d: Dict[int, float]) -> Dict[int, float]:
            if not d:
                return {}
            vals = np.array(list(d.values()))
            min_v, max_v = float(vals.min()), float(vals.max())
            if max_v - min_v < 1e-9:
                return {k: 0.0 for k in d}
            return {k: (v - min_v) / (max_v - min_v) for k, v in d.items()}

        sem_n = norm_scores({k: v for k, v in sem_scores.items() if k in cand_idx})
        bm25_n = norm_scores(bm25_scores)

        # Hybrid blend
        keys = set(sem_n.keys()) | set(bm25_n.keys())
        blended = {k: hybrid_alpha * sem_n.get(k, 0.0) + (1 - hybrid_alpha) * bm25_n.get(k, 0.0)
                   for k in keys if k in cand_idx}

        # Rank and assemble DataFrame
        top = sorted(blended.items(), key=lambda x: x[1], reverse=True)[:top_k]
        rows = []
        for idx, score in top:
            row = self.df.iloc[idx].to_dict()
            row["score"] = float(score)
            rows.append(row)
        return pd.DataFrame(rows)

if __name__ == "__main__":
    # Quick manual test (assumes you've built the index)
    s = SemanticSearcher("data")
    res = s.query("space survival on mars", top_k=5, filters={"genres": ["Science Fiction"]})
    print(res[["title","author","genres","score"]])
