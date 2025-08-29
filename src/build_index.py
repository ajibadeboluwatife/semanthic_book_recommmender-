import argparse
import os
import pickle
import numpy as np
import pandas as pd
import faiss

from sentence_transformers import SentenceTransformer
from utils import join_fields, l2_normalize

def build(args):
    df = pd.read_csv(args.data)
    assert "book_id" in df.columns, "CSV must have a 'book_id' column"
    df["doc"] = df.apply(join_fields, axis=1)

    model_name = args.model
    model = SentenceTransformer(model_name)
    X = model.encode(df["doc"].tolist(), batch_size=args.batch, convert_to_numpy=True, show_progress_bar=True)
    X = l2_normalize(X)  # cosine => inner product on normalized vectors

    d = X.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(X)

    # Save artifacts
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    faiss.write_index(index, os.path.join(out_dir, "books.index"))
    df.drop(columns=["doc"]).to_parquet(os.path.join(out_dir, "books.parquet"), index=False)

    meta = {
        "model_name": model_name,
        "dim": int(d),
        "n_items": int(X.shape[0]),
        "fields": ["title","author","year","genres","description","isbn13","rating"],
    }
    with open(os.path.join(out_dir, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)

    print(f"Indexed {X.shape[0]} books with dim={d} using {model_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to books CSV")
    parser.add_argument("--out_dir", type=str, default="data", help="Where to write index + parquet")
    parser.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()
    build(args)
