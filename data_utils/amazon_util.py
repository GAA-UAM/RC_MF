import os
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix


def read_amazon(
    data,
    min_user_ratings=5,
    min_item_ratings=5,
):
    proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(proj_root, "data")
    path = os.path.join(data_dir, "Amazon", f"{data}.jsonl")

    chunks = []
    for chunk in pd.read_json(path, lines=True, chunksize=100_000):
        chunks.append(chunk[["user_id", "asin", "rating"]])
    df = pd.concat(chunks, ignore_index=True)

    print(f"Before filtering: {len(df)} interactions")

    df = df.dropna(subset=["user_id", "asin", "rating"]).copy()

    # Collapse duplicate raw user-item pairs BEFORE factorization
    df = df.groupby(["user_id", "asin"], as_index=False)["rating"].mean()

    print(f"After deduplicating user-item pairs: {len(df)} interactions")

    user_counts = df["user_id"].value_counts()
    valid_users = user_counts[user_counts >= min_user_ratings].index
    df = df[df["user_id"].isin(valid_users)].copy()

    item_counts = df["asin"].value_counts()
    valid_items = item_counts[item_counts >= min_item_ratings].index
    df = df[df["asin"].isin(valid_items)].copy()

    print(f"After filtering: {len(df)} interactions")

    df["user"], user_ids = pd.factorize(df["user_id"], sort=True)
    df["item"], item_ids = pd.factorize(df["asin"], sort=True)

    rows = df["user"].to_numpy(dtype=np.int64)
    cols = df["item"].to_numpy(dtype=np.int64)
    values = df["rating"].to_numpy(dtype=np.float32)

    pairs = np.column_stack((rows, cols)).astype(np.int64, copy=False)

    num_users = len(user_ids)
    num_items = len(item_ids)

    R_obs = coo_matrix(
        (values, (rows, cols)),
        shape=(num_users, num_items),
        dtype=np.float32,
    ).tocsr()

    print(f"Final df rows: {len(df)} | Sparse nnz: {R_obs.nnz}")

    return df, R_obs, pairs, num_users, num_items
