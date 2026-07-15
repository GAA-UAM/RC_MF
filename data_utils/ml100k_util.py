import os
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix


def read_data_ml100k():
    user_col, item_col, title_col = "userId", "movieId", "title"
    names = [user_col, item_col, "rating", "timestamp"]

    proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(proj_root, "data")

    if not os.path.exists(data_dir):
        from d2l import torch as d2l

        d2l.DATA_HUB["ml-100k"] = (
            "https://files.grouplens.org/datasets/movielens/ml-100k.zip",
            "cd4dcac4241c8a4ad7badc7ca635da8a69dddb83",
        )
        data_dir = d2l.download_extract("ml-100k")
        ratings = pd.read_csv(os.path.join(data_dir, "u.data"), sep="\t", names=names)
    else:
        data_dir = os.path.join(data_dir, "ml-100k")
        ratings = pd.read_csv(os.path.join(data_dir, "u.data"), sep="\t", names=names)

    movies = pd.read_csv(
        os.path.join(data_dir, "u.item"),
        delimiter="|",
        encoding="latin-1",
        header=None,
        names=[item_col, title_col, "date", "N", "url", *[f"g{i}" for i in range(19)]],
    )

    rating_movie = ratings.merge(movies[[item_col, title_col]], on=item_col, how="left")
    
    rating_movie = rating_movie.dropna(subset=["userId", "movieId", "rating"])

    user_counts = rating_movie.groupby(user_col).size()
    valid_users = user_counts[user_counts >= 25].index
    rating_movie = rating_movie[rating_movie[user_col].isin(valid_users)].copy()

    item_counts = rating_movie.groupby(item_col).size()
    valid_items = item_counts[item_counts >= 25].index
    rating_movie = rating_movie[rating_movie[item_col].isin(valid_items)].copy()

    rating_movie["user"], user_ids = pd.factorize(rating_movie[user_col], sort=True)
    rating_movie["item"], item_ids = pd.factorize(rating_movie[item_col], sort=True)

    rows = rating_movie["user"].to_numpy(dtype=np.int64)
    cols = rating_movie["item"].to_numpy(dtype=np.int64)
    values = rating_movie["rating"].to_numpy(dtype=np.float32)

    pairs = np.column_stack((rows, cols)).astype(np.int64, copy=False)

    num_users = len(user_ids)
    num_items = len(item_ids)

    R_obs = coo_matrix(
        (values, (rows, cols)),
        shape=(num_users, num_items),
        dtype=np.float32,
    ).tocsr()

    return rating_movie, R_obs, pairs, num_users, num_items
