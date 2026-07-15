import numpy as np
import pandas as pd
from scipy import sparse
from collections import Counter
from data_utils.data_loader import DataLoader


class DataManager:
    def __init__(self, dataset, seed):
        self.dataset = dataset
        self.seed = seed
        self.data_loader = DataLoader(dataset, seed)

    def _stage_load(self):
        df, R_obs = self.data_loader()

        if not sparse.issparse(R_obs):
            raise ValueError(
                "The model expects R_obs to be a sparse observed rating matrix."
            )

        if df is None:
            raise ValueError(
                "The data loader must return a DataFrame-like object as df."
            )

        if not isinstance(df, pd.DataFrame):
            try:
                df = pd.DataFrame(df)
            except Exception as e:
                raise ValueError(
                    "df returned by data_loader could not be converted to a pandas DataFrame."
                ) from e

        return df, R_obs.tocsr()

    def load_data(self):
        self.df_obs, self.sparse_obs = self._stage_load()
        self._validate_data()
        return self.df_obs, self.sparse_obs

    def get_sparse_info(self):
        split = self._ensure_split_cache(test_fraction=0.2)

        sparse_train_pairs = split["train_pairs"]
        sparse_test_pairs = split["test_pairs"]
        sparse_train = split["sparse_train"]
        sparse_test = split["sparse_test"]

        obs = np.vstack([sparse_train_pairs, sparse_test_pairs])

        y_true = pd.DataFrame(
            {
                "user_id": sparse_test_pairs[:, 0],
                "item_id": sparse_test_pairs[:, 1],
                "y_true": np.asarray(sparse_test.data, dtype=np.float64).reshape(-1),
            }
        )

        n_users, n_items = self.sparse_obs.shape
        n_obs = len(obs)
        density = n_obs / (n_users * n_items)

        self.sparse_info = {
            "n_users": n_users,
            "n_items": n_items,
            "n_observed_sparse": n_obs,
            "density": density,
            "n_train_pairs_sparse": len(sparse_train_pairs),
            "n_test_pairs_sparse": len(sparse_test_pairs),
        }

        return (obs, y_true)

    def get_df_info(self):
        split = self._ensure_split_cache(test_fraction=0.2)

        train_df = split["train_df"].copy()
        test_df = split["test_df"].copy()

        self.df_info = {
            "n_observed_df": len(self.df_obs),
            "train_df_shape": train_df.shape,
            "test_df_shape": test_df.shape,
        }

        return train_df, test_df

    def observed_pairs_from_sparse(self, X):
        if not sparse.issparse(X):
            raise ValueError("X must be sparse.")

        coo = X.tocoo()
        return np.column_stack((coo.row, coo.col)).astype(np.int64, copy=False)

    def pairs_to_sparse(self, pairs, R_obs):
        pairs = np.asarray(pairs, dtype=np.int64)

        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError("pairs must have shape (n_pairs, 2).")

        rows = pairs[:, 0]
        cols = pairs[:, 1]
        values = np.asarray(R_obs[rows, cols]).reshape(-1)

        return sparse.csr_matrix(
            (values, (rows, cols)),
            shape=R_obs.shape,
            dtype=R_obs.dtype,
        )

    def train_test_split(
        self,
        obs,
        test_fraction=0.2,
        seed=None,
    ):
        if seed is None:
            seed = self.seed

        obs = np.asarray(obs, dtype=np.int64)
        n_obs = len(obs)

        rng = np.random.RandomState(seed)
        order = rng.permutation(n_obs)

        users = obs[:, 0]
        items = obs[:, 1]

        user_train_count = Counter(users)
        item_train_count = Counter(items)

        target_test = int(round(test_fraction * n_obs))

        train_mask = np.ones(n_obs, dtype=bool)
        n_test = 0

        for idx in order:
            if n_test >= target_test:
                break

            u, i = obs[idx]

            if user_train_count[u] > 1 and item_train_count[i] > 1:
                train_mask[idx] = False
                user_train_count[u] -= 1
                item_train_count[i] -= 1
                n_test += 1

        train_pairs = obs[train_mask]
        test_pairs = obs[~train_mask]

        return train_pairs, test_pairs

    def cv_splits(self, train_pairs, n_splits=5, val_fraction=0.2, seed=None):
        if seed is None:
            seed = self.seed

        rng = np.random.RandomState(seed)
        train_pairs = np.asarray(train_pairs, dtype=np.int64)

        for _ in range(n_splits):
            split_seed = rng.randint(0, 10**9)
            fold_train_pairs, fold_val_pairs = self.train_test_split(
                train_pairs,
                test_fraction=val_fraction,
                seed=split_seed,
            )
            yield fold_train_pairs, fold_val_pairs

    def _infer_df_columns(self, df):
        cols = list(df.columns)
        lower_map = {str(c).lower(): c for c in cols}

        candidates_user = [
            "user",
            "user_idx",
            "userid",
            "user_id",
            "uid",
            "reviewerid",
            "userid",
            "useridx",
        ]
        candidates_item = [
            "item",
            "item_idx",
            "itemid",
            "item_id",
            "asin",
            "iid",
            "movieid",
            "movie_id",
            "productid",
            "product_id",
        ]
        candidates_rating = [
            "rating",
            "overall",
            "ratings",
            "score",
            "value",
            "label",
        ]

        user_col = next((lower_map[c] for c in candidates_user if c in lower_map), None)
        item_col = next((lower_map[c] for c in candidates_item if c in lower_map), None)
        rating_col = next(
            (lower_map[c] for c in candidates_rating if c in lower_map), None
        )

        if user_col is None or item_col is None or rating_col is None:
            raise ValueError(
                "Could not infer user/item/rating columns from df. "
                f"Columns found: {list(df.columns)}"
            )

        return user_col, item_col, rating_col

    def get_uir_df(
        self,
        df=None,
        user_col=None,
        item_col=None,
        rating_col=None,
        dropna=True,
    ):
        if df is None:
            if not hasattr(self, "df_obs"):
                self.load_data()
            df = self.df_obs

        if user_col is None or item_col is None or rating_col is None:
            inferred_user, inferred_item, inferred_rating = self._infer_df_columns(df)
            user_col = user_col or inferred_user
            item_col = item_col or inferred_item
            rating_col = rating_col or inferred_rating

        out = df[[user_col, item_col, rating_col]].copy()
        out.columns = ["user", "item", "rating"]

        if dropna:
            out = out.dropna(subset=["user", "item", "rating"])

        return out.reset_index(drop=True)

    def _validate_data(self):

        df = self.df_obs
        R_obs = self.sparse_obs

        uir_df = self.get_uir_df(df)

        if not {"user", "item", "rating"}.issubset(uir_df.columns):
            raise ValueError("Canonical columns user/item/rating are missing.")

        rows = uir_df["user"].to_numpy(dtype=np.int64)
        cols = uir_df["item"].to_numpy(dtype=np.int64)
        vals = uir_df["rating"].to_numpy(dtype=np.float64)
        if len(uir_df) == 0 or R_obs.nnz == 0:
            raise ValueError(
                f"Dataset '{self.dataset}' is empty after preprocessing/filtering. "
                f"df rows={len(uir_df)}, sparse nnz={R_obs.nnz}. "
                "Lower min_user_ratings/min_item_ratings or skip this dataset."
            )

        if rows.max() >= R_obs.shape[0]:
            raise ValueError(
                "Some dataframe user ids exceed sparse matrix row dimension."
            )
        if cols.max() >= R_obs.shape[1]:
            raise ValueError(
                "Some dataframe item ids exceed sparse matrix column dimension."
            )

        dup_count = uir_df.duplicated(subset=["user", "item"]).sum()
        if dup_count > 0:
            dup_examples = (
                uir_df[uir_df.duplicated(subset=["user", "item"], keep=False)]
                .sort_values(["user", "item"])
                .head(10)
            )
            raise ValueError(
                f"Mismatch in number of observations: df has {len(uir_df)} rows, "
                f"sparse has {R_obs.nnz} nonzeros. "
                f"Found {dup_count} duplicate (user, item) rows. "
                f"Examples:\n{dup_examples}"
            )

        nnz = R_obs.nnz
        if len(uir_df) != nnz:
            raise ValueError(
                f"Mismatch in number of observations: df has {len(uir_df)} rows, sparse has {nnz} nonzeros."
            )

        sparse_vals = np.asarray(R_obs[rows, cols]).reshape(-1)

        if not np.allclose(vals, sparse_vals):
            bad_idx = np.where(~np.isclose(vals, sparse_vals))[0][:10]
            examples = [
                {
                    "user": int(rows[i]),
                    "item": int(cols[i]),
                    "df_rating": float(vals[i]),
                    "sparse_rating": float(sparse_vals[i]),
                }
                for i in bad_idx
            ]
            raise ValueError(
                "DataFrame and sparse matrix are not aligned on some entries. "
                f"Examples: {examples}"
            )

        return True

    def df_to_cornac_uir(
        self,
        df=None,
        user_col=None,
        item_col=None,
        rating_col=None,
    ):
        uir_df = self.get_uir_df(
            df=df,
            user_col=user_col,
            item_col=item_col,
            rating_col=rating_col,
        )

        return list(uir_df.itertuples(index=False, name=None))

    def get_uir_data(
        self,
        user_col=None,
        item_col=None,
        rating_col=None,
    ):
        df = self.load_df()
        return self.df_to_cornac_uir(
            df=df,
            user_col=user_col,
            item_col=item_col,
            rating_col=rating_col,
        )

    def train_test_split_df(
        self,
        df=None,
        test_fraction=0.2,
        seed=None,
        user_col=None,
        item_col=None,
        rating_col=None,
    ):
        if seed is None:
            seed = self.seed

        uir_df = self.get_uir_df(
            df=df,
            user_col=user_col,
            item_col=item_col,
            rating_col=rating_col,
        )

        n_obs = len(uir_df)
        rng = np.random.RandomState(seed)
        order = rng.permutation(n_obs)

        users = uir_df["user"].to_numpy()
        items = uir_df["item"].to_numpy()

        user_train_count = Counter(users)
        item_train_count = Counter(items)

        target_test = int(round(test_fraction * n_obs))
        train_mask = np.ones(n_obs, dtype=bool)
        n_test = 0

        for idx in order:
            if n_test >= target_test:
                break

            u = users[idx]
            i = items[idx]

            if user_train_count[u] > 1 and item_train_count[i] > 1:
                train_mask[idx] = False
                user_train_count[u] -= 1
                item_train_count[i] -= 1
                n_test += 1

        train_df = uir_df.loc[train_mask].reset_index(drop=True)
        test_df = uir_df.loc[~train_mask].reset_index(drop=True)

        return train_df, test_df

    def cv_splits_df(
        self,
        train_df,
        n_splits=5,
        val_fraction=0.2,
    ):

        rng = np.random.RandomState(self.seed)
        train_df = train_df.reset_index(drop=True)

        for _ in range(n_splits):
            split_seed = rng.randint(0, 10**9)
            fold_train_df, fold_val_df = self.train_test_split_df(
                df=train_df,
                test_fraction=val_fraction,
                seed=split_seed,
            )
            yield fold_train_df, fold_val_df

    def _ensure_split_cache(self, test_fraction=0.2):
        cache_key = f"split_{test_fraction}"

        if not hasattr(self, "_split_cache"):
            self._split_cache = {}

        if cache_key in self._split_cache:
            return self._split_cache[cache_key]

        uir_df = self.get_uir_df(self.df_obs)

        train_df, test_df = self.train_test_split_df(
            df=uir_df,
            test_fraction=test_fraction,
            seed=self.seed,
        )

        train_pairs = train_df[["user", "item"]].to_numpy(dtype=np.int64)
        test_pairs = test_df[["user", "item"]].to_numpy(dtype=np.int64)

        sparse_train = self.pairs_to_sparse(train_pairs, self.sparse_obs)
        sparse_test = self.pairs_to_sparse(test_pairs, self.sparse_obs)

        self._split_cache[cache_key] = {
            "train_df": train_df.reset_index(drop=True),
            "test_df": test_df.reset_index(drop=True),
            "train_pairs": train_pairs,
            "test_pairs": test_pairs,
            "sparse_train": sparse_train,
            "sparse_test": sparse_test,
        }
        return self._split_cache[cache_key]

    def build_dataset_summary(self):

        return {
            **self.sparse_info,
            **self.df_info,
        }
