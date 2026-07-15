import numpy as np
import pandas as pd
from .base import Base
from sklearn.utils.validation import check_is_fitted


class MF(Base):
    def __init__(
        self,
        n_users=None,
        n_items=None,
        k=20,
        n_epochs=50,
        random_state=0,
        learning_rate=0.003,
        lambda_reg=0.02,
        use_bias=True,
        init_scale=0.01,
        verbose=False,
    ):
        super().__init__(
            k=k,
            n_epochs=n_epochs,
            random_state=random_state,
            verbose=verbose,
        )

        self.n_users = n_users
        self.n_items = n_items
        self.learning_rate = float(learning_rate)
        self.lambda_reg = float(lambda_reg)
        self.use_bias = bool(use_bias)
        self.init_scale = float(init_scale)

    def _input_to_triples(self, X):
        if isinstance(X, pd.DataFrame):
            required = {"user", "item", "rating"}
            missing = required.difference(X.columns)

            if missing:
                raise ValueError(
                    f"DataFrame input must contain columns {required}. "
                    f"Missing columns: {missing}"
                )

            users = X["user"].to_numpy(dtype=np.int64)
            items = X["item"].to_numpy(dtype=np.int64)
            ratings = X["rating"].to_numpy(dtype=np.float64)

            if len(ratings) == 0:
                raise ValueError("Cannot fit MF on empty data.")

            inferred_n_users = int(users.max()) + 1
            inferred_n_items = int(items.max()) + 1

            n_users = (
                int(self.n_users) if self.n_users is not None else inferred_n_users
            )

            n_items = (
                int(self.n_items) if self.n_items is not None else inferred_n_items
            )

            if users.min() < 0 or items.min() < 0:
                raise ValueError("User and item ids must be non-negative integers.")

            if users.max() >= n_users:
                raise ValueError(f"Found user id {users.max()}, but n_users={n_users}.")

            if items.max() >= n_items:
                raise ValueError(f"Found item id {items.max()}, but n_items={n_items}.")

            return users, items, ratings, n_users, n_items

        X = self._validate_data(X)
        X = X.astype(np.float64).tocsr()

        if X.nnz == 0:
            raise ValueError("Cannot fit MF on empty data.")

        coo = X.tocoo()

        users = coo.row.astype(np.int64)
        items = coo.col.astype(np.int64)
        ratings = coo.data.astype(np.float64)

        n_users, n_items = X.shape

        return users, items, ratings, n_users, n_items

    def fit(self, X, y=None):
        users, items, ratings, n_users, n_items = self._input_to_triples(X)

        self.n_users_ = int(n_users)
        self.n_items_ = int(n_items)

        rng = np.random.RandomState(self.random_state)

        self.global_mean_ = float(ratings.mean())

        self.user_bias_ = np.zeros(self.n_users_, dtype=np.float64)
        self.item_bias_ = np.zeros(self.n_items_, dtype=np.float64)

        self.P_ = self.init_scale * rng.normal(size=(self.n_users_, self.k))

        self.Q_ = self.init_scale * rng.normal(size=(self.n_items_, self.k))

        order = np.arange(len(ratings), dtype=np.int64)

        lr = float(self.learning_rate)
        reg = float(self.lambda_reg)

        for epoch in range(self.n_epochs):
            rng.shuffle(order)

            sq_error = 0.0

            for idx in order:
                u = users[idx]
                i = items[idx]
                r = ratings[idx]

                pred = self.global_mean_

                if self.use_bias:
                    pred += self.user_bias_[u] + self.item_bias_[i]

                pred += float(np.dot(self.P_[u], self.Q_[i]))

                err = r - pred
                sq_error += err * err

                p_old = self.P_[u].copy()
                q_old = self.Q_[i].copy()

                if self.use_bias:
                    self.user_bias_[u] += lr * (err - reg * self.user_bias_[u])

                    self.item_bias_[i] += lr * (err - reg * self.item_bias_[i])

                self.P_[u] += lr * (err * q_old - reg * p_old)
                self.Q_[i] += lr * (err * p_old - reg * q_old)

            if self.verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
                rmse = float(np.sqrt(sq_error / len(ratings)))
                msg = f"[MF] epoch={epoch + 1}, train_rmse={rmse:.6f}"

                if hasattr(self, "logger_console"):
                    self.logger_console.info(msg)

        self.is_fitted_ = True
        return self

    def _df_to_pairs(self, df):
        return df[["user", "item"]].to_numpy(dtype=np.int64)

    def predict(self, X):
        check_is_fitted(self, attributes=["is_fitted_"])

        if isinstance(X, pd.DataFrame):
            pairs = self._df_to_pairs(X)
            return self._predict_pairs_impl(pairs)

        X = self._validate_data(X)
        pairs = self._pairs_from_sparse(X)

        return self._predict_pairs_impl(pairs)

    def predict_pairs(self, users, items=None):
        check_is_fitted(self, attributes=["is_fitted_"])

        if items is None:
            pairs = np.asarray(users, dtype=np.int64)

            if pairs.ndim != 2 or pairs.shape[1] != 2:
                raise ValueError("pairs must have shape (n_pairs, 2).")

            return self._predict_pairs_impl(pairs)

        users = np.asarray(users, dtype=np.int64).reshape(-1)
        items = np.asarray(items, dtype=np.int64).reshape(-1)

        if len(users) != len(items):
            raise ValueError(
                f"users and items must have the same length. "
                f"Got len(users)={len(users)} and len(items)={len(items)}."
            )

        pairs = np.column_stack([users, items])

        return self._predict_pairs_impl(pairs)

    def _predict_pairs_impl(self, pairs):
        check_is_fitted(self, attributes=["is_fitted_"])

        pairs = np.asarray(pairs, dtype=np.int64)

        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError("pairs must have shape (n_pairs, 2).")

        users = pairs[:, 0]
        items = pairs[:, 1]

        pred = np.full(len(pairs), self.global_mean_, dtype=np.float64)

        valid = (
            (users >= 0)
            & (users < self.n_users_)
            & (items >= 0)
            & (items < self.n_items_)
        )

        if np.any(valid):
            u = users[valid]
            i = items[valid]

            pred_valid = np.full(len(u), self.global_mean_, dtype=np.float64)

            if self.use_bias:
                pred_valid += self.user_bias_[u] + self.item_bias_[i]

            pred_valid += np.sum(self.P_[u] * self.Q_[i], axis=1)

            pred[valid] = pred_valid

        return pred

    def score_items(self, user_id, item_ids):
        check_is_fitted(self, attributes=["is_fitted_"])

        item_ids = np.asarray(item_ids, dtype=np.int64).reshape(-1)

        if item_ids.size == 0:
            return np.asarray([], dtype=np.float64)

        users = np.full(len(item_ids), int(user_id), dtype=np.int64)
        return self.predict_pairs(users, item_ids)

    def predict_all(self):
        check_is_fitted(self, attributes=["is_fitted_"])

        out = np.empty((self.n_users_, self.n_items_), dtype=np.float64)

        for u in range(self.n_users_):
            out[u] = self.score_items(
                user_id=u,
                item_ids=np.arange(self.n_items_, dtype=np.int64),
            )

        return out

    def reconstruct(self):
        check_is_fitted(self, attributes=["is_fitted_"])

        R_hat = self.global_mean_ + self.P_ @ self.Q_.T

        if self.use_bias:
            R_hat += self.user_bias_[:, None]
            R_hat += self.item_bias_[None, :]

        return R_hat

    def transform(self):
        check_is_fitted(self, attributes=["is_fitted_"])

        return {
            "P": self.P_.copy(),
            "Q": self.Q_.copy(),
            "user_bias": self.user_bias_.copy(),
            "item_bias": self.item_bias_.copy(),
            "global_mean": np.asarray([self.global_mean_], dtype=np.float64),
        }
