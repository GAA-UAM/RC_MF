from .mf import MF
import numpy as np
import pandas as pd
from .base import Base
from sklearn.utils.validation import check_is_fitted


class ResidualCalibratedMF(Base):
    def __init__(
        self,
        k=20,
        n_epochs=50,
        random_state=0,
        verbose=False,
        learning_rate=0.003,
        lambda_reg=0.02,
        use_bias=True,
        init_scale=0.01,
        n_user_groups=3,
        n_item_groups=3,
        lam_user_calib=20.0,
        lam_item_calib=20.0,
        lam_group_calib=10.0,
        calibration_iters=5,
        use_oof_calibration=True,
        oof_folds=3,
        clip_correction=1.0,
        use_global_calib=True,
        use_user_calib=True,
        use_item_calib=True,
        use_group_calib=True,
    ):
        super().__init__(
            k=k,
            n_epochs=n_epochs,
            random_state=random_state,
            verbose=verbose,
        )

        self.learning_rate = float(learning_rate)
        self.lambda_reg = float(lambda_reg)
        self.use_bias = bool(use_bias)
        self.init_scale = float(init_scale)

        self.n_user_groups = int(n_user_groups)
        self.n_item_groups = int(n_item_groups)

        self.lam_user_calib = float(lam_user_calib)
        self.lam_item_calib = float(lam_item_calib)
        self.lam_group_calib = float(lam_group_calib)
        self.calibration_iters = int(calibration_iters)

        self.use_oof_calibration = bool(use_oof_calibration)
        self.oof_folds = int(oof_folds)

        self.clip_correction = (
            None if clip_correction is None else float(clip_correction)
        )

        self.use_global_calib = bool(use_global_calib)
        self.use_user_calib = bool(use_user_calib)
        self.use_item_calib = bool(use_item_calib)
        self.use_group_calib = bool(use_group_calib)

    @staticmethod
    def _sparse_to_df(R):
        R = R.tocoo()
        return pd.DataFrame(
            {
                "user": R.row.astype(np.int64),
                "item": R.col.astype(np.int64),
                "rating": R.data.astype(np.float64),
            }
        )

    @staticmethod
    def _pairs_from_df(df):
        return df[["user", "item"]].to_numpy(dtype=np.int64)

    @staticmethod
    def _axis_counts(R, axis):
        return np.asarray(R.getnnz(axis=axis), dtype=np.float64).reshape(-1)

    @staticmethod
    def _quantile_groups(counts, n_groups):
        counts = np.asarray(counts, dtype=np.float64)
        n = len(counts)

        if n == 0:
            return np.asarray([], dtype=np.int64)

        n_groups = max(1, min(int(n_groups), n))

        if n_groups == 1:
            return np.zeros(n, dtype=np.int64)

        order = np.lexsort((np.arange(n), counts))
        groups = np.empty(n, dtype=np.int64)

        for rank, idx in enumerate(order):
            groups[idx] = min(int(rank * n_groups / n), n_groups - 1)

        return groups

    @staticmethod
    def _ridge_effect(values, ids, n, lam):
        counts = np.bincount(ids, minlength=n).astype(np.float64)
        sums = np.zeros(n, dtype=np.float64)
        np.add.at(sums, ids, values)

        coef = np.zeros(n, dtype=np.float64)
        mask = counts > 0
        coef[mask] = sums[mask] / (counts[mask] + lam)

        return coef

    @staticmethod
    def _rmse(x):
        return float(np.sqrt(np.mean(np.square(x))))

    def _build_groups(self, R):
        user_groups = self._quantile_groups(
            self._axis_counts(R, axis=1),
            self.n_user_groups,
        )

        item_groups = self._quantile_groups(
            self._axis_counts(R, axis=0),
            self.n_item_groups,
        )

        return user_groups, item_groups

    def _group_ids(self, pairs):
        users = pairs[:, 0]
        items = pairs[:, 1]

        return (
            self.user_groups_[users] * self.n_item_groups_used_
            + self.item_groups_[items]
        ).astype(np.int64)

    def _clip(self, pred):
        return np.clip(pred, self.min_rating_, self.max_rating_)

    def _new_backbone(self, seed):
        return MF(
            n_users=self.n_users_,
            n_items=self.n_items_,
            k=self.k,
            n_epochs=self.n_epochs,
            learning_rate=self.learning_rate,
            lambda_reg=self.lambda_reg,
            use_bias=self.use_bias,
            init_scale=self.init_scale,
            random_state=seed,
            verbose=self.verbose,
        )

    def _fit_backbone(self, df, seed):
        return self._new_backbone(seed).fit(df)

    def _oof_predictions(self, df):
        n = len(df)

        if not self.use_oof_calibration or self.oof_folds < 2 or n < self.oof_folds:
            return self._fit_backbone(df, self.random_state).predict(df)

        rng = np.random.RandomState(self.random_state)
        indices = np.arange(n)
        rng.shuffle(indices)

        pred = np.full(n, np.nan, dtype=np.float64)
        folds = np.array_split(indices, self.oof_folds)

        for fold_id, val_idx in enumerate(folds):
            train_mask = np.ones(n, dtype=bool)
            train_mask[val_idx] = False

            train_fold = df.iloc[train_mask].reset_index(drop=True)
            val_fold = df.iloc[val_idx].reset_index(drop=True)

            if train_fold.empty or val_fold.empty:
                continue

            model = self._fit_backbone(
                train_fold,
                seed=self.random_state + fold_id + 1,
            )

            pred[val_idx] = model.predict(val_fold)

        missing = ~np.isfinite(pred)

        if np.any(missing):
            model = self._fit_backbone(df, seed=self.random_state + 999)
            missing_df = df.iloc[np.where(missing)[0]].reset_index(drop=True)
            pred[missing] = model.predict(missing_df)

        return pred

    def _fit_calibrator(self, df, residual):
        pairs = self._pairs_from_df(df)
        users = pairs[:, 0]
        items = pairs[:, 1]
        groups = self._group_ids(pairs)

        alpha = 0.0
        user_calib = np.zeros(self.n_users_, dtype=np.float64)
        item_calib = np.zeros(self.n_items_, dtype=np.float64)
        group_calib = np.zeros(self.n_groups_, dtype=np.float64)

        for _ in range(max(1, self.calibration_iters)):
            if self.use_global_calib:
                current = residual.copy()

                if self.use_user_calib:
                    current -= user_calib[users]

                if self.use_item_calib:
                    current -= item_calib[items]

                if self.use_group_calib:
                    current -= group_calib[groups]

                alpha = float(current.mean())
            else:
                alpha = 0.0

            if self.use_user_calib:
                current = residual.copy()

                if self.use_global_calib:
                    current -= alpha

                if self.use_item_calib:
                    current -= item_calib[items]

                if self.use_group_calib:
                    current -= group_calib[groups]

                user_calib = self._ridge_effect(
                    current,
                    users,
                    self.n_users_,
                    self.lam_user_calib,
                )
            else:
                user_calib = np.zeros(self.n_users_, dtype=np.float64)

            if self.use_item_calib:
                current = residual.copy()

                if self.use_global_calib:
                    current -= alpha

                if self.use_user_calib:
                    current -= user_calib[users]

                if self.use_group_calib:
                    current -= group_calib[groups]

                item_calib = self._ridge_effect(
                    current,
                    items,
                    self.n_items_,
                    self.lam_item_calib,
                )
            else:
                item_calib = np.zeros(self.n_items_, dtype=np.float64)

            if self.use_group_calib:
                current = residual.copy()

                if self.use_global_calib:
                    current -= alpha

                if self.use_user_calib:
                    current -= user_calib[users]

                if self.use_item_calib:
                    current -= item_calib[items]

                group_calib = self._ridge_effect(
                    current,
                    groups,
                    self.n_groups_,
                    self.lam_group_calib,
                )
            else:
                group_calib = np.zeros(self.n_groups_, dtype=np.float64)

        return alpha, user_calib, item_calib, group_calib

    def _correction(self, pairs):
        users = pairs[:, 0]
        items = pairs[:, 1]
        groups = self._group_ids(pairs)

        corr = np.zeros(len(users), dtype=np.float64)

        if self.use_global_calib:
            corr += self.alpha_calib_

        if self.use_user_calib:
            corr += self.user_calib_[users]

        if self.use_item_calib:
            corr += self.item_calib_[items]

        if self.use_group_calib:
            corr += self.group_calib_[groups]

        if self.clip_correction is not None and self.clip_correction > 0:
            corr = np.clip(corr, -self.clip_correction, self.clip_correction)

        return corr

    def _build_residual_diagnostics(self, df, residual_before, residual_after):
        pairs = self._pairs_from_df(df)

        out = pd.DataFrame(
            {
                "user": pairs[:, 0],
                "item": pairs[:, 1],
                "rating": df["rating"].to_numpy(dtype=np.float64),
                "user_group": self.user_groups_[pairs[:, 0]],
                "item_group": self.item_groups_[pairs[:, 1]],
                "group_id": self._group_ids(pairs),
                "residual_before": residual_before,
                "residual_after": residual_after,
            }
        )

        out["abs_residual_before"] = np.abs(out["residual_before"])
        out["abs_residual_after"] = np.abs(out["residual_after"])

        return out

    def _build_group_diagnostics(self, residual_df):
        group_df = (
            residual_df.groupby(["user_group", "item_group", "group_id"])
            .agg(
                n_obs=("rating", "size"),
                mean_residual_before=("residual_before", "mean"),
                mean_residual_after=("residual_after", "mean"),
                mae_before=("abs_residual_before", "mean"),
                mae_after=("abs_residual_after", "mean"),
                rmse_before=("residual_before", self._rmse),
                rmse_after=("residual_after", self._rmse),
            )
            .reset_index()
        )

        group_df["abs_mean_residual_before"] = group_df["mean_residual_before"].abs()
        group_df["abs_mean_residual_after"] = group_df["mean_residual_after"].abs()

        return group_df

    def fit(self, X, y=None):
        X = self._validate_data(X)
        R = X.astype(np.float64).tocsr()

        if R.nnz == 0:
            raise ValueError("Cannot fit ResidualCalibratedMF on empty data.")

        self.n_users_, self.n_items_ = R.shape

        train_df = self._sparse_to_df(R)
        ratings = train_df["rating"].to_numpy(dtype=np.float64)

        self.global_mean_ = float(ratings.mean())
        self.min_rating_ = float(ratings.min())
        self.max_rating_ = float(ratings.max())

        self.user_groups_, self.item_groups_ = self._build_groups(R)
        self.n_user_groups_used_ = int(self.user_groups_.max()) + 1
        self.n_item_groups_used_ = int(self.item_groups_.max()) + 1
        self.n_groups_ = self.n_user_groups_used_ * self.n_item_groups_used_

        oof_pred = self._oof_predictions(train_df)
        residual_before = ratings - oof_pred

        (
            self.alpha_calib_,
            self.user_calib_,
            self.item_calib_,
            self.group_calib_,
        ) = self._fit_calibrator(train_df, residual_before)

        self.backbone_ = self._fit_backbone(
            train_df,
            seed=self.random_state + 12345,
        )

        pairs = self._pairs_from_df(train_df)
        residual_after = residual_before - self._correction(pairs)

        self.residual_diagnostics_df_ = self._build_residual_diagnostics(
            train_df,
            residual_before,
            residual_after,
        )

        self.residual_group_diagnostics_df_ = self._build_group_diagnostics(
            self.residual_diagnostics_df_
        )

        self.calibration_info_ = self._calibration_info(
            residual_before,
            residual_after,
        )

        self.fairgrad_info_ = self.calibration_info_

        self.U_ = self.backbone_.P_
        self.V_ = self.backbone_.Q_

        return self


    def _calibration_info(self, residual_before, residual_after):
        group_df = self.residual_group_diagnostics_df_

        return {
            "method": "RC-MF",
            "k": self.k,
            "n_epochs": self.n_epochs,
            "learning_rate": self.learning_rate,
            "lambda_reg": self.lambda_reg,

            "use_global_calib": self.use_global_calib,
            "use_user_calib": self.use_user_calib,
            "use_item_calib": self.use_item_calib,
            "use_group_calib": self.use_group_calib,
            "use_oof_calibration": self.use_oof_calibration,

            "n_user_groups": self.n_user_groups_used_,
            "n_item_groups": self.n_item_groups_used_,
            "n_groups": self.n_groups_,
            "oof_folds": self.oof_folds,

            "lam_user_calib": self.lam_user_calib,
            "lam_item_calib": self.lam_item_calib,
            "lam_group_calib": self.lam_group_calib,
            "clip_correction": self.clip_correction,

            "oof_residual_rmse_before": self._rmse(residual_before),
            "oof_residual_rmse_after_calibration": self._rmse(residual_after),

            "mean_abs_group_residual_before": float(
                group_df["abs_mean_residual_before"].mean()
            ),
            "mean_abs_group_residual_after": float(
                group_df["abs_mean_residual_after"].mean()
            ),
            "mean_group_rmse_before": float(group_df["rmse_before"].mean()),
            "mean_group_rmse_after": float(group_df["rmse_after"].mean()),
        }


    def _predict_pairs_impl(self, pairs):
        check_is_fitted(
            self,
            attributes=[
                "backbone_",
                "alpha_calib_",
                "user_calib_",
                "item_calib_",
                "group_calib_",
            ],
        )

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
            valid_pairs = pairs[valid]
            valid_users = valid_pairs[:, 0]
            valid_items = valid_pairs[:, 1]

            pred[valid] = self.backbone_.predict_pairs(
                valid_users, valid_items
            ) + self._correction(valid_pairs)

        return self._clip(pred)

    def predict(self, X):
        X = self._validate_data(X)
        pairs = self._pairs_from_sparse(X)
        return self._predict_pairs_impl(pairs)

    def score_items(self, user_id, item_ids):
        check_is_fitted(self, attributes=["backbone_"])

        item_ids = np.asarray(item_ids, dtype=np.int64).reshape(-1)

        if item_ids.size == 0:
            return np.asarray([], dtype=np.float64)

        pairs = np.column_stack(
            [
                np.full(len(item_ids), int(user_id), dtype=np.int64),
                item_ids,
            ]
        )

        return self._predict_pairs_impl(pairs)

    def predict_all(self):
        check_is_fitted(self, attributes=["backbone_"])

        out = np.empty((self.n_users_, self.n_items_), dtype=np.float64)

        for u in range(self.n_users_):
            out[u] = self.score_items(u, np.arange(self.n_items_, dtype=np.int64))

        return out

    def transform(self):
        check_is_fitted(self, attributes=["group_calib_"])
        return {
            "alpha_calib": np.asarray([self.alpha_calib_], dtype=np.float64),
            "user_calib": self.user_calib_.copy(),
            "item_calib": self.item_calib_.copy(),
            "group_calib": self.group_calib_.copy(),
        }

    def get_residual_diagnostics(self):
        check_is_fitted(self, attributes=["residual_diagnostics_df_"])
        return self.residual_diagnostics_df_.copy()

    def get_residual_group_diagnostics(self):
        check_is_fitted(self, attributes=["residual_group_diagnostics_df_"])
        return self.residual_group_diagnostics_df_.copy()
