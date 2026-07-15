import torch
import traceback
import numpy as np
from joblib import Parallel, delayed
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.fit import fit_gpytorch_mll
from sklearn.model_selection import ParameterGrid
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import LogExpectedImprovement
from botorch.models.transforms.outcome import Standardize

from utils.model_registry import build_model, decode_bo_params


class Tuner:
    def __init__(self, data_manager, seed=0, n_jobs=1):
        self.data_manager = data_manager
        self.seed = seed
        self.n_jobs = n_jobs

    def fit_surrogate_model(self, train_X, train_Y):
        gp = SingleTaskGP(train_X, train_Y, outcome_transform=Standardize(m=1))
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        return gp

    def _rmse(self, y_true, y_pred):
        y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
        y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    def _prepare_params(self, model_spec, params, epochs):
        params = dict(params or {})
        class_name = model_spec["class_name"]
        epochs = int(epochs)

        if model_spec["type"] == "base":
            params["n_epochs"] = epochs

        elif model_spec["type"] == "cornac":
            if class_name in {"MF", "PMF", "BPR", "WMF"}:
                params["max_iter"] = epochs

            elif class_name == "MLP":
                params["num_epochs"] = epochs

            elif class_name == "ItemKNN":
                pass

        return params

    def _df_to_sparse_matrix(self, df, shape):
        from scipy import sparse

        rows = df["user"].to_numpy(dtype=np.int64)
        cols = df["item"].to_numpy(dtype=np.int64)
        vals = df["rating"].to_numpy(dtype=np.float64)

        return sparse.csr_matrix(
            (vals, (rows, cols)),
            shape=shape,
            dtype=np.float64,
        )

    def _evaluate_base_model_cv(
        self,
        model_name,
        model_spec,
        params,
        train_pairs,
        R_obs,
        train_df=None,
        epochs=20,
    ):
        params = self._prepare_params(model_spec, params, epochs=epochs)

        # RC-MF is rating-dataframe based internally, so use dataframe CV.
        if model_spec["class_name"] == "ResidualCalibratedMF":
            if train_df is None:
                raise ValueError("ResidualCalibratedMF CV requires train_df.")

            n_splits = 5 if len(train_df) >= 5 else 2
            fold_rmses = []

            for fold_id, (fold_train_df, fold_val_df) in enumerate(
                self.data_manager.cv_splits_df(
                    train_df=train_df,
                    n_splits=n_splits,
                    val_fraction=1.0 / n_splits,
                )
            ):
                if len(fold_val_df) == 0:
                    continue

                fold_train_df = self.data_manager.get_uir_df(fold_train_df)
                fold_val_df = self.data_manager.get_uir_df(fold_val_df)

                R_fold_train = self._df_to_sparse_matrix(fold_train_df, R_obs.shape)
                R_fold_val = self._df_to_sparse_matrix(fold_val_df, R_obs.shape)

                model = build_model(
                    model_spec=model_spec,
                    seed=self.seed + fold_id,
                    epochs=epochs,
                    tuned_params=params,
                )

                model.fit(R_fold_train)
                pred = model.predict(R_fold_val)
                y_true = fold_val_df["rating"].to_numpy(dtype=np.float64)

                fold_rmses.append(self._rmse(y_true, pred))

                del model, R_fold_train, R_fold_val, pred

            if len(fold_rmses) == 0:
                raise ValueError(
                    "No valid CV folds were generated for ResidualCalibratedMF."
                )

            mean_rmse = float(np.mean(fold_rmses))
            std_rmse = float(np.std(fold_rmses))

            clean_params = params.copy()
            clean_params.pop("n_epochs", None)

            return mean_rmse, std_rmse, clean_params

        # Original path for other base models.
        n_splits = min(5, max(2, len(train_pairs)))
        fold_rmses = []

        for fold_id, (fold_train_pairs, fold_val_pairs) in enumerate(
            self.data_manager.cv_splits(
                train_pairs=train_pairs,
                n_splits=n_splits,
                val_fraction=1.0 / n_splits,
                seed=self.seed,
            )
        ):
            if len(fold_val_pairs) == 0:
                continue

            R_fold_train = self.data_manager.pairs_to_sparse(fold_train_pairs, R_obs)
            R_fold_val = self.data_manager.pairs_to_sparse(fold_val_pairs, R_obs)

            model = build_model(
                model_spec=model_spec,
                seed=self.seed + fold_id,
                epochs=epochs,
                tuned_params=params,
            )

            model.fit(R_fold_train)
            pred = model.predict(R_fold_val)
            y_true = R_fold_val.data.astype(np.float64)

            fold_rmses.append(self._rmse(y_true, pred))

            del model, R_fold_train, R_fold_val, pred

        if len(fold_rmses) == 0:
            raise ValueError("No valid CV folds were generated for custom model.")

        mean_rmse = float(np.mean(fold_rmses))
        std_rmse = float(np.std(fold_rmses))

        clean_params = params.copy()
        clean_params.pop("n_epochs", None)

        return mean_rmse, std_rmse, clean_params

    def _evaluate_cornac_model_cv(
        self,
        model_name,
        model_spec,
        params,
        train_df,
        epochs=20,
    ):
        n_splits = 5 if len(train_df) >= 5 else 2
        params = self._prepare_params(model_spec, params, epochs=epochs)

        fold_rmses = []

        for fold_id, (fold_train_df, fold_val_df) in enumerate(
            self.data_manager.cv_splits_df(
                train_df=train_df,
                n_splits=n_splits,
                val_fraction=1.0 / n_splits,
            )
        ):
            if len(fold_val_df) == 0:
                continue

            fold_train_df = self.data_manager.get_uir_df(fold_train_df)
            fold_val_df = self.data_manager.get_uir_df(fold_val_df)

            model = build_model(
                model_spec=model_spec,
                seed=self.seed + fold_id,
                epochs=epochs,
                tuned_params=params,
            )

            model.fit(fold_train_df)
            pred = model.predict(fold_val_df)
            y_true = fold_val_df["rating"].to_numpy(dtype=np.float64)

            fold_rmse = self._rmse(y_true, pred)
            fold_rmses.append(fold_rmse)

            del model, pred

        if len(fold_rmses) == 0:
            raise ValueError("No valid CV folds were generated for Cornac model.")

        mean_rmse = float(np.mean(fold_rmses))
        std_rmse = float(np.std(fold_rmses))

        clean_params = params.copy()
        clean_params.pop("max_iter", None)
        clean_params.pop("num_epochs", None)

        return mean_rmse, std_rmse, clean_params

    def evaluate_model_cv(
        self,
        model_name,
        model_spec,
        params,
        train_pairs=None,
        R_obs=None,
        train_df=None,
        epochs=20,
    ):
        model_type = model_spec["type"]

        if model_type == "base":
            if train_pairs is None or R_obs is None:
                raise ValueError("Custom model CV requires train_pairs and R_obs.")
            return self._evaluate_base_model_cv(
                model_name=model_name,
                model_spec=model_spec,
                params=params,
                train_pairs=train_pairs,
                R_obs=R_obs,
                train_df=train_df,
                epochs=epochs,
            )

        elif model_type == "cornac":
            if train_df is None:
                raise ValueError("Cornac model CV requires train_df.")
            return self._evaluate_cornac_model_cv(
                model_name=model_name,
                model_spec=model_spec,
                params=params,
                train_df=train_df,
                epochs=epochs,
            )

        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def grid_search(
        self,
        model_name,
        model_spec,
        param_grid,
        train_pairs=None,
        R_obs=None,
        train_df=None,
        epochs=20,
    ):
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self.evaluate_model_cv)(
                model_name=model_name,
                model_spec=model_spec,
                params=params,
                train_pairs=train_pairs,
                R_obs=R_obs,
                train_df=train_df,
                epochs=epochs,
            )
            for params in ParameterGrid(param_grid)
        )

        best_score, best_std, best_params = min(results, key=lambda x: x[0])
        return best_score, best_std, best_params

    def bayesian_optimize(
        self,
        model_name,
        model_spec,
        bounds,
        train_pairs=None,
        R_obs=None,
        train_df=None,
        n_init=8,
        n_iter=12,
        epochs=20,
    ):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        dim = bounds.shape[1]
        X_list = []
        Y_list = []
        eval_log = []

        def evaluate_candidate(x_tensor):
            x_np = x_tensor.detach().cpu().numpy().reshape(-1)
            params = decode_bo_params(model_spec, x_np)

            try:
                mean_rmse, std_rmse, used_params = self.evaluate_model_cv(
                    model_name=model_name,
                    model_spec=model_spec,
                    params=params,
                    train_pairs=train_pairs,
                    R_obs=R_obs,
                    train_df=train_df,
                    epochs=epochs,
                )

                if not np.isfinite(mean_rmse):
                    mean_rmse = 1e6
                if not np.isfinite(std_rmse):
                    std_rmse = 0.0

                score = -mean_rmse

            except Exception as e:
                print(
                    f"[BO WARNING] Candidate failed for {model_name} with params={params}"
                )
                print(f"[BO WARNING] Error: {repr(e)}")
                print(traceback.format_exc())

                mean_rmse = 1e6
                std_rmse = 0.0
                used_params = params
                score = -mean_rmse

            return score, mean_rmse, std_rmse, used_params

        # random initialization
        for i in range(n_init):
            x = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(
                dim, dtype=torch.double
            )

            score, mean_rmse, std_rmse, used_params = evaluate_candidate(x)

            X_list.append(x.unsqueeze(0))
            Y_list.append(torch.tensor([[score]], dtype=torch.double))
            eval_log.append(
                {
                    "iter": i,
                    "score": score,
                    "cv_rmse_mean": mean_rmse,
                    "cv_rmse_std": std_rmse,
                    "params": used_params,
                }
            )

        # BO
        for i in range(n_iter):
            train_X = torch.cat(X_list, dim=0)
            train_Y = torch.cat(Y_list, dim=0)

            mask = torch.isfinite(train_Y).squeeze(-1)
            train_X = train_X[mask]
            train_Y = train_Y[mask]

            if train_Y.numel() == 0:
                raise ValueError("All BO observations are non-finite.")

            gp = self.fit_surrogate_model(train_X, train_Y)
            best_f = train_Y.max().item()
            acq = LogExpectedImprovement(gp, best_f=best_f)

            candidate, _ = optimize_acqf(
                acq_function=acq,
                bounds=bounds,
                q=1,
                num_restarts=10,
                raw_samples=64,
            )

            candidate = candidate.detach().view(-1).double()

            score, mean_rmse, std_rmse, used_params = evaluate_candidate(candidate)

            X_list.append(candidate.unsqueeze(0))
            Y_list.append(torch.tensor([[score]], dtype=torch.double))
            eval_log.append(
                {
                    "iter": n_init + i,
                    "score": score,
                    "cv_rmse_mean": mean_rmse,
                    "cv_rmse_std": std_rmse,
                    "params": used_params,
                }
            )

        valid_rows = [
            r
            for r in eval_log
            if np.isfinite(r["cv_rmse_mean"]) and r["cv_rmse_mean"] < 1e6
        ]

        if len(valid_rows) == 0:
            raise ValueError(f"All BO candidates failed for model {model_name}.")

        best_row = min(valid_rows, key=lambda r: r["cv_rmse_mean"])

        best_params = best_row["params"].copy()
        best_params.pop("n_epochs", None)
        best_params.pop("max_iter", None)
        best_params.pop("num_epochs", None)

        return (
            best_row["cv_rmse_mean"],
            best_row["cv_rmse_std"],
            best_params,
            eval_log,
        )
