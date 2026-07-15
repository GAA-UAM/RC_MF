import os
import json
import argparse
import warnings
import traceback
import faulthandler

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy import sparse

from utils.tuner import Tuner
from utils.logger import Logger
from utils.logger_table import *
from utils.data_manager import DataManager
from utils.metrics import MetricsEvaluator
from utils.model_registry import (
    load_model_spaces,
    get_model_spec,
    build_model,
    build_bo_bounds_tensor,
)

faulthandler.enable(all_threads=True)
warnings.filterwarnings("ignore")

N_JOBS = min(4, int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
MAX_OBS = 5_000_000

DEFAULT_MODELS = [
    "rc_mf",
    "cornac_pmf",
    "cornac_bpr",
    "cornac_itemknn",
    "cornac_mf",
    "cornac_wmf",
    "cornac_mlp",
]


class ExperimentRunner:
    def __init__(
        self,
        dataset,
        use_gridsearch=False,
        use_bo=False,
        epochs=50,
        seed=0,
        model_names=None,
        config_path="config/model_search_spaces.yml",
        results_root="results",
    ):
        if use_gridsearch and use_bo:
            raise ValueError("Choose only one tuning mode.")

        self.dataset = dataset
        self.use_gridsearch = use_gridsearch
        self.use_bo = use_bo
        self.epochs = int(epochs)
        self.seed = int(seed)

        self.results_root = results_root
        self.dataset_dir = os.path.join(self.results_root, self.dataset)

        model_tag = "all_models" if model_names is None else "_".join(model_names)
        self.run_dir = os.path.join(
            self.dataset_dir,
            f"seed_{self.seed}",
            model_tag,
        )

        os.makedirs(self.run_dir, exist_ok=True)

        self.logger = Logger(log_file=os.path.join(self.run_dir, "run_experiment.log"))

        self.data_manager = DataManager(dataset=dataset, seed=seed)
        self.tuner = Tuner(
            data_manager=self.data_manager,
            seed=seed,
            n_jobs=N_JOBS,
        )
        self.evaluator = MetricsEvaluator(ks=(5, 10, 20))

        self.model_spaces = load_model_spaces(config_path)
        self.model_names = model_names or DEFAULT_MODELS

    def _path(self, filename):
        return os.path.join(self.run_dir, filename)

    def _save_csv(self, df, filename):
        df.to_csv(self._path(filename), index=False)

    def _df_to_sparse_matrix(self, df, shape):
        return sparse.csr_matrix(
            (
                df["rating"].to_numpy(dtype=np.float64),
                (
                    df["user"].to_numpy(dtype=np.int64),
                    df["item"].to_numpy(dtype=np.int64),
                ),
            ),
            shape=shape,
            dtype=np.float64,
        )

    @staticmethod
    def _safe_metric(d, key, default=np.nan):
        value = d.get(key, default)
        return default if value is None else value

    def get_tuning_mode(self):
        if self.use_gridsearch:
            return "gridsearch"
        if self.use_bo:
            return "bayesian_optimization"
        return "none"

    def comparison(self, metrics_df):
        df = metrics_df.copy()

        df["rmse"] = pd.to_numeric(df["rmse"], errors="coerce")
        df["ndcg@10"] = pd.to_numeric(df["ndcg@10"], errors="coerce")

        df["rank_rmse"] = df["rmse"].rank(method="min", ascending=True)
        df["rank_ndcg@10"] = df["ndcg@10"].rank(method="min", ascending=False)

        df = df.sort_values(
            by=["rank_rmse", "rank_ndcg@10"],
            ascending=[True, True],
        ).reset_index(drop=True)

        first_cols = ["model", "rmse", "ndcg@10", "rank_rmse", "rank_ndcg@10"]
        other_cols = [c for c in df.columns if c not in first_cols]

        return df[first_cols + other_cols]

    def _fit_model(self, model_name, model_spec, train_df, R_train, train_pairs, R_obs):
        model_type = model_spec["type"]

        if model_type not in {"base", "cornac"}:
            raise ValueError(f"Unknown model type for {model_name}: {model_type}")

        if self.use_gridsearch:
            best_score, best_std, best_params = self._grid_search(
                model_name=model_name,
                model_spec=model_spec,
                model_type=model_type,
                train_df=train_df,
                train_pairs=train_pairs,
                R_obs=R_obs,
            )

        elif self.use_bo:
            best_score, best_std, best_params = self._bayesian_optimize(
                model_name=model_name,
                model_spec=model_spec,
                train_df=train_df,
                train_pairs=train_pairs,
                R_obs=R_obs,
            )

        else:
            best_score, best_std = None, None
            best_params = model_spec.get("default_params", {})

        model = build_model(
            model_spec=model_spec,
            seed=self.seed,
            epochs=self.epochs,
            tuned_params=best_params,
        )

        if model_type == "base":
            model.fit(R_train)
        else:
            model.fit(train_df)

        return model, best_score, best_std, best_params

    def _grid_search(
        self,
        model_name,
        model_spec,
        model_type,
        train_df,
        train_pairs,
        R_obs,
    ):
        if model_type == "base":
            return self.tuner.grid_search(
                model_name=model_name,
                model_spec=model_spec,
                param_grid=model_spec["grid_params"],
                train_pairs=train_pairs,
                R_obs=R_obs,
                epochs=self.epochs,
            )

        return self.tuner.grid_search(
            model_name=model_name,
            model_spec=model_spec,
            param_grid=model_spec["grid_params"],
            train_df=train_df,
            epochs=self.epochs,
        )

    def _bayesian_optimize(
        self,
        model_name,
        model_spec,
        train_df,
        train_pairs,
        R_obs,
    ):
        bo_cfg = model_spec["bo"]
        bounds = build_bo_bounds_tensor(model_spec)

        best_score, best_std, best_params, bo_log = self.tuner.bayesian_optimize(
            model_name=model_name,
            model_spec=model_spec,
            bounds=bounds,
            train_pairs=train_pairs,
            R_obs=R_obs,
            train_df=train_df,
            n_init=bo_cfg.get("n_init", 8),
            n_iter=bo_cfg.get("n_iter", 12),
            epochs=self.epochs,
        )

        self.logger.info(f"{model_name} BO evaluated {len(bo_log)} candidates.")
        return best_score, best_std, best_params

    def _predict_model_on_df(self, model, model_spec, df, R_shape):
        if model_spec["type"] == "base":
            X = self._df_to_sparse_matrix(df, R_shape)
            y_pred = np.asarray(model.predict(X), dtype=np.float64).reshape(-1)
        else:
            y_pred = np.asarray(model.predict(df), dtype=np.float64).reshape(-1)

        return pd.DataFrame(
            {
                "user_id": df["user"].to_numpy(dtype=np.int64),
                "item_id": df["item"].to_numpy(dtype=np.int64),
                "y_true": df["rating"].to_numpy(dtype=np.float64),
                "y_pred": y_pred,
            }
        )

    def _regression_summary(self, model_name, split, pred_df):
        metrics = self.evaluator.regression_metrics(
            pred_df["y_true"].to_numpy(dtype=np.float64),
            pred_df["y_pred"].to_numpy(dtype=np.float64),
        )

        return {
            "dataset": self.dataset,
            "seed": self.seed,
            "model": model_name,
            "split": split,
            "rmse": metrics["rmse"],
            "mse": metrics["mse"],
            "mae": metrics["mae"],
            "r2": metrics["r2"],
            "n_obs": len(pred_df),
        }

    def _ranking_metrics(self, model, model_name, train_df, test_df, all_items):
        default = {
            "precision@5": None,
            "recall@5": None,
            "ndcg@5": None,
            "precision@10": None,
            "recall@10": None,
            "ndcg@10": None,
            "precision@20": None,
            "recall@20": None,
            "ndcg@20": None,
            "n_eval_users": None,
        }

        if not hasattr(model, "score_items"):
            self.logger.info(f"[{model_name}] Ranking skipped: no score_items method.")
            return default, pd.DataFrame()

        try:

            def score_fn(user_id, item_ids):
                return np.asarray(
                    model.score_items(user_id, item_ids),
                    dtype=np.float64,
                ).reshape(-1)

            metrics, per_user_df = self.evaluator.ranking_metrics(
                score_fn=score_fn,
                train_df=train_df,
                test_df=test_df,
                all_items=all_items,
                user_col="user",
                item_col="item",
            )

            return metrics, per_user_df

        except Exception as e:
            self.logger.info(f"[{model_name}] Ranking evaluation skipped: {str(e)}")
            return default, pd.DataFrame()

    def _metrics_row(self, model_name, regression_metrics, ranking_metrics):
        row = {
            "model": model_name,
            "rmse": regression_metrics["rmse"],
            "mse": regression_metrics["mse"],
            "mae": regression_metrics["mae"],
            "r2": regression_metrics["r2"],
        }

        for key in [
            "precision@5",
            "recall@5",
            "ndcg@5",
            "precision@10",
            "recall@10",
            "ndcg@10",
            "precision@20",
            "recall@20",
            "ndcg@20",
            "n_eval_users",
        ]:
            row[key] = self._safe_metric(
                ranking_metrics,
                key,
                default=0 if key == "n_eval_users" else np.nan,
            )

        return row

    def _model_row(
        self,
        model_name,
        train_metrics,
        test_metrics,
        ranking_metrics,
        cv_mean,
        cv_std,
        best_params,
    ):
        row = {
            "model": model_name,
            "train_rmse": train_metrics["rmse"],
            "train_mse": train_metrics["mse"],
            "train_mae": train_metrics["mae"],
            "train_r2": train_metrics["r2"],
            "final_test_rmse": test_metrics["rmse"],
            "final_test_mse": test_metrics["mse"],
            "final_test_mae": test_metrics["mae"],
            "final_test_r2": test_metrics["r2"],
        }

        for key in [
            "precision@5",
            "recall@5",
            "ndcg@5",
            "precision@10",
            "recall@10",
            "ndcg@10",
            "precision@20",
            "recall@20",
            "ndcg@20",
            "n_eval_users",
        ]:
            row[key] = self._safe_metric(
                ranking_metrics,
                key,
                default=0 if key == "n_eval_users" else np.nan,
            )

        row["cv_rmse_mean"] = cv_mean
        row["cv_rmse_std"] = cv_std
        row["best_params"] = best_params

        return row

    def _save_rc_diagnostics(self, model_name, model):
        if model_name != "rc_mf":
            return

        if hasattr(model, "get_residual_diagnostics"):
            residual_df = model.get_residual_diagnostics()
            self._save_csv(residual_df, f"{model_name}_residual_diagnostics.csv")

        if hasattr(model, "get_residual_group_diagnostics"):
            group_df = model.get_residual_group_diagnostics()

            self._save_csv(
                group_df,
                f"{model_name}_residual_group_diagnostics.csv",
            )

            self._save_residual_reduction_summary(model_name, group_df)

            self._save_residual_plots(model_name, group_df)

    def _save_train_test_error_plot(self, error_df):
        if error_df is None or error_df.empty:
            return

        plot_df = error_df[error_df["model"].isin(["cornac_mf", "rc_mf"])]

        if plot_df.empty:
            return

        pivot = plot_df.pivot_table(
            index="model",
            columns="split",
            values="rmse",
            aggfunc="mean",
        )

        if pivot.empty:
            return

        ax = pivot.plot(kind="bar", figsize=(8, 5))
        ax.set_title(f"Train vs Test RMSE - {self.dataset} seed {self.seed}")
        ax.set_xlabel("Model")
        ax.set_ylabel("RMSE")
        ax.legend(title="Split")
        plt.tight_layout()
        plt.savefig(self._path("mf_vs_rc_mf_train_test_rmse.png"), dpi=200)
        plt.close()

    def _save_residual_reduction_summary(self, model_name, group_df):
        if model_name != "rc_mf":
            return

        required = {
            "mean_residual_before",
            "mean_residual_after",
            "rmse_before",
            "rmse_after",
        }

        if group_df is None or group_df.empty:
            return

        if not required.issubset(group_df.columns):
            missing = required - set(group_df.columns)
            self.logger.info(
                f"[{model_name}] Residual reduction summary skipped: missing {missing}"
            )
            return

        df = group_df.copy()

        for col in required:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=list(required))

        if df.empty:
            return

        bias_before = df["mean_residual_before"].abs().mean()
        bias_after = df["mean_residual_after"].abs().mean()

        rmse_before = df["rmse_before"].mean()
        rmse_after = df["rmse_after"].mean()

        bias_reduction_pct = (
            100.0 * (bias_before - bias_after) / bias_before
            if bias_before != 0
            else np.nan
        )

        rmse_reduction_pct = (
            100.0 * (rmse_before - rmse_after) / rmse_before
            if rmse_before != 0
            else np.nan
        )

        summary_df = pd.DataFrame(
            [
                {
                    "dataset": self.dataset,
                    "seed": self.seed,
                    "model": model_name,
                    "bias_before": bias_before,
                    "bias_after": bias_after,
                    "bias_reduction_pct": bias_reduction_pct,
                    "group_rmse_before": rmse_before,
                    "group_rmse_after": rmse_after,
                    "group_rmse_reduction_pct": rmse_reduction_pct,
                    "n_groups": len(df),
                }
            ]
        )

        self._save_csv(
            summary_df,
            f"{model_name}_residual_reduction_summary.csv",
        )

    def _save_residual_plots(self, model_name, group_df):
        required = {
            "user_group",
            "item_group",
            "mean_residual_before",
            "mean_residual_after",
            "rmse_before",
            "rmse_after",
        }

        if group_df is None or group_df.empty:
            return

        if not required.issubset(group_df.columns):
            missing = required - set(group_df.columns)
            self.logger.info(f"[{model_name}] Residual plots skipped: {missing}")
            return

        plot_df = group_df.copy()
        plot_df["group_label"] = (
            "U"
            + plot_df["user_group"].astype(str)
            + "_I"
            + plot_df["item_group"].astype(str)
        )

        self._before_after_barplot(
            df=plot_df,
            before_col="mean_residual_before",
            after_col="mean_residual_after",
            ylabel="Mean residual",
            title=f"Mean residual by user-item group - {self.dataset} seed {self.seed}",
            filename=f"{model_name}_group_mean_residual_before_after.png",
            add_zero_line=True,
        )

        self._before_after_barplot(
            df=plot_df,
            before_col="rmse_before",
            after_col="rmse_after",
            ylabel="Residual RMSE",
            title=f"Residual RMSE by user-item group - {self.dataset} seed {self.seed}",
            filename=f"{model_name}_group_rmse_before_after.png",
            add_zero_line=False,
        )

    def _before_after_barplot(
        self,
        df,
        before_col,
        after_col,
        ylabel,
        title,
        filename,
        add_zero_line=False,
    ):
        x = np.arange(len(df))
        width = 0.35

        plt.figure(figsize=(10, 5))
        plt.bar(x - width / 2, df[before_col], width, label="Before calibration")
        plt.bar(x + width / 2, df[after_col], width, label="After calibration")

        if add_zero_line:
            plt.axhline(0.0, linewidth=1)

        plt.xticks(x, df["group_label"], rotation=45, ha="right")
        plt.title(title)
        plt.xlabel("User activity group × item popularity group")
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self._path(filename), dpi=200)
        plt.close()

    def _log_metrics(self, model_name, train_metrics, test_metrics, ranking_metrics):
        self.logger.info(
            f"{model_name} Train Metrics: "
            f"RMSE={train_metrics['rmse']:.6f}, "
            f"MSE={train_metrics['mse']:.6f}, "
            f"MAE={train_metrics['mae']:.6f}, "
            f"R2={train_metrics['r2']:.6f}"
        )

        self.logger.info(
            f"{model_name} Test Metrics: "
            f"RMSE={test_metrics['rmse']:.6f}, "
            f"MSE={test_metrics['mse']:.6f}, "
            f"MAE={test_metrics['mae']:.6f}, "
            f"R2={test_metrics['r2']:.6f}"
        )

        if ranking_metrics["precision@5"] is None:
            self.logger.info(f"{model_name} Ranking Metrics: unavailable")
            return

        self.logger.info(
            f"{model_name} Ranking Metrics: "
            f"P@5={ranking_metrics['precision@5']:.6f}, "
            f"R@5={ranking_metrics['recall@5']:.6f}, "
            f"NDCG@5={ranking_metrics['ndcg@5']:.6f}, "
            f"P@10={ranking_metrics['precision@10']:.6f}, "
            f"R@10={ranking_metrics['recall@10']:.6f}, "
            f"NDCG@10={ranking_metrics['ndcg@10']:.6f}, "
            f"P@20={ranking_metrics['precision@20']:.6f}, "
            f"R@20={ranking_metrics['recall@20']:.6f}, "
            f"NDCG@20={ranking_metrics['ndcg@20']:.6f}, "
            f"n_eval_users={ranking_metrics['n_eval_users']}"
        )

    def _save_partial_outputs(self, model_rows, metrics_rows, train_test_rows):
        self._save_csv(pd.DataFrame(model_rows), "model_results_partial.csv")
        self._save_csv(pd.DataFrame(metrics_rows), "metrics_summary_partial.csv")
        self._save_csv(
            pd.DataFrame(train_test_rows),
            "train_test_error_summary_partial.csv",
        )

    def _save_final_outputs(
        self,
        model_rows,
        metrics_rows,
        train_test_rows,
        best_params_by_model,
    ):
        model_results_df = pd.DataFrame(model_rows)
        metrics_df = pd.DataFrame(metrics_rows)
        train_test_df = pd.DataFrame(train_test_rows)

        self._save_csv(model_results_df, "model_results.csv")
        self._save_csv(metrics_df, "metrics_summary.csv")
        self._save_csv(train_test_df, "train_test_error_summary.csv")

        self._save_train_test_error_plot(train_test_df)

        with open(self._path(f"{self.dataset}_best_params.json"), "w") as f:
            json.dump(to_jsonable(best_params_by_model), f, indent=4)

        if not metrics_df.empty and "rmse" in metrics_df.columns:
            comparison_df = self.comparison(metrics_df)
            self._save_csv(comparison_df, f"{self.dataset}_model_comparison.csv")
        else:
            self.logger.info(
                "Comparison table skipped because no valid model metrics were generated."
            )

    def run(self):
        model_rows = []
        metrics_rows = []
        train_test_rows = []
        fitted_models = {}
        best_params_by_model = {}

        df_obs, sparse_obs = self.data_manager.load_data()
        obs, y_true_sparse_df = self.data_manager.get_sparse_info()
        train_df, test_df = self.data_manager.get_df_info()

        self.logger.info(
            f"{self.dataset} Loaded sparse matrix with shape={sparse_obs.shape}"
        )
        self.logger.info(
            f"{self.dataset} Loaded data frame matrix with shape={df_obs.shape}"
        )

        if self.use_gridsearch and len(obs) > MAX_OBS:
            self.logger.info(
                f"Grid search disabled because n_obs={len(obs)} exceeds MAX_OBS={MAX_OBS}."
            )
            self.use_gridsearch = False

        if self.use_bo and len(obs) > MAX_OBS:
            self.logger.info(
                f"Bayesian optimization disabled because n_obs={len(obs)} exceeds MAX_OBS={MAX_OBS}."
            )
            self.use_bo = False

        R_train = self._df_to_sparse_matrix(train_df, sparse_obs.shape)
        train_pairs = train_df[["user", "item"]].to_numpy(dtype=np.int64)
        all_items = np.arange(sparse_obs.shape[1], dtype=np.int64)

        self._save_csv(y_true_sparse_df, "y_true_sparse.csv")
        self._save_csv(
            test_df.rename(
                columns={"user": "user_id", "item": "item_id", "rating": "y_true"}
            ),
            "y_true_df.csv",
        )

        dataset_summary = self.data_manager.build_dataset_summary()

        for model_name in self.model_names:
            try:
                model_spec = get_model_spec(self.model_spaces, model_name)
                self.logger.info(f"[{model_name}] spec_type={model_spec['type']}")

                model, cv_mean, cv_std, best_params = self._fit_model(
                    model_name=model_name,
                    model_spec=model_spec,
                    train_df=train_df,
                    R_train=R_train,
                    train_pairs=train_pairs,
                    R_obs=sparse_obs,
                )

                best_params_by_model[model_name] = best_params
                fitted_models[model_name] = model

                train_pred = self._predict_model_on_df(
                    model, model_spec, train_df, sparse_obs.shape
                )
                test_pred = self._predict_model_on_df(
                    model, model_spec, test_df, sparse_obs.shape
                )

                self._save_csv(train_pred, f"{model_name}_train_predictions.csv")
                self._save_csv(test_pred, f"{model_name}_test_predictions.csv")
                self._save_csv(
                    test_pred[["user_id", "item_id", "y_pred"]],
                    f"{model_name}_y_pred.csv",
                )

                train_metrics = self._regression_summary(
                    model_name, "train", train_pred
                )
                test_metrics = self._regression_summary(model_name, "test", test_pred)

                train_test_rows.extend([train_metrics, test_metrics])

                self._save_rc_diagnostics(model_name, model)

                ranking_metrics, ranking_per_user_df = self._ranking_metrics(
                    model=model,
                    model_name=model_name,
                    train_df=train_df,
                    test_df=test_df,
                    all_items=all_items,
                )

                self._save_csv(
                    ranking_per_user_df,
                    f"{model_name}_ranking_per_user.csv",
                )

                metrics_rows.append(
                    self._metrics_row(model_name, test_metrics, ranking_metrics)
                )

                model_rows.append(
                    self._model_row(
                        model_name=model_name,
                        train_metrics=train_metrics,
                        test_metrics=test_metrics,
                        ranking_metrics=ranking_metrics,
                        cv_mean=cv_mean,
                        cv_std=cv_std,
                        best_params=best_params,
                    )
                )

                self._log_metrics(
                    model_name,
                    train_metrics,
                    test_metrics,
                    ranking_metrics,
                )

                self._save_partial_outputs(
                    model_rows,
                    metrics_rows,
                    train_test_rows,
                )

                self.logger.info(f"[{model_name}] Finished successfully")

            except Exception as e:
                self.logger.error(
                    f"[ERROR] model={model_name} failed with error: {str(e)}"
                )
                self.logger.info(traceback.format_exc())
                continue

        experiment_summary = {
            "dataset": self.dataset,
            "seed": self.seed,
            "epochs": self.epochs,
            "tuning_mode": self.get_tuning_mode(),
            "gridsearch_enabled": self.use_gridsearch,
            "bo_enabled": self.use_bo,
            "models_fitted": ",".join(fitted_models.keys()),
            "n_jobs": N_JOBS,
        }

        run_summary = {
            "experiment_summary": experiment_summary,
            "dataset_summary": dataset_summary,
        }

        self.logger.info(dict_to_table("RUN SUMMARY", run_summary))

        self.logger.info(
            list_of_dicts_to_table(
                "MODEL RESULTS",
                model_rows,
                columns=[
                    "model",
                    "train_rmse",
                    "train_mse",
                    "train_mae",
                    "train_r2",
                    "final_test_rmse",
                    "final_test_mse",
                    "final_test_mae",
                    "final_test_r2",
                    "precision@5",
                    "recall@5",
                    "ndcg@5",
                    "precision@10",
                    "recall@10",
                    "ndcg@10",
                    "precision@20",
                    "recall@20",
                    "ndcg@20",
                    "n_eval_users",
                    "cv_rmse_mean",
                    "cv_rmse_std",
                    "best_params",
                ],
            )
        )

        self._save_final_outputs(
            model_rows=model_rows,
            metrics_rows=metrics_rows,
            train_test_rows=train_test_rows,
            best_params_by_model=best_params_by_model,
        )

        self.logger.info("[DONE] Experiment finished successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use_gridsearch", action="store_true")
    parser.add_argument("--use_bo", action="store_true")

    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
    )

    parser.add_argument(
        "--config_path",
        type=str,
        default="config/model_search_spaces.yml",
    )

    args = parser.parse_args()

    runner = ExperimentRunner(
        dataset=args.dataset,
        use_gridsearch=args.use_gridsearch,
        use_bo=args.use_bo,
        epochs=args.epochs,
        seed=args.seed,
        model_names=args.models,
        config_path=args.config_path,
    )

    runner.run()
