import numpy as np
import pandas as pd
from collections import defaultdict


class MetricsEvaluator:
    def __init__(self, ks=(5, 10, 20)):
        self.ks = tuple(sorted(set(int(k) for k in ks if int(k) > 0)))

    @staticmethod
    def regression_metrics(y_true, y_pred):
        y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
        y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)

        if y_true.shape != y_pred.shape:
            raise ValueError("y_true and y_pred must have the same shape.")

        errors = y_true - y_pred
        mse = float(np.mean(errors**2))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(errors)))

        y_mean = np.mean(y_true)
        ss_res = np.sum(errors**2)
        ss_tot = np.sum((y_true - y_mean) ** 2)
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        return {
            "rmse": rmse,
            "mse": mse,
            "mae": mae,
            "r2": r2,
        }

    @staticmethod
    def _group_items(df, threshold=4):
        grouped = defaultdict(set)
        for row in df.itertuples(index=False):
            if hasattr(row, "rating"):
                if row.rating >= threshold:
                    grouped[row.user].add(row.item)
            else:
                grouped[row.user].add(row.item)
        return grouped

    @staticmethod
    def _precision_at_k(ranked_items, relevant_items, k):
        if k <= 0:
            return 0.0
        top_k = ranked_items[:k]
        if len(top_k) == 0:
            return 0.0
        hits = sum(1 for x in top_k if x in relevant_items)
        return hits / k

    @staticmethod
    def _recall_at_k(ranked_items, relevant_items, k):
        if len(relevant_items) == 0:
            return 0.0
        top_k = ranked_items[:k]
        hits = sum(1 for x in top_k if x in relevant_items)
        return hits / len(relevant_items)

    @staticmethod
    def _dcg_at_k(ranked_items, relevant_items, k):
        dcg = 0.0
        for idx, item in enumerate(ranked_items[:k], start=1):
            if item in relevant_items:
                dcg += 1.0 / np.log2(idx + 1)
        return dcg

    @classmethod
    def _ndcg_at_k(cls, ranked_items, relevant_items, k):
        if len(relevant_items) == 0:
            return 0.0

        dcg = cls._dcg_at_k(ranked_items, relevant_items, k)
        ideal_len = min(k, len(relevant_items))
        ideal_ranking = list(relevant_items)[:ideal_len]
        idcg = cls._dcg_at_k(ideal_ranking, set(ideal_ranking), ideal_len)

        return dcg / idcg if idcg > 0 else 0.0

    def ranking_metrics(
        self,
        score_fn,
        train_df,
        test_df,
        all_items,
        user_col="user",
        item_col="item",
    ):

        required_cols = {user_col, item_col}
        if not required_cols.issubset(train_df.columns):
            raise ValueError(f"train_df must contain columns {required_cols}")
        if not required_cols.issubset(test_df.columns):
            raise ValueError(f"test_df must contain columns {required_cols}")

        train_ui = train_df[[user_col, item_col]].copy()
        # test_ui = test_df[[user_col, item_col]].copy()
        test_ui = test_df[[user_col, item_col, "rating"]].copy()
        test_ui.columns = ["user", "item", "rating"]
        train_ui.columns = ["user", "item"]
        # test_ui.columns = ["user", "item"]

        train_items_by_user = self._group_items(train_ui)
        test_items_by_user = self._group_items(test_ui, threshold=4)

        train_users = set(train_items_by_user.keys())
        test_users = set(test_items_by_user.keys())

        # users seen in both train and test
        eval_users = sorted(train_users & test_users)

        all_items = np.asarray(sorted(set(all_items)))

        results = {f"precision@{k}": [] for k in self.ks}
        results.update({f"recall@{k}": [] for k in self.ks})
        results.update({f"ndcg@{k}": [] for k in self.ks})

        user_rows = []

        for user in eval_users:
            train_items = train_items_by_user[user]
            relevant_items = test_items_by_user[user]

            # Full-catalog ranking excluding user's training items
            candidate_items = np.array(
                [item for item in all_items if item not in train_items]
            )

            if len(candidate_items) == 0:
                continue

            scores = np.asarray(
                score_fn(user, candidate_items), dtype=np.float64
            ).reshape(-1)

            if not np.all(np.isfinite(scores)):
                scores = np.nan_to_num(scores, nan=-np.inf, posinf=1e9, neginf=-1e9)

            if len(scores) != len(candidate_items):
                raise ValueError(
                    f"score_fn returned {len(scores)} scores, "
                    f"but {len(candidate_items)} candidates were provided for user={user}."
                )

            order = np.argsort(-scores)  # descending
            ranked_items = candidate_items[order].tolist()

            row = {
                "user": user,
                "n_relevant": len(relevant_items),
                "n_candidates": len(candidate_items),
            }

            for k in self.ks:
                p = self._precision_at_k(ranked_items, relevant_items, k)
                r = self._recall_at_k(ranked_items, relevant_items, k)
                n = self._ndcg_at_k(ranked_items, relevant_items, k)

                results[f"precision@{k}"].append(p)
                results[f"recall@{k}"].append(r)
                results[f"ndcg@{k}"].append(n)

                row[f"precision@{k}"] = p
                row[f"recall@{k}"] = r
                row[f"ndcg@{k}"] = n

            user_rows.append(row)

        summary = {}
        for key, values in results.items():
            summary[key] = float(np.mean(values)) if values else 0.0

        summary["n_eval_users"] = len(user_rows)

        per_user_df = pd.DataFrame(user_rows)
        return summary, per_user_df
