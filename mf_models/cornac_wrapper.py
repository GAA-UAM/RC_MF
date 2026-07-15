import numpy as np
from cornac.data import Dataset


def get_cornac_model_class(class_name):
    if class_name == "PMF":
        from cornac.models import PMF

        return PMF
    elif class_name == "BPR":
        from cornac.models import BPR

        return BPR
    elif class_name == "ItemKNN":
        from cornac.models import ItemKNN

        return ItemKNN
    elif class_name == "MF":
        from cornac.models import MF

        return MF
    elif class_name == "WMF":
        from cornac.models import WMF

        return WMF
    elif class_name == "MLP":
        from cornac.models import MLP

        return MLP
    else:
        raise ValueError(f"Unsupported Cornac model: {class_name}")


SUPPORTED_CORNAC_MODELS = {"PMF", "BPR", "ItemKNN", "MF", "WMF", "MLP"}


class CornacModelWrapper:
    def __init__(self, class_name, random_state=0, **params):
        if class_name not in SUPPORTED_CORNAC_MODELS:
            raise ValueError(f"Unsupported Cornac model: {class_name}")

        self.class_name = class_name
        self.random_state = random_state
        self.params = dict(params)

        if "seed" not in self.params:
            self.params["seed"] = random_state

        model_cls = get_cornac_model_class(class_name)
        self.model = model_cls(**self.params)
        self.train_set = None
        self.train_mean_ = None
        self.history_ = None

    def _df_to_uir(self, df):
        return list(df[["user", "item", "rating"]].itertuples(index=False, name=None))

    def fit(self, train_df):
        self.train_mean_ = float(train_df["rating"].mean())
        train_uir = self._df_to_uir(train_df)
        self.train_set = Dataset.from_uir(train_uir)
        self.model.fit(self.train_set)
        self.history_ = {"status": "trained"}
        return self

    def predict(self, test_df):
        preds = []
        for row in test_df.itertuples(index=False):
            user_idx = self.train_set.uid_map.get(row.user)
            item_idx = self.train_set.iid_map.get(row.item)
            if user_idx is None or item_idx is None:
                pred = self.train_mean_
            else:
                pred = self.model.score(user_idx, item_idx)
            preds.append(pred)

        return np.asarray(preds, dtype=np.float64)

    def score_items(self, user_id, item_ids):
        if self.train_set is None:
            raise ValueError("Model must be fitted first.")

        item_ids = np.asarray(item_ids, dtype=np.int64).reshape(-1)

        if item_ids.size == 0:
            return np.asarray([], dtype=np.float64)

        user_idx = self.train_set.uid_map.get(user_id)

        if user_idx is None:
            return np.full(item_ids.shape[0], self.train_mean_, dtype=np.float64)

        preds = []
        for item_id in item_ids:
            item_idx = self.train_set.iid_map.get(item_id)

            if item_idx is None:
                pred = self.train_mean_
            else:
                pred = self.model.score(user_idx, item_idx)

            preds.append(pred)

        return np.asarray(preds, dtype=np.float64)
