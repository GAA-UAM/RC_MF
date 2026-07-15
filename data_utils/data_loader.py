import os
import json
import numpy as np
import pandas as pd
from .amazon_util import read_amazon
from .ml100k_util import read_data_ml100k
from .synthetic_util import generate_synthetic_ratings


class DataLoader:
    def __init__(self, dataset, seed):
        self.dataset = dataset
        self.seed = seed

    def __call__(self, *args, **kwds):

        if self.dataset == "synthetic":
            with open(os.path.join("data_utils", "synthetic_config.json"), "r") as f:
                config = json.load(f)

            (
                data,
                R_obs,
                pairs,
                num_users,
                num_items,
                metadata,
            ) = generate_synthetic_ratings(
                num_users=config["num_users"],
                num_tasks=config["num_tasks"],
                num_item_clusters=config["num_item_clusters"],
                num_user_clusters=config["num_user_clusters"],
                sparsity=config["sparsity"],
                latent_dim=config["latent_dim"],
                seed=self.seed,
            )

            rows, cols = R_obs.nonzero()
            values = (
                R_obs[rows, cols].A1
                if hasattr(R_obs[rows, cols], "A1")
                else np.asarray(R_obs[rows, cols]).ravel()
            )

            data = pd.DataFrame(
                {
                    "user_idx": rows.astype(np.int64),
                    "item_idx": cols.astype(np.int64),
                    "rating": values.astype(np.float32),
                }
            )

        elif self.dataset == "ml100k":
            (
                data,
                R_obs,
                pairs,
                num_users,
                num_items,
            ) = read_data_ml100k()

        else:

            (
                data,
                R_obs,
                pairs,
                num_users,
                num_items,
            ) = read_amazon(self.dataset)

        return data, R_obs
