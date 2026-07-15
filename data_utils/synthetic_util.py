import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix


def generate_synthetic_ratings(
    num_users=1000,
    num_tasks=500,
    num_item_clusters=10,
    num_user_clusters=5,
    sparsity=0.6,
    rating_min=1,
    rating_max=5,
    noise_std=0.1,
    latent_dim=10,
    item_cluster_scale=3.0,
    user_cluster_scale=3.0,
    seed=0,
):
    rng = np.random.RandomState(seed)

    # True clusters
    task_labels = rng.randint(0, num_item_clusters, size=num_tasks)
    user_labels = rng.randint(0, num_user_clusters, size=num_users)

    # True latent factors
    A_true = rng.normal(0, user_cluster_scale, size=(num_user_clusters, latent_dim))
    B_true = rng.normal(0, 0.3, size=(num_users, latent_dim))

    C_true = rng.normal(0, item_cluster_scale, size=(num_item_clusters, latent_dim))
    D_true = rng.normal(0, 0.3, size=(num_tasks, latent_dim))

    U_true = A_true[user_labels] + B_true
    V_true = C_true[task_labels] + D_true

    # Full dense ratings
    R_full = U_true @ V_true.T
    R_full += rng.normal(0, noise_std, size=R_full.shape)

    # Scale to [rating_min, rating_max]
    R_full = (R_full - R_full.min()) / (R_full.max() - R_full.min() + 1e-12)
    R_full = rating_min + R_full * (rating_max - rating_min)
    R_full = R_full.astype(np.float32, copy=False)

    # Sparsify
    observed_fraction = 1.0 - sparsity
    rates_per_user = int(observed_fraction * num_tasks)

    mask = rng.rand(num_users, num_tasks) < observed_fraction

    for u in range(num_users):
        idx = rng.choice(
            num_tasks,
            size=max(1, min(rates_per_user, num_tasks)),
            replace=False,
        )
        mask[u, idx] = True

    rows, cols = np.where(mask)
    values = R_full[rows, cols].astype(np.float32, copy=False)

    pairs = np.column_stack((rows, cols)).astype(np.int64, copy=False)

    df = pd.DataFrame(
        {
            "user": rows.astype(np.int64, copy=False),
            "item": cols.astype(np.int64, copy=False),
            "rating": values,
            "user_cluster": user_labels[rows].astype(np.int64, copy=False),
            "item_cluster": task_labels[cols].astype(np.int64, copy=False),
        }
    )

    num_users_eff = num_users
    num_items_eff = num_tasks

    R_obs = coo_matrix(
        (values, (rows, cols)),
        shape=(num_users_eff, num_items_eff),
        dtype=np.float32,
    ).tocsr()

    sparse_vals = np.asarray(R_obs[rows, cols]).reshape(-1)
    if len(df) != R_obs.nnz:
        raise ValueError(
            f"Mismatch: dataframe has {len(df)} rows but sparse matrix has {R_obs.nnz} nonzero entries."
        )
    if not np.allclose(values, sparse_vals):
        raise ValueError("DataFrame ratings and sparse matrix values are not aligned.")

    metadata = {
        "R_full": R_full,
        "task_labels": task_labels,
        "user_labels": user_labels,
        "A_true": A_true,
        "B_true": B_true,
        "C_true": C_true,
        "D_true": D_true,
    }

    return df, R_obs, pairs, num_users_eff, num_items_eff, metadata
