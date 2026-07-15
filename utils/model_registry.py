import yaml
import torch
import numpy as np

from mf_models.mf import MF
from mf_models.cornac_wrapper import CornacModelWrapper
from mf_models.residual_calibrated_mf import ResidualCalibratedMF

BASE_MODEL_REGISTRY = {
    "StandardMF": MF,
    "ResidualCalibratedMF": ResidualCalibratedMF,
}


def load_model_spaces(config_path="config/model_search_spaces.yml"):
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(
            f"YAML config at {config_path} must contain a top-level mapping."
        )

    return data


def get_model_spec(model_spaces, model_name):
    if model_name not in model_spaces:
        raise ValueError(f"Model '{model_name}' not found in config.")

    return model_spaces[model_name]


def build_model(model_spec, seed=0, epochs=50, tuned_params=None):
    tuned_params = tuned_params or {}

    fixed_params = model_spec.get("fixed_params", {})
    default_params = model_spec.get("default_params", {})

    params = {**default_params, **fixed_params, **tuned_params}

    seed = int(seed)
    epochs = int(epochs)

    if model_spec["type"] == "base":
        class_name = model_spec["class_name"]

        if class_name not in BASE_MODEL_REGISTRY:
            raise ValueError(
                f"Unsupported base model class '{class_name}'. "
                f"Available classes: {list(BASE_MODEL_REGISTRY.keys())}"
            )

        cls = BASE_MODEL_REGISTRY[class_name]

        params["random_state"] = seed
        params["n_epochs"] = epochs

        return cls(**params)

    if model_spec["type"] == "cornac":
        class_name = model_spec["class_name"]

        params["seed"] = seed

        if class_name in {"MF", "PMF", "BPR", "WMF"}:
            params["max_iter"] = epochs

        elif class_name == "MLP":
            params["num_epochs"] = epochs

        elif class_name == "ItemKNN":
            pass

        return CornacModelWrapper(
            class_name=class_name,
            random_state=seed,
            **params,
        )

    raise ValueError(f"Unknown model type: {model_spec['type']}")

def build_bo_bounds_tensor(model_spec):
    if "bo" not in model_spec:
        raise ValueError(
            f"Model '{model_spec.get('class_name', 'unknown')}' has no BO config."
        )

    bo = model_spec["bo"]
    param_order = bo["param_order"]
    bounds = bo["bounds"]

    missing = [p for p in param_order if p not in bounds]
    if missing:
        raise ValueError(
            f"BO config mismatch for model "
            f"'{model_spec.get('class_name', 'unknown')}': "
            f"missing bounds for parameters {missing}. "
            f"param_order={param_order}, bounds_keys={list(bounds.keys())}"
        )

    lower = [float(bounds[p][0]) for p in param_order]
    upper = [float(bounds[p][1]) for p in param_order]

    return torch.tensor([lower, upper], dtype=torch.double)


def decode_bo_params(model_spec, x):
    bo = model_spec["bo"]
    param_order = bo["param_order"]
    transform = bo.get("transform", {})

    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()

    x = np.asarray(x, dtype=np.float64).reshape(-1)

    if len(x) != len(param_order):
        raise ValueError(
            f"BO candidate has length {len(x)}, but param_order has "
            f"length {len(param_order)}. param_order={param_order}"
        )

    out = {}

    for i, p in enumerate(param_order):
        val = float(x[i])
        tr = transform.get(p, "identity")

        if tr == "pow10":
            out[p] = float(10.0**val)

        elif tr == "int":
            lower = bo["bounds"][p][0]
            upper = bo["bounds"][p][1]
            out[p] = int(np.clip(np.round(val), lower, upper))

        elif tr == "identity":
            out[p] = float(val)

        elif tr == "bool":
            out[p] = bool(int(np.clip(np.round(val), 0, 1)))

        elif tr == "exp15m7":
            out[p] = float(np.exp(val * 15.0 - 7.0))

        else:
            raise ValueError(f"Unknown transform '{tr}' for param '{p}'")

    return out
