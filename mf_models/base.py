import numpy as np
from scipy import sparse
from utils.logger import Logger
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted


class Base(ABC, BaseEstimator):
    def __init__(
        self,
        k,
        n_epochs,
        random_state,
        verbose=False,
    ):
        self.k = k
        self.n_epochs = n_epochs
        self.random_state = random_state
        self.verbose = verbose

        if verbose:
            self.logger_console = Logger(
                log_file=f"{self.__class__.__name__}.log",
                overwrite=True,
            ).get_logger()

    def _validate_data(self, X):
        if sparse.issparse(X):
            return X.tocsr()

        if isinstance(X, np.ndarray):
            if X.ndim != 2:
                raise ValueError("Dense X must be a 2D numpy array.")

            X = np.asarray(X, dtype=np.float64)
            mask = np.isfinite(X) & (X != 0)
            rows, cols = np.where(mask)
            values = X[rows, cols]

            return sparse.csr_matrix(
                (values, (rows, cols)),
                shape=X.shape,
                dtype=np.float64,
            )

        raise TypeError("X must be either a scipy sparse matrix or a 2D numpy array.")

    def mse_on_pairs(self, R_obs, pairs, pred):
        pairs = np.asarray(pairs, dtype=np.int64)
        pred = np.asarray(pred, dtype=np.float64).reshape(-1)

        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError("pairs must have shape (n_pairs, 2).")

        if len(pairs) == 0:
            return 0.0

        if len(pred) != len(pairs):
            raise ValueError(
                f"pred must have one value per pair. Got len(pred)={len(pred)} "
                f"and len(pairs)={len(pairs)}."
            )

        y = self._get_values(R_obs, pairs[:, 0], pairs[:, 1])
        return float(np.mean((y - pred) ** 2))

    @abstractmethod
    def predict(self, X):
        pass

    def score(self, X, y=None):
        check_is_fitted(self)

        X = self._validate_data(X)
        pairs = self._pairs_from_sparse(X)
        pred = self.predict(X)
        mse = self.mse_on_pairs(X, pairs, pred)
        return -float(np.sqrt(mse))

    @abstractmethod
    def fit(self, X, y=None):
        pass

    def _get_values(self, X, u, i):
        X = self._validate_data(X)
        u = np.asarray(u, dtype=np.int64)
        i = np.asarray(i, dtype=np.int64)

        return np.asarray(X[u, i]).reshape(-1).astype(np.float64)

    def _pairs_from_sparse(self, X):
        X = self._validate_data(X)

        coo = X.tocoo()
        return np.column_stack((coo.row, coo.col)).astype(np.int64, copy=False)

    def _ratings_from_sparse(self, X):
        X = self._validate_data(X)
        pairs = self._pairs_from_sparse(X)

        if len(pairs) == 0:
            raise ValueError("No observed entries found.")

        ratings = np.empty((len(pairs), 3), dtype=np.float64)
        ratings[:, :2] = pairs
        ratings[:, 2] = (
            np.asarray(X[pairs[:, 0], pairs[:, 1]]).reshape(-1).astype(np.float64)
        )
        return ratings

    def predict_pairs(self, pairs):
        check_is_fitted(self)
        pairs = np.asarray(pairs, dtype=np.int64)
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError("pairs must have shape (n_pairs, 2).")
        return self._predict_pairs_impl(pairs)

    @abstractmethod
    def _predict_pairs_impl(self, pairs):
        pass

    @abstractmethod
    def transform(self):
        pass

    def get_random_state(self):
        return self.random_state
