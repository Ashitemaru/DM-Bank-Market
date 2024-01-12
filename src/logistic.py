import numpy as np


class BinaryLogisticClassification:
    def __init__(self, max_step=500000, batch_size=256, lr=1e-1):
        self.params = None
        self.max_step = max_step
        self.batch_size = batch_size
        self.lr = lr

    def fit(self, X, y):
        X, y = X.to_numpy(), y.to_numpy()
        np.random.seed(42)
        self.params = np.random.randn(X.shape[1])
        for _ in range(self.max_step):
            batch_idx = np.random.choice(len(X), self.batch_size)
            X_batch = X[batch_idx]
            y_batch = y[batch_idx]
            prob = 1 / (1 + np.exp(-X_batch.dot(self.params)))
            self.params -= (
                self.lr * (1 / self.batch_size) * X_batch.T.dot(prob - y_batch)
            )

    def predict(self, X):
        X = X.to_numpy()
        return (1 / (1 + np.exp(-X.dot(self.params))) >= 0.5).astype(int)


if __name__ == "__main__":
    pass
