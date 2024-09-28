import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model._base import LinearClassifierMixin


class LogisticRegression(BaseEstimator, LinearClassifierMixin):
    """
    Logistic Regression classifier.

    This classifier implements a logistic regression model using gradient descent.

    Attributes:
        theta (np.ndarray): Weights after fitting the model.
        lr (float): Learning rate.
        num_iter (int): Number of iterations.
        fit_intercept (bool): Whether the model is fit with an intercept term.
        verbose (bool): Whether to print training progress.

    Args:
        lr (float): Learning rate.
        num_iter (int): Number of iterations.
        fit_intercept (bool): Whether the model is fit with an intercept term.
        tol (float): Tolerance for stopping criteria.
        verbose (bool): Whether to print training progress.
    """

    def __init__(
        self,
        *,
        lr: float = 0.01,
        num_iter: int = 100000,
        fit_intercept: bool = True,
        tol: float = 1e-4,
        verbose: bool = False,
    ) -> None:
        self.theta = np.array([])
        self.lr = lr
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept
        self.tol = tol
        self.verbose = verbose

    @staticmethod
    def _add_intercept(X: np.ndarray) -> np.ndarray:
        """add intercept term to X"""
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def _loss(h: np.ndarray, y: np.ndarray) -> float:
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()  # mean cross-entropy loss

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the model according to the given training data.

        Args:
            X (np.ndarray): Training feature matrix.
            y (np.ndarray): Training labels.
        """
        if self.fit_intercept:
            X = self._add_intercept(X)

        # weights initialization
        self.theta = np.random.rand(X.shape[1])

        prev_loss = np.inf
        for i in range(self.num_iter):
            z = np.dot(X, self.theta)  # linear combination
            h = self._sigmoid(z)  # apply sigmoid
            gradient = np.dot(X.T, (h - y)) / y.size  # gradient of the loss function
            self.theta -= self.lr * gradient  # update weights

            z = np.dot(X, self.theta)  # linear combination
            h = self._sigmoid(z)  # apply sigmoid

            # early stopping
            loss = self._loss(h, y)
            if abs(prev_loss - loss) < self.tol:
                print(f'Converged at iteration {i}')
                break
            prev_loss = loss

            if self.verbose and i % 100 == 0:
                print(f'iteration: {i}, loss: {loss}')  # print loss

    def predict_prob(self, X: np.ndarray) -> np.ndarray:
        if self.fit_intercept:
            X = self._add_intercept(X)
        return self._sigmoid(np.dot(X, self.theta))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_prob(X).round()

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if self.fit_intercept:
            X = self._add_intercept(X)

        return np.dot(X, self.theta)


if __name__ == '__main__':
    import numpy as np
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score

    data = load_iris()
    X_ = data.data
    y_ = (data.target == 2).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X_, y_,
        test_size=0.2,  # 20% for testing
        random_state=42  # Random seed
    )

    scaler = StandardScaler()  # 标准化
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression(
        lr=0.1,
        num_iter=100000,
        fit_intercept=True,
        verbose=True
    )

    # 5. Train the model
    model.fit(X_train, y_train)

    # 6. Make predictions on the test set
    y_pred = model.predict(X_test)

    # 7. Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy on the test set: {accuracy}')

    # 8. Print decision function values for some test samples
    decision_values = model.decision_function(X_test)
    print('Decision function values for some test samples:', decision_values[:5])
