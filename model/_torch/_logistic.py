from typing import Union

import torch


class LogisticRegression(torch.nn.Module):
    """
    Logistic regression model implemented in PyTorch.

    This model implements a logistic regression model using gradient descent.

    Attributes:
        lr (float): Learning rate.
        num_iter (int): Number of iterations.
        fit_intercept (bool): Whether the model is fit with an intercept term.
        verbose (bool): Whether to print training progress.
        device (torch.device): Device to run the model on.
        linear (torch.nn.Linear): Linear layer for logistic regression.

    Args:
        input_dim (int): Number of input features.
        lr (float): Learning rate.
        num_iter (int): Number of iterations.
        fit_intercept (bool): Whether the model is fit
    """

    def __init__(
        self,
        input_dim: int,
        *,
        lr: float = 0.01,
        num_iter: int = 100000,
        C: float = 1.0,
        fit_intercept: bool = True,
        verbose: bool = False,
        device: Union[str, torch.device] = 'cpu',
    ):
        super(LogisticRegression, self).__init__()
        self.lr = lr
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        self.C = C
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.linear = torch.nn.Linear(input_dim, 1, bias=fit_intercept).to(device)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(self.device)
        return torch.sigmoid(self.linear(X))

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        """
        Fit the logistic regression model.

        Args:
            X (torch.Tensor): Training feature matrix.
            y (torch.Tensor): Training labels.
        """
        X = X.to(self.device)  # move X to device
        y = y.to(self.device).view(-1, 1)  # reshape y to column vector

        criterion = torch.nn.BCELoss()  # binary cross-entropy loss
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, weight_decay=1 / self.C)  # L2 regularization

        for i in range(self.num_iter):
            optimizer.zero_grad()
            outputs = self.forward(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            if self.verbose and i % 10000 == 0:
                print(f'Iteration {i}, Loss: {loss.item()}')

    def predict_prob(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(self.device)
        self.eval()
        with torch.no_grad():
            outputs = self.forward(X)
        return outputs.cpu()

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(self.device)
        probabilities = self.predict_prob(X)
        return (probabilities >= 0.5).float()

    def score(self, X: torch.Tensor, y: torch.Tensor) -> float:
        y_pred = self.predict(X)
        return (y_pred.squeeze() == y.cpu()).float().mean().item()


if __name__ == '__main__':
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

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    # Initialize and train the model
    model = LogisticRegression(
        input_dim=X_train.shape[1],
        lr=0.1,
        num_iter=100000,
        fit_intercept=True,
        device='cuda',
        verbose=True
    )
    model.fit(X_train_tensor, y_train_tensor)

    # Make predictions on the test set
    y_pred_ = model.predict(X_test_tensor)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred_)
    print(f'Accuracy on the test set: {accuracy}')

    # Print decision function values for some test samples
    decision_values = model.predict_prob(X_test_tensor)
    print('Decision function values for some test samples:', decision_values[:5])
