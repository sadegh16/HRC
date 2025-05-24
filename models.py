import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class MinecraftNet(nn.Module):
    """docstring for Net"""

    def __init__(self, goal_dim, action_dim, H, hidden_size=256):
        super(MinecraftNet, self).__init__()
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))
        self.goal_dim = goal_dim
        self.action_dim = action_dim
        self.H = H
        self.hidden_size = hidden_size

        self.critic = nn.Sequential(
            # nn.Linear(52 + goal_dim + 1, hidden_size),
            nn.Linear(96 + goal_dim + 1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
            nn.Sigmoid()
        )

    def forward(self, x, goal):
        B = x.shape[0]
        goal_one_hot = nn.functional.one_hot(goal.long(), num_classes=self.goal_dim + 1).view(B,
                                                                                              self.goal_dim + 1).float()
        x = torch.cat([x, goal_one_hot], dim=1)
        return -self.critic(x) * self.H


class EVModel(nn.Module):
    def __init__(self, num_vars, n_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(1, num_vars-1, n_layers, batch_first=True)

    def forward(self, action, ev):
        out, hidden = self.lstm(action, (ev,ev))
        out = out[:, -1:, :]
        return out


class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, X):
        return self.relu(self.linear(X))


class MultiTaskLassoL1:
    def __init__(self, alpha=0.1, warm_start=False, max_iter=1000, tol=1e-4):
        """
        Custom MultiTaskLasso with L1 regularization and warm start.

        Parameters:
        - alpha: Regularization strength (L1 penalty).
        - warm_start: If True, reuse the solution of the previous fit to initialize the next.
        - max_iter: Maximum number of iterations for Lasso solver.
        - tol: Tolerance for optimization convergence.
        """
        print("Hi from Lasso!")
        self.alpha = alpha
        self.warm_start = warm_start
        self.max_iter = max_iter
        self.tol = tol
        self.models = None  # This will store the individual Lasso models for each target
        self.coef_ = None  # To store the coefficients for all tasks

    def fit(self, X, Y):
        """
        Fit the model to the input features X and multi-output target Y.

        Parameters:
        - X: Input features (num_samples, num_features)
        - Y: Multi-output targets (num_samples, num_targets)
        """
        num_targets = Y.shape[1]  # Number of target variables
        num_features = X.shape[1]  # Number of input features

        # Initialize the list of Lasso models if not already initialized or if warm_start is False
        if not self.warm_start or self.models is None:
            self.models = [Lasso(alpha=self.alpha, max_iter=self.max_iter, tol=self.tol, warm_start=self.warm_start)
                           for _ in range(num_targets)]

        # Fit each model on its corresponding target and store the coefficients
        self.coef_ = np.zeros((num_targets, num_features))  # Initialize coefficient matrix
        for i in range(num_targets):
            self.models[i].fit(X, Y[:, i])  # Fit Lasso for each target
            self.coef_[i, :] = self.models[i].coef_  # Store the coefficients for each target

        return self

    def predict(self, X):
        """
        Predict the target values for the input features X using the learned models.

        Parameters:
        - X: Input features (num_samples, num_features)

        Returns:
        - Predictions for each target (num_samples, num_targets)
        """
        if self.models is None:
            raise ValueError("The model has not been fitted yet!")

        # Predict using each model for its corresponding target
        predictions = np.column_stack([model.predict(X) for model in self.models])
        return predictions


class MultiTaskLogisticL1:
    def __init__(self, alpha=0.1, warm_start=False, max_iter=1000, tol=1e-4):
        """
        Multi-task logistic regression with L1 regularization and warm start.

        Parameters:
        - alpha: Regularization strength (L1 penalty).
        - warm_start: If True, reuse the solution of the previous fit to initialize the next.
        - max_iter: Maximum number of iterations for the solver.
        - tol: Tolerance for stopping criteria.
        """
        self.alpha = alpha
        self.warm_start = warm_start
        self.max_iter = max_iter
        self.tol = tol
        self.models = None  # This will store the individual Logistic Regression models for each target
        self.coef_ = None  # To store the coefficients for all tasks

    def fit(self, X, Y):
        """
        Fit the model to the input features X and multi-output binary target Y.

        Parameters:
        - X: Input features (num_samples, num_features)
        - Y: Multi-output binary targets (num_samples, num_targets)
        """
        num_targets = Y.shape[1]  # Number of target variables
        num_features = X.shape[1]  # Number of input features

        # Initialize the list of Logistic Regression models if not already initialized or if warm_start is False
        if not self.warm_start or self.models is None:
            self.models = [LogisticRegression(penalty='l1', C=1.0 / self.alpha, solver='saga',class_weight='balanced', n_jobs=-1,
                                              max_iter=self.max_iter, tol=self.tol, warm_start=self.warm_start)
                           for _ in range(num_targets)]



        # Fit each model on its corresponding binary target and store the coefficients
        self.coef_ = np.zeros((num_targets, num_features))  # Initialize coefficient matrix
        for i in range(num_targets):
            unique_classes = np.unique(Y[:, i])

            # Skip fitting if the target contains only one unique class (all 0s or all 1s)
            if len(unique_classes) == 1:
                print(f"Skipping target {i}, only one class present: {unique_classes[0]}")
                continue
            self.models[i].fit(X, Y[:, i])  # Fit Logistic Regression for each target
            self.coef_[i, :] = self.models[i].coef_[0]  # Store the coefficients for each target

        return self

    def predict(self, X):
        """
        Predict the target values for the input features X using the learned models.

        Parameters:
        - X: Input features (num_samples, num_features)

        Returns:
        - Binary predictions for each target (num_samples, num_targets)
        """
        if self.models is None:
            raise ValueError("The model has not been fitted yet!")

        # Predict using each model for its corresponding target
        predictions = np.column_stack([model.predict(X) for model in self.models])
        return predictions

    def predict_proba(self, X):
        """
        Predict the probability for the binary target values using the learned models.

        Parameters:
        - X: Input features (num_samples, num_features)

        Returns:
        - Predicted probabilities for each target (num_samples, num_targets)
        """
        if self.models is None:
            raise ValueError("The model has not been fitted yet!")

        # Predict probability for each target
        prob_predictions = np.column_stack([model.predict_proba(X)[:, 1] for model in self.models])
        return prob_predictions
