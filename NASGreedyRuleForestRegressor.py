import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

class NASGreedyRuleForestRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, n_estimators=300, max_depth=10, min_samples_split=5):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.model_ = None
        self.nas_history_ = []
        self.deep_model_ = None  

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        self.model_ = XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            reg_alpha=0.0,
            min_child_weight=self.min_samples_split,
            gamma=0.0,
            tree_method="hist",
            eval_metric="rmse",
            random_state=42
        )

        self.model_.fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(np.array(X))

    def predict_proba(self, X):
        raise AttributeError("Regressor does not support predict_proba.")

    def build_deep_nas_network(self, input_dim):
        model = Sequential()
        model.add(Dense(64, activation='relu', input_dim=input_dim))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))

        model.add(Dense(32, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))

        model.add(Dense(16, activation='relu'))
        model.add(Dense(1))  # regression output

        model.compile(optimizer="adam", loss="mse")
        self.deep_model_ = model
        return model

    def train_nas_deep_learner(self, X, y, epochs=3, batch_size=32):
        """Train the optional DL observer — does NOT affect main model."""
        if self.deep_model_ is None:
            self.build_deep_nas_network(input_dim=X.shape[1])

        X = np.array(X)
        y = np.array(y)

        history = self.deep_model_.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )

        return history.history
    def nas_monitor_performance(self, X_val, y_val):
        if self.model_ is None:
            raise ValueError("Model is not trained. Call fit() first.")

        preds = self.model_.predict(np.array(X_val))
        rmse = np.sqrt(mean_squared_error(y_val, preds))

        self.nas_history_.append({
            "rmse": rmse,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split
        })

        return rmse
