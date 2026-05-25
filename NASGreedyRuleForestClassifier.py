import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from sklearn.metrics import log_loss


class NASGreedyRuleForestClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=300, max_depth=10, min_samples_split=5):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.model_ = None
        self.classes_ = None
        self.deep_model_ = None     
        self.nas_history_ = []     

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.classes_ = np.unique(y)

        # XGBoost classifier (original logic preserved)
        self.model_ = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            min_child_weight=self.min_samples_split,
            gamma=0.0,
            tree_method="hist",
            eval_metric="logloss",
            random_state=42
        )

        self.model_.fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(np.array(X))

    def predict_proba(self, X):
        return self.model_.predict_proba(np.array(X))

    def build_deep_nas_network(self, input_dim, num_classes):
        """Create a small neural network for monitoring only."""
        model = Sequential()
        model.add(Dense(64, activation='relu', input_dim=input_dim))
        model.add(BatchNormalization())
        model.add(Dropout(0.25))

        model.add(Dense(32, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.25))

        model.add(Dense(num_classes, activation='softmax'))

        model.compile(optimizer="adam", loss="categorical_crossentropy")
        self.deep_model_ = model
        return model

    def train_nas_deep_learner(self, X, y, epochs=3, batch_size=32):
        """Train the DL monitoring module (non-intrusive)."""
        X = np.array(X)
        y = np.array(y)

        num_classes = len(np.unique(y))

        # Convert labels to one-hot
        y_onehot = np.eye(num_classes)[y]

        if self.deep_model_ is None:
            self.build_deep_nas_network(X.shape[1], num_classes)

        history = self.deep_model_.fit(
            X, y_onehot,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )

        return history.history

    def nas_monitor_performance(self, X_val, y_val):
        """Monitor validation log-loss for NAS search."""
        if self.model_ is None:
            raise ValueError("Model not trained. Call fit() first.")

        preds = self.model_.predict_proba(np.array(X_val))
        loss = log_loss(y_val, preds)

        self.nas_history_.append({
            "logloss": loss,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split
        })

        return loss
