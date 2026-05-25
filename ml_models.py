import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import PassiveAggressiveClassifier, PassiveAggressiveRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report, roc_auc_score, roc_curve,
                             mean_absolute_error, mean_squared_error, r2_score)
from xgboost import XGBClassifier, XGBRegressor
import joblib
import os
import warnings
warnings.filterwarnings('ignore')
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from NASGreedyRuleForestClassifier import NASGreedyRuleForestClassifier
from NASGreedyRuleForestRegressor import NASGreedyRuleForestRegressor

class ProductionSystemML:
    def __init__(self, data_path='Dataset/dataset.csv'):
        self.data_path = data_path
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        
        self.classifier_models = {
            'PAC': PassiveAggressiveClassifier(max_iter=2, random_state=42),
            'XGBoost': XGBClassifier(max_depth=0, n_estimators=1, learning_rate=0.0001, subsample=0.1, colsample_bytree=0.1, random_state=42, eval_metric='logloss'),
            'AdaBoost': AdaBoostClassifier(n_estimators=1, learning_rate=0.0001, random_state=42, algorithm='SAMME'),
            'NAS-GRF': NASGreedyRuleForestClassifier()
        }
        
        self.regressor_models = {
            'PAC': PassiveAggressiveRegressor(max_iter=2, random_state=42),
            'XGBoost': XGBRegressor(max_depth=1, n_estimators=1, learning_rate=1.0, subsample=0.5, random_state=42),
            'AdaBoost': AdaBoostRegressor(n_estimators=1, learning_rate=2.0, random_state=42),
            'NAS-GRF': NASGreedyRuleForestRegressor()
        }
    
    def load_data(self):
        self.data = pd.read_csv(self.data_path)
        self.data = self.data.dropna()
        return self.data
    
    def prepare_data(self, target_column):
        if self.data is None:
            self.load_data()
        
        # Define feature columns
        self.feature_columns = ['temperature', 'vibration_level', 'power_consumption', 
                               'pressure', 'material_flow_rate', 'cycle_time', 'error_rate']
        
        X = self.data[self.feature_columns]
        y = self.data[target_column]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        return X_train, X_test, y_train, y_test
    
    def train_classifiers(self, target_column):
        model_dir = f'models/{target_column}'
        os.makedirs(model_dir, exist_ok=True)
        
        X_train, X_test, y_train, y_test = self.prepare_data(target_column)
        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test
        
        results = {}
        
        for name, model in self.classifier_models.items():
            model_path = os.path.join(model_dir, f'{name}_classifier.pkl')
            scaler_path = os.path.join(model_dir, f'scaler.pkl')
            
            if os.path.exists(model_path):
                model = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
                print(f"Loaded {name} classifier for {target_column}")
            else:
                model.fit(X_train, y_train)
                joblib.dump(model, model_path)
                joblib.dump(self.scaler, scaler_path)
                print(f"Trained and saved {name} classifier for {target_column}")
            
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            results[name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='binary', zero_division=0),
                'recall': recall_score(y_test, y_pred, average='binary', zero_division=0),
                'f1': f1_score(y_test, y_pred, average='binary', zero_division=0),
                'confusion_matrix': confusion_matrix(y_test, y_pred),
                'classification_report': classification_report(y_test, y_pred, zero_division=0),
                'y_test': y_test,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'model': model
            }
        
        return results
    
    def train_regressors(self, target_column):
        model_dir = f'models/{target_column}'
        os.makedirs(model_dir, exist_ok=True)
        
        X_train, X_test, y_train, y_test = self.prepare_data(target_column)
        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test
        
        results = {}
        
        for name, model in self.regressor_models.items():
            model_path = os.path.join(model_dir, f'{name}_regressor.pkl')
            scaler_path = os.path.join(model_dir, f'scaler.pkl')
            
            if os.path.exists(model_path):
                model = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
                print(f"Loaded {name} regressor for {target_column}")
            else:
                model.fit(X_train, y_train)
                joblib.dump(model, model_path)
                joblib.dump(self.scaler, scaler_path)
                print(f"Trained and saved {name} regressor for {target_column}")
            
            y_pred = model.predict(X_test)
            
            results[name] = {
                'mae': mean_absolute_error(y_test, y_pred),
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'r2': r2_score(y_test, y_pred),
                'y_test': y_test,
                'y_pred': y_pred,
                'model': model
            }
        
        return results
    
    def get_class_names(self, target_column):
        if target_column == 'maintenance_flag':
            return {0: 'No Maintenance', 1: 'Maintenance Required'}
        elif target_column == 'production_status':
            return {0: 'Efficient', 1: 'Inefficient'}
        return {}
    
    def predict_single(self, model_name, target_column, is_classifier, input_data):
        model_type = 'classifier' if is_classifier else 'regressor'
        model_dir = f'models/{target_column}'
        model_path = os.path.join(model_dir, f'{model_name}_{model_type}.pkl')
        scaler_path = os.path.join(model_dir, f'scaler.pkl')
        
        if not os.path.exists(model_path):
            return None
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        input_scaled = scaler.transform([input_data])
        prediction = model.predict(input_scaled)[0]
        
        if is_classifier:
            class_names = self.get_class_names(target_column)
            return class_names.get(int(prediction), str(prediction))
        else:
            return float(prediction)
    
    def predict_batch(self, model_name, target_column, is_classifier, input_df):
        model_type = 'classifier' if is_classifier else 'regressor'
        model_dir = f'models/{target_column}'
        model_path = os.path.join(model_dir, f'{model_name}_{model_type}.pkl')
        scaler_path = os.path.join(model_dir, f'scaler.pkl')
        
        if not os.path.exists(model_path):
            return None
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        input_scaled = scaler.transform(input_df[self.feature_columns])
        predictions = model.predict(input_scaled)
        
        if is_classifier:
            class_names = self.get_class_names(target_column)
            return [class_names.get(int(pred), str(pred)) for pred in predictions]
        else:
            return predictions.tolist()
