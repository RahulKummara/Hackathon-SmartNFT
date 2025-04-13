import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import logging

class ModelTrainer:
    def __init__(self, model_params = None, target = "Market_Cap_USD", test_size = 0.2, random_state = 42):
        self.target = target
        self.test_size = test_size
        self.random_state = random_state
        self.model_params = model_params or {
            'n_estimators': 200,
            'max_depth': 20,
            'min_samples_split': 2,
            'random_state': random_state
        }
        self.model = RandomForestRegressor(**self.model_params)
        self.scaler = StandardScaler()
        self.logger = logging.getLogger(__name__)
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.y_train = None
        self.y_test = None
    
    def split_data(self, data):
        try:
            if self.target not in data.columns:
                raise KeyError(f"Target column not found in data")

            X = data.drop(self.target, axis=1)
            y = data[self.target]

            # Validate numerical features
            non_numeric_cols = X.select_dtypes(exclude=['float64', 'int64']).columns
            if non_numeric_cols.any():
                raise ValueError(f"Non-numeric columns found: {non_numeric_cols.tolist()}")

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size = self.test_size, random_state = self.random_state
            )
            
            # Scale features
            self.X_train_scaled = self.scaler.fit_transform(X_train)
            self.X_test_scaled = self.scaler.transform(X_test)
            self.y_train = y_train
            self.y_test = y_test
            
            self.logger.info(f"Data split: {len(X_train)} train, {len(X_test)} test samples")
            return self.X_train_scaled, self.X_test_scaled, y_train, y_test
        
        except Exception as e:
            self.logger.error(f"Data splitting error: {e}")
            raise
            
    def train(self, X_train, y_train, tune):
        try:
            if tune:
                param_dist = {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2],
                    'max_features': ['sqrt', 0.5]
                }
                search = RandomizedSearchCV(
                    self.model,
                    param_distributions = param_dist,
                    n_iter = 10,
                    cv = 5,
                    scoring = 'r2',
                    n_jobs = -1,
                    random_state = self.random_state
                )
                search.fit(X_train, y_train)
                self.model = search.best_estimator_
                self.logger.info(f"Best parameters: {search.best_params_}")
            else:
                self.model.fit(X_train, y_train)
            
        except Exception as e:
            self.logger.error(f"Training Error: {e}")
            raise
    
    def evaluate(self, X_test, y_test):
        try:
            y_pred = self.model.predict(X_test)
            rmse = round(np.sqrt(mean_squared_error(y_test, y_pred)), 3)
            mae = round(mean_absolute_error(y_test, y_pred), 3)
            r2 = round(r2_score(y_test, y_pred), 3)

            metrics = {
                'Root Mean Squared Error': rmse,
                'Mean Absolute Error': mae,
                'R²': r2
            }
            
            self.logger.info(f"Evaluation metrics: {metrics}")
            return metrics
        
        except Exception as e:
            self.logger.error(f"Evaluation error: {e}")
            raise

    def save_model(self, path = 'models/nft_pricing_model.pkl'):
        try:
            joblib.dump({'Model': self.model, 'Scaler': self.scaler}, path)
            self.logger.info(f"Model and scaler saved to {path}")
        except Exception as e:
            self.logger.error(f"Model saving error: {e}")
            raise
    
    def run(self, data, tune):
        try:
            # Split the data
            X_train_scaled, X_test_scaled, y_train, y_test = self.split_data(data)


            # Train model
            model = self.train(X_train_scaled, y_train, tune = tune)
            
            # Evaluate
            metrics = self.evaluate(X_test_scaled, y_test)

            return model, metrics
    
        except Exception as e:
            self.logger.error(f"Training Pipeline Error: {e}")
            raise