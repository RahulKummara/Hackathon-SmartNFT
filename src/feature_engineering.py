import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging

class FeatureEngineer:
    def __init__(self, target = "Market_Cap_USD", log_transform = True, scale_features = True):
        self.target = target
        self.log_transform = log_transform
        self.scale_features = scale_features
        self.scaler = StandardScaler() if scale_features else None
        self.logger = logging.getLogger(__name__)

        # Define new columns for processing
        self.new_features = {
            'Volume_per_Sale': lambda df: df['Volume_USD'] / df['Sales'].replace(0, 1),
            'Avg_Price_per_Asset': lambda df: df['Average_Price_USD'] / df['Assets'].replace(0, 1),
            'Market_Cap_to_Volume': lambda df: df['Market_Cap_USD'] / df['Volume_USD'].replace(0, 1)
        }

        self.drop_cols = ['Volume', 'Market_Cap', 'Floor_Price', 'Average_Price']
        self.outlier_cols = ['Volume_USD', 'Market_Cap_USD', 'Floor_Price_USD', 'Sales', 'Owners', 'Assets']
        self.skewed_cols = ['Volume_USD', 'Market_Cap_USD', 'Floor_Price_USD', 'Sales', 'Owners', 'Assets', 'Volume_per_Sale']

    def add_features(self, data):
        try:
            df = data.copy()
            self.logger.info(f"Starting feature engineering for {len(df)} rows")

            # Step 1: Create new features
            for feature_name, func in self.new_features.items():
                if all(col in df.columns for col in func.__code__.co_varnames if col in df.columns):
                    df[feature_name] = func(df)
                    self.logger.info(f"Created feature: {feature_name}")
                else:
                    self.logger.warning(f"Skipping {feature_name}: required columns missing")

            # Step 2: Handle infinities and NaNs
            df.replace([np.inf, -np.inf], np.nan, inplace = True)
            numeric_cols = df.select_dtypes(include = ['float64', 'int64']).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
            self.logger.info("Handled infinities and NaNs")

            # Step 3: Drop redundant columns
            existing_drop_cols = [col for col in self.drop_cols if col in df.columns]
            if existing_drop_cols:
                df.drop(existing_drop_cols, axis = 1, inplace = True)
                self.logger.info(f"Dropped columns: {existing_drop_cols}")
            else:
                self.logger.warning("No columns to drop")
            
            # Step 4: Cap outliers
            for col in [c for c in self.outlier_cols if c in df.columns]:
                lower, upper = df[col].quantile([0.01, 0.99])
                df[col] = df[col].clip(lower, upper)
            self.logger.info(f"Capped outliers in: {[c for c in self.outlier_cols if c in df.columns]}")

            # Step 5: Log-transform skewed columns
            if self.log_transform:
                for col in [c for c in self.skewed_cols if c in df.columns]:
                    df[col] = np.log1p(df[col])
                self.logger.info(f"Log-transformed: {[c for c in self.skewed_cols if c in df.columns]}")

            # Step 6: Scale numerical features (exclude Category_* and target)
            if self.scale_features and self.scaler:
                num_cols = [col for col in df.columns if not col.startswith('Category_') and col != self.target]
                if num_cols:
                    df[num_cols] = self.scaler.fit_transform(df[num_cols])
                    self.logger.info(f"Scaled columns: {num_cols}")
                else:
                    self.logger.warning("No columns to scale")

            # Step 7: Verify numerical features
            non_numeric_cols = df.select_dtypes(exclude = ['float64', 'int64']).columns
            if non_numeric_cols.any():
                raise ValueError(f"Non-numeric columns found: {[x for x in non_numeric_cols]}")

            self.logger.info("Feature engineering completed")
            return df

        except Exception as e:
            self.logger.error(f"Feature engineering error: {e}")
            raise