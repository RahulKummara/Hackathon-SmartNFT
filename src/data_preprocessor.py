import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import logging

class Preprocessor:
    def __init__(self, log_transform = True, scaler_type = 'standard', target = 'Market_Cap_USD'):
        self.log_transform = log_transform
        self.scaler_type = scaler_type
        self.target = target
        self.imputer_num = SimpleImputer(strategy = 'median')
        self.imputer_cat = SimpleImputer(strategy = 'constant', fill_value = 'Unknown')
        self.scaler = StandardScaler() if scaler_type == "standard" else None
        self.logger = logging.getLogger(__name__)
    
    def preprocess(self, data):
        try:
            # Copy data
            df = data.copy()
            self.logger.info(f"Starting preprocessing for {len(df)} rows")

            # Step 1: Impute numerical columns
            numerical_cols = df.select_dtypes(include = ['float64', 'int64']).columns
            if numerical_cols.empty:
                raise ValueError("No numerical columns found")
            df[numerical_cols] = self.imputer_num.fit_transform(df[numerical_cols])
            self.logger.info(f"Imputed numerical columns: {numerical_cols.tolist()}")

            # Step 2: Impute categorical columns
            categorical_cols = ['Name', 'Category', 'Website', 'Logo']
            existing_cat_cols = [col for col in categorical_cols if col in df.columns]
            if existing_cat_cols:
                df[existing_cat_cols] = self.imputer_cat.fit_transform(df[existing_cat_cols])
                self.logger.info(f"Imputed categorical columns: {existing_cat_cols}")
            else:
                self.logger.warning("No categorical columns found for imputation")

            # Step 3: Verify missing values
            missing = df.isnull().sum()
            if missing.any():
                self.logger.warning(f"Remaining missing values:\n{missing[missing > 0]}")
            else:
                self.logger.info("No missing values after imputation")

            # Step 4: Encode Category (handle comma-separated values)
            categories = set()
            df['Category'].apply(lambda x: categories.update(x.split(',')) if x != 'Unknown' else None)
            categories = [cat.strip() for cat in categories if cat.strip()]
            if not categories:
                self.logger.warning("No categories found; using 'Unknown' only")
                categories = ['Unknown']
            
            for cat in categories:
                df[f'Category_{cat}'] = df['Category'].apply(lambda x: 1 if cat in x else 0)
            self.logger.info(f"Encoded categories: {categories}")

            # Step 5: Drop non-predictive columns
            drop_cols = ['Index', 'Name', 'Website', 'Logo', 'Category']
            existing_drop_cols = [col for col in drop_cols if col in df.columns]
            if existing_drop_cols:
                df.drop(existing_drop_cols, axis=1, inplace=True)
                self.logger.info(f"Dropped columns: {existing_drop_cols}")
            else:
                self.logger.warning("No columns to drop from drop_cols")

            # Step 6: Log-transform numerical columns
            if self.log_transform:
                skewed_cols = ['Volume_USD', 'Floor_Price_USD', 'Sales', 'Owners',
                              'Assets', 'Market_Cap_USD']
                existing_skewed_cols = [col for col in skewed_cols if col in df.columns]
                for col in existing_skewed_cols:
                    df[col] = np.log1p(df[col])
                self.logger.info(f"Log-transformed columns: {existing_skewed_cols}")
            
            # Step 7: Split features and target
            if self.target not in df.columns:
                raise KeyError(f"Target column {self.target} not found")
            X = df.drop(self.target, axis=1)
            y = df[self.target]
            self.logger.info(f"Features: {X.columns.tolist()}")

            # Step 8: Verify numerical features
            non_numeric_cols = X.select_dtypes(exclude=['float64', 'int64']).columns
            if non_numeric_cols.any():
                raise ValueError(f"Non-numeric columns in X: {non_numeric_cols.tolist()}")
            
            # Step 9: Scale features
            X_scaled = X.values
            if self.scaler:
                X_scaled = self.scaler.fit_transform(X)
                self.logger.info("Features scaled")
            else:
                self.logger.info("No scaling applied")

            self.logger.info("Preprocessing completed")
            return df

        except Exception as e:
            self.logger.error(f"Preprocessing error: {e}")
            raise