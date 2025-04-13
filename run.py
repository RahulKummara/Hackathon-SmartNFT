import sys
sys.path.append('src')
from data_loader import DataLoader
from data_preprocessor import Preprocessor
from feature_engineering import FeatureEngineer
from model_trainer import ModelTrainer

# LOAD DATA
loader = DataLoader("data/nft_data.csv")
data = loader.load_data()

# PREPROCESS DATA
preprocessor = Preprocessor(log_transform = True, target = 'Market_Cap_USD')
data = preprocessor.preprocess(data)

# FEATURE ENGINEERING
feature_engineer = FeatureEngineer(target = 'Market_Cap_USD', log_transform = True)
data = feature_engineer.add_features(data)

# MODEL TRAINER
trainer = ModelTrainer()
model, metrics = trainer.run(data, True)
trainer.save_model()

# print("Data loaded successfully")
# print("Shape: ", data.shape)

# print("X_scaled_shape", X_scaled.shape)
# print("y shape:", y.shape)
# print("Feature names:", [f for f in feature_names])
# print("First 5 target values:\n", y.head())

# print("Feature engineering successful!")
# print("Columns:", [col for col in data.columns])
# print("Shape:", data.shape)

print("Model training successful!")
print(f"Metrics: {metrics}")