import pandas as pd
import logging

logging.basicConfig(level = logging.INFO, filename='logs/pipeline.log')

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.logger = logging.getLogger(__name__)

    def load_data(self):
        try:
            dataset = pd.read_csv(self.file_path)
            excepted_cols = ['Index', 'Name', 'Volume', 'Volume_USD', 'Market_Cap',
                            'Market_Cap_USD', 'Sales', 'Floor_Price', 'Floor_Price_USD',
                            'Average_Price', 'Average_Price_USD', 'Owners', 'Assets',
                            'Owner_Asset_Ratio', 'Category', 'Website']
            
            if not all(col in dataset.columns for col in excepted_cols):
                raise ValueError("Missing Expected Columns")
            self.logger.info(f"Loaded data with {len(dataset)} rows")
            return dataset
        
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise