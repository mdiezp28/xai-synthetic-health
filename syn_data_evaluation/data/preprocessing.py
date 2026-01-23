

from typing import List, Optional
from click import Tuple
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split

from syn_data_evaluation.data.mice_imputation import MICE_Imputer


class DataPreprocessor:

    def __init__(self, target_col: str = "in_hospital_death", dummy_separator: str = "__"):
        self.target_col = target_col
        self.dummy_separator = dummy_separator
        self.scaler = None

    # Clean the data (split the data from the outcome)
    def get_raw_xy(self, dataset):
        # Prepare data
        x = dataset.drop(self.target_col, axis=1)  # Exclude outcome
        y = dataset[self.target_col]

        # Remove unnamed and subject_id columns
        x = x.loc[:, ~x.columns.str.contains(
            r'^Unnamed|subject_id', 
            case=False, regex=True
        )]
        return  x, y
    
    def split_data(self, X: pd.DataFrame, y: pd.Series, 
                   test_size: float = 0.3, 
                   random_state: int = 42):
        """Split data into train and test sets with stratification."""
        return train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=y
        )
    

    def encode_categorical(self, train: pd.DataFrame, test: pd.DataFrame) -> Tuple:
        """One-hot encode categorical variables."""
        train_encoded = pd.get_dummies(train, prefix_sep=self.dummy_separator)
        test_encoded = pd.get_dummies(test, prefix_sep=self.dummy_separator)

        # Align test columns with train
        test_encoded = test_encoded.reindex(columns=train_encoded.columns, fill_value=0)
        
        return train_encoded, test_encoded
        
    
    def normalize_data(self, train: pd.DataFrame, test: pd.DataFrame) -> Tuple:
        """Normalize data using StandardScaler."""
        if self.scaler is None:
            self.scaler = StandardScaler()
            train_scaled = self.scaler.fit_transform(train)
        else:
            train_scaled = self.scaler.transform(train)
        
        train_df = pd.DataFrame(train_scaled, columns=train.columns, index=train.index)
        

        test_scaled = self.scaler.transform(test)
        test_df = pd.DataFrame(test_scaled, columns=test.columns, index=test.index)
        
        return train_df, test_df
        
    
    def prepare_data(self, train_data: pd.DataFrame, 
                    test_data: Optional[pd.DataFrame] = None,
                    impute=False,
                    categorical_cols=None,
                    normalize: bool = True,
                    test_size: float = 0.3,
                    random_state: int = 42) -> dict:
        """Complete data preparation pipeline."""
        X, y = self.get_raw_xy(train_data)
        
        if test_data is None:
            X_train, X_test, y_train, y_test = self.split_data(
                X, y, test_size, random_state
            )
        else:
            X_test, y_test = self.get_raw_xy(test_data)
            X_train, y_train = X, y
        
        # Reset indices
        X_train = X_train.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)
        
        # Imputation
        if impute:
            print("Imputing missing data...")
            imputer = MICE_Imputer(X_train, categorical_cols)
            X_train = imputer.transform_train()
            X_test = imputer.transform_test(X_test)
            print("Imputation complete.")
        
        # Encoding
        X_train, X_test = self.encode_categorical(X_train, X_test)
        
        # Normalization
        if normalize:
            print("Normalizing data...")
            X_train_model, X_test_model = self.normalize_data(X_train, X_test)
        else:
            X_train_model = X_train.copy()
            X_test_model = X_test.copy()
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'X_train_model': X_train_model,
            'X_test_model': X_test_model,
            'y_train': y_train,
            'y_test': y_test
        }