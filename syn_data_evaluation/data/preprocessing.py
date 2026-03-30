

import os
from typing import List, Optional
from click import Tuple
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split

from syn_data_evaluation.data.mice_imputation import MICE_Imputer


class DataPreprocessor:

    def __init__(self, target_col: str = "in_hospital_death", dummy_separator: str = "__"):
        self.target_col = target_col
        self.dummy_separator = dummy_separator

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
    
    def save_5_fold_splits(self, data, fold_folder) -> int:
        """Create and save 5 stratified folds from the data."""
        X, y = self.get_raw_xy(data)
        # Split train data in 5-folds for cross-validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
                
                train_fold = data.iloc[train_idx].reset_index(drop=True)
                validation_fold  = data.iloc[test_idx].reset_index(drop=True)
                # print(f"Train fold distribution:\n{train_fold['in_hospital_death'].value_counts()}")
                # Save fold data
                # create fold folder if not exists
                if not os.path.exists(fold_folder):
                    os.makedirs(fold_folder)
                train_fold.to_csv(os.path.join(fold_folder,f'train_fold_{fold_idx}.csv'), index=False)
                validation_fold.to_csv(os.path.join(fold_folder,f'val_fold_{fold_idx}.csv'), index=False)
    

    def encode_categorical(self, train: pd.DataFrame, test: pd.DataFrame) -> Tuple:
        """One-hot encode categorical variables."""
        train_encoded = pd.get_dummies(train, prefix_sep=self.dummy_separator)
        test_encoded = pd.get_dummies(test, prefix_sep=self.dummy_separator)

        extra_in_test = set(test_encoded.columns) - set(train_encoded.columns)
        if extra_in_test:
            print(f"[encode_categorical] Warning: {len(extra_in_test)} unseen test columns were dropped "
                f"(categories not present in train).")

        # Align test columns with train
        test_encoded = test_encoded.reindex(columns=train_encoded.columns, fill_value=0)
        
        return train_encoded, test_encoded
        
    
    def normalize_data(self, train: pd.DataFrame, test: pd.DataFrame) -> Tuple:
        """Normalize data using StandardScaler."""
        
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train)
        test_scaled = scaler.transform(test)
        
        train_df = pd.DataFrame(train_scaled, columns=train.columns, index=train.index)
        test_df = pd.DataFrame(test_scaled, columns=test.columns, index=test.index)
        
        return train_df, test_df
        
    
    def prepare_data(self, train_data: pd.DataFrame, 
                    test_data: pd.DataFrame,
                    impute=False,
                    categorical_cols=None,
                    normalize: bool = True) -> dict:
        """Complete data preparation pipeline."""
        X_train, y_train = self.get_raw_xy(train_data)
        
        X_test, y_test = self.get_raw_xy(test_data)
        
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