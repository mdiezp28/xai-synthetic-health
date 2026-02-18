import glob
import os
import re
import pandas as pd
from syn_data_evaluation.data.preprocessing import DataPreprocessor

RESOURCE_FOLDER = "./dp_cgans/resources/"

def main(train_data=None, real_data=None, save_folds=True):

    preprocessor = DataPreprocessor(target_col="in_hospital_death")
    # Split data into train and test
    if train_data is None:
        X, y = preprocessor.get_raw_xy(real_data)
        X_train, X_test, y_train, y_test = preprocessor.split_data(X, y, test_size=0.25, random_state=42)
        train_data = pd.concat([X_train, y_train], axis=1)
        test_data = pd.concat([X_test, y_test], axis=1)
        train_data.to_csv(os.path.join(RESOURCE_FOLDER,'icu_dka_train_data.csv'), index=False)
        test_data.to_csv(os.path.join(RESOURCE_FOLDER,'icu_dka_test_data.csv'), index=False)
        print(f"Test data distribution:\n{test_data['in_hospital_death'].value_counts()}")
    print(f"Train data distribution:\n{train_data['in_hospital_death'].value_counts()}")
    
    
    # Split train data in 5-folds for cross-validation
    if save_folds:
        preprocessor.save_5_fold_splits(train_data,os.path.join(RESOURCE_FOLDER,"folds"))
    
    train_files = sorted(glob.glob(os.path.join(RESOURCE_FOLDER,"folds","train_fold_*.csv")))
    for file in train_files:
        tabular_data = pd.read_csv(file)
        # tabular_data = convert_string_columns_to_object(tabular_data)
        print("Tabular data shape:", tabular_data.shape)
        print("Tabular data:", tabular_data.dtypes)
        match = re.search(r'train_fold_(\d+)\.csv', file)
        if match:
            fold_number = match.group(1)
            print(f"Fold number: {fold_number}")
            
    

if __name__ == "__main__":
    real_data = pd.read_csv(os.path.join(RESOURCE_FOLDER,'icu_dka_dataset_simplify.csv'))
    # train_data = pd.read_csv(os.path.join(RESOURCE_FOLDER,'icu_dka_train_data.csv'))
    # test_data = pd.read_csv(os.path.join(RESOURCE_FOLDER,'icu_dka_test_data.csv'))
    # main(train_data=train_data, real_data=real_data, save_folds=False)
    main(real_data=real_data, save_folds=True)