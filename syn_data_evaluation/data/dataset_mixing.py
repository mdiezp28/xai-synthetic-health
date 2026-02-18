import os
import numpy as np
import pandas as pd
from pathlib import Path
from syn_data_evaluation.data.preprocessing import DataPreprocessor


def get_hybrid_data(
    train_data: pd.DataFrame,
    syn_data: pd.DataFrame,
    syn_pct: float,
    target_col: str = "in_hospital_death",
    seed: int = 42,
    constant_size: bool = True,         
    match_real: bool = False,  
    verbose: bool = True,
):
    """
    Create a training set by mixing real training data with sampled synthetic rows.
    """
    hybrid_size = len(train_data)
    syn_size = int(hybrid_size * syn_pct)
    
    if syn_size > len(syn_data):
        syn_size = len(syn_data)
        print(f"Adjusted synthetic data size to available data: {syn_size}")
    if syn_size == 0:
        return train_data.copy()
        
        
    if constant_size:
        real_size = hybrid_size - syn_size
    else:
        real_size = hybrid_size
        hybrid_size = hybrid_size + syn_size
        sampled_real = train_data.copy()
    
    print(f"Total size: {hybrid_size}")
    print(f"Synthetic data size: {syn_size} ({syn_pct*100:.0f}%)")

    # Decide how many pos/neg to sample from synthetic
    n_classes = syn_data[target_col].nunique(dropna=True)
    if match_real and n_classes > 1:
        syn_pos = syn_data[syn_data[target_col] == 1]
        syn_neg = syn_data[syn_data[target_col] == 0]

        # Match the REAL fold prevalence
        real_pos_rate = (train_data[target_col] == 1).mean()
        pos_n = int(round(syn_size * real_pos_rate))
        neg_n = syn_size - pos_n

        pos_part = syn_pos.sample(n=min(pos_n, len(syn_pos)), random_state=seed)
        neg_part = syn_neg.sample(n=min(neg_n, len(syn_neg)), random_state=seed)
        sampled_syn = pd.concat([pos_part, neg_part], ignore_index=True)

        if constant_size:
            # Adjust real training set to match sampled synthetic prevalence
            real_pos = train_data[train_data[target_col] == 1]
            real_neg = train_data[train_data[target_col] == 0]
            real_pos_n = int(round(real_size * real_pos_rate))
            real_neg_n = real_size - real_pos_n
            sampled_real = pd.concat([
                real_pos.sample(n=real_pos_n, random_state=seed),
                real_neg.sample(n=real_neg_n, random_state=seed)
            ], ignore_index=True)

    else:
        sampled_syn = syn_data.sample(n=syn_size, random_state=seed)
        if constant_size:
            sampled_real = train_data.sample(n=real_size, random_state=seed)

    # If we couldn't reach syn_size because of limited syn_pos/syn_neg, -> error
    if len(sampled_syn) < syn_size:
        raise ValueError(
            f"Not enough synthetic samples to match real prevalence. "
            f"Generate a larger synthetic dataset for this fold/method."
        )
    # Combine
    hybrid_data = pd.concat([sampled_real, sampled_syn], ignore_index=True)
    hybrid_data = hybrid_data.sample(frac=1, random_state=seed).reset_index(drop=True)

    if len(hybrid_data) != hybrid_size:
        raise ValueError(
            f"Hybrid data size mismatch: expected {hybrid_size}, got {len(hybrid_data)}"
        )

    if verbose:
        def vc(df): return df[target_col].value_counts(dropna=False).to_dict()
        print(f"[datamixer] seed={seed}")
        print(f"  real initial size: {len(train_data)}")
        print(f"  real counts: {vc(sampled_real)}")
        print(f"  syn sampled counts: {vc(sampled_syn)}")
        print(f"  final train counts: {vc(hybrid_data)}")

    return hybrid_data

def get_csv_data(data_dir, pattern = "*_syn_data_10f.csv"):
    files = sorted(Path(data_dir).glob(pattern))

    print(f"Found {len(files)} files")

    runs = []
    for f in files:
        df = pd.read_csv(f)
        df["source_file"] = f.name 
        runs.append(df)
    

    return runs

def build_dataset_from_folds(fold_path, target_col="in_hospital_death"):
    X_parts = []
    y_parts = []
    test_fold = []
    for i in range(5):
        train_fold = pd.read_csv(os.path.join(fold_path, f"train_fold_{i+1}.csv"))
        validation_fold = pd.read_csv(os.path.join(fold_path, f"validation_fold_{i+1}.csv"))

        preprocessing = DataPreprocessor(target_col=target_col)
        data = preprocessing.prepare_data(
            train_data=train_fold, 
            test_data=validation_fold,
            impute=False,
            categorical_cols=["gender", "insurance", "race"],
            normalize=False
        )
        X_train = data['X_train_model'].values
        X_validation = data['X_test_model'].values
        y_train = data['y_train'].values
        y_validation = data['y_test'].values
        

        X_parts.extend([X_train, X_validation])
        y_parts.extend([y_train, y_validation])

        # -1 => always train
        test_fold.extend([-1] * len(train_fold))
        # i => validation fold id
        test_fold.extend([i] * len(validation_fold))

    X_data = np.vstack(X_parts)
    y_data = np.concatenate(y_parts)
    test_fold = np.array(test_fold, dtype=int)
    return X_data, y_data, test_fold