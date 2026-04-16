import os
import glob
import re
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from syn_data_evaluation.data import postprocessing
from syn_data_evaluation.evaluation.fidelity_evaluator import run_simple_evaluation
from syn_data_evaluation.experiments.run_fidelity import get_submetadata
from syn_data_evaluation.experiments.run_utility import ExperimentRunner
from tests.run_dp_cgans import DPCGANConfig, run_dp_cgans
RESOURCE_FOLDER = "./dp_cgans/resources/"

def make_group_stratified_train_test(df, label_col, group_col, test_size=0.25, random_state=42):
    """
    Uses StratifiedGroupKFold to create a single 75/25 split without patient leakage.
    With n_splits=4, each fold is ~25%.
    """
    assert abs(test_size - 0.25) < 1e-9, "This helper currently assumes test_size=0.25 (n_splits=4)."

    sgkf = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=random_state)

    X = df.drop(columns=[label_col])
    y = df[label_col]
    groups = df[group_col]

    # Take the first split as test fold
    train_idx, test_idx = next(sgkf.split(X, y, groups=groups))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df  = df.iloc[test_idx].reset_index(drop=True)

    return train_df, test_df


def save_group_stratified_5folds(train_df, label_col, group_col, out_dir, random_state=42):
    os.makedirs(out_dir, exist_ok=True)

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state)

    X = train_df.drop(columns=[label_col])
    y = train_df[label_col]
    groups = train_df[group_col]

    for fold, (tr_idx, va_idx) in enumerate(sgkf.split(X, y, groups=groups), start=1):
        tr = train_df.iloc[tr_idx].reset_index(drop=True)
        va = train_df.iloc[va_idx].reset_index(drop=True)

        tr.to_csv(os.path.join(out_dir, f"train_fold_{fold}.csv"), index=False)
        va.to_csv(os.path.join(out_dir, f"val_fold_{fold}.csv"), index=False)


def assert_no_patient_leakage(train_df, test_df, group_col):
    train_ids = set(train_df[group_col].unique())
    test_ids  = set(test_df[group_col].unique())
    overlap = train_ids.intersection(test_ids)
    assert len(overlap) == 0, f"Patient leakage detected! Overlap size: {len(overlap)}"


def main(real_data=None, train_data=None, test_data=None, label_col="in_hospital_death", group_col="subject_id", save_folds=True):
    if real_data is not None:
        # 1) Train/Test 75/25, stratified + grouped
        train_data, test_data = make_group_stratified_train_test(
            real_data,
            label_col=label_col,
            group_col=group_col,
            test_size=0.25,
            random_state=42
        )
        assert_no_patient_leakage(train_data, test_data, group_col)

        train_path = os.path.join(RESOURCE_FOLDER, "icu_dka_train_data.csv")
        test_path  = os.path.join(RESOURCE_FOLDER, "icu_dka_test_data.csv")
        train_data.to_csv(train_path, index=False)
        test_data.to_csv(test_path, index=False)
    elif train_data is None and test_data is None:
        raise ValueError("Either real_data or both train_data and test_data must be provided.")

    assert_no_patient_leakage(train_data, test_data, group_col)

    print(f"Train rows: {len(train_data)} | Test rows: {len(test_data)}")
    print(f"Train patients: {train_data[group_col].nunique()} | Test patients: {test_data[group_col].nunique()}")
    print(f"Train label distribution:\n{train_data[label_col].value_counts(dropna=False)}")
    print(f"Test label distribution:\n{test_data[label_col].value_counts(dropna=False)}")

    folds_dir = os.path.join(RESOURCE_FOLDER, "folds")
    # 2) 5-fold CV within training, stratified + grouped
    if save_folds:
        save_group_stratified_5folds(
            train_data,
            label_col=label_col,
            group_col=group_col,
            out_dir=folds_dir,
            random_state=42
        )

    # quick sanity checks: no patient overlap between train/val within each fold
    train_files = sorted(glob.glob(os.path.join(folds_dir, "train_fold_*.csv")))
    for file in train_files:
        fold_number = re.search(r"train_fold_(\d+)\.csv", file).group(1)
        tr = pd.read_csv(file)
        va = pd.read_csv(os.path.join(folds_dir, f"val_fold_{fold_number}.csv"))

        tr_ids = set(tr[group_col].unique())
        va_ids = set(va[group_col].unique())
        overlap = tr_ids.intersection(va_ids)

        print(f"Fold {fold_number}: train rows={len(tr)}, val rows={len(va)} | overlap patients={len(overlap)}")
        assert len(overlap) == 0, f"Leakage inside fold {fold_number}!"
    
    # Data is now ready for training and evaluation with no patient leakage across train/test and train/val splits.
    # Clean subject_id column and SOFA
    # Create synthetic data using DP-CGANS
    generated_model_path = 'output/generators'
    os.makedirs(generated_model_path, exist_ok=True)
    transformers_path = 'output/transformer'
    os.makedirs(transformers_path, exist_ok=True)
    evaluation_path = 'output/evaluation'
    os.makedirs(evaluation_path, exist_ok=True)
    # syn_path = 'output/synthetic_data'
    # os.makedirs(syn_path, exist_ok=True)
    config = DPCGANConfig(
        epochs=5,
        batch_size=60, # ~6% of 1086 (64) and ~6% of 1711 (100)
        generator_dim=(128, 128, 128),
        discriminator_dim=(128, 128, 128),
        generator_lr=2e-5,
        discriminator_lr=2e-5,
        discriminator_steps=10,
        private=False,
        xai_type=None,
        xai_weight=0,
        saved_transformer=transformers_path+'/fitted_transformer.pkl'
    )
    metadata = get_submetadata("./notebooks/icu_dka_metadata.json", train_data.columns)
    exp_name = "baseline"
    for file in train_files:
        fold_number = re.search(r"train_fold_(\d+)\.csv", file).group(1)
        tabular_data = pd.read_csv(file).drop(columns=["subject_id", "sofa"])
        validation = pd.read_csv(os.path.join(folds_dir, f"val_fold_{fold_number}.csv")).drop(columns=["subject_id", "sofa"])
        print(f"----- Training fold {fold_number} with {len(tabular_data)} rows. -----")
        model_name, syn_data =run_dp_cgans(tabular_data, f"syn_data_fold_{fold_number}", generated_model_path, config, save_output=True)
        # Evaluate Fidelity
        print(f"Evaluating fidelity for fold {fold_number}... - Model: {model_name}")
        syn_data_fi = postprocessing.postprocess_for_fidelity(syn_data)
        results = run_simple_evaluation(
            tabular_data, syn_data_fi, metadata,
            experiment_name=f"{exp_name}_fold_{fold_number}",
            results_csv=os.path.join(evaluation_path, 'fidelity_results.csv')
        )
        # Evaluate Utility - for now, run only experiment 1
        syn_data_util = postprocessing.postprocess_for_utility(syn_data)
        ExperimentRunner().run_experiment(
            train_data=syn_data_util, 
            test_data=test_data,
            out_dir=f"{evaluation_path}/{exp_name}/fold_{fold_number}",
            exp_name=exp_name,
        )

    # Repeat for the final model training on the entire training set and evaluation on the test set.

if __name__ == "__main__":
    real_data = pd.read_csv(os.path.join(RESOURCE_FOLDER, "icu_dka_dataset_20260415.csv"))
    train_data = pd.read_csv(os.path.join(RESOURCE_FOLDER, "icu_dka_train_data.csv"))
    test_data = pd.read_csv(os.path.join(RESOURCE_FOLDER, "icu_dka_test_data.csv"))

    # main(real_data=real_data, save_folds=True)
    main(real_data=None, train_data=train_data, test_data=test_data, save_folds=False)
