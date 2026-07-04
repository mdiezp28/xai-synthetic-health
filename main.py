import os
import glob
import re
import time
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from syn_data_evaluation.data import postprocessing
from syn_data_evaluation.evaluation.fidelity_evaluator import run_simple_evaluation
from syn_data_evaluation.experiments.run_fidelity import get_submetadata
from syn_data_evaluation.experiments.run_utility import ExperimentRunner, run_exp_syn_folds as run_experiment_list, run_exp_syn as run_final_experiment_list
from tests.run_dp_cgans import DPCGANConfig, run_dp_cgans, sample_dp_cgans, generate_balanced_samples


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

def generate_positive_samples(syn_path, syn_pattern, generators_path):
    # find the original syn-data files
    original_files = glob.glob(os.path.join(syn_path, syn_pattern))
    # for each fold file → locate its model → generate positives ─
    for syn_file in sorted(original_files):
        syn_data_len = len(pd.read_csv(syn_file))
        basename = os.path.basename(syn_file)
        print(f"\n  Processing: {basename}")

        # extract fold number (optional, used for output naming)
        fold_match = re.search(r"fold_(\d+)", basename)
        fold_num   = fold_match.group(1) if fold_match else "X"

        # extract date string embedded in the filename
        date_str = extract_date_from_filename(basename)
        if date_str is None:
            print(f"  [WARN] Could not extract date from '{basename}'. Skipping.")
            continue

        print(f"  Extracted date : {date_str}")

        # find the matching generator model
        model_path = find_model_for_date(date_str, generators_path)
        if model_path is None:
            print(f"  [WARN] No model found for date {date_str} in {generators_path}. Skipping.")
            continue

        print(f"  Model found    : {model_path}")
        if fold_num == "X":
            output_stem = os.path.join(
            syn_path,
            f"{data_name}_pos",
        )
        else:
            output_stem = os.path.join(
                syn_path,
                f"{data_name}_pos_fold_{fold_num}_{date_str}",
            )
            output_file = os.path.join(
                syn_path,
                f"{data_name}_balance_fold_{fold_num}_{date_str}",
            )
        # generate positive samples using the model and save
        sample_dp_cgans(model_path, nb_rows=syn_data_len, output_file=output_stem, current_time=None, conditions={"in_hospital_death": 1})
        # generate_balanced_samples(model_path, nb_rows=syn_data_len, output_file=output_file, condition_col="in_hospital_death", postprocess=False)

def extract_date_from_filename(filename: str) -> str | None:
    """
    Pull the date/timestamp token out of a syn-data filename.

    Expects patterns like:
        baseline_syn_data_fold_1_2026_01_03_20_13_38.csv
        shap_focus_10f_weight_1_int25_fold_2_2026_02_14_10_00_00
    Returns e.g. '2026_01_03_20_13_38', or None if not found.
    """
    match = re.search(r"(\d{4}(?:_\d{2}){5})", filename)
    return match.group(1) if match else None


def find_model_for_date(date_str: str, generators_path: str) -> str | None:
    """
    Given a date string like '2026_01_03_20_13_38', look for
    <generators_path>/<date_str>_icu_dka.pkl
    """
    model_path = os.path.join(generators_path, f"{date_str}_dpcgans_icu_dka.pkl")
    return model_path if os.path.exists(model_path) else None

def main(real_data=None, train_data=None, test_data=None, label_col="in_hospital_death", group_col="subject_id", save_folds=True, config=None, exp_name="baseline", generated_model_path="output/generators", syn_path="output/synthetic_data", evaluation_path="output/evaluation", skip_fold=[], run_final=False):
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
    train_data = train_data.drop(columns=["subject_id", "sofa"], errors="ignore")
    test_data = test_data.drop(columns=["subject_id", "sofa"], errors="ignore")
    # Create synthetic data using DP-CGANS
    metadata = get_submetadata("./notebooks/icu_dka_metadata.json", train_data.columns)
    if not run_final:
        for file in train_files:
            fold_number = re.search(r"train_fold_(\d+)\.csv", file).group(1)
            if int(fold_number) in skip_fold:
                print(f"Fold {fold_number} skipped.")
                continue
            tabular_data = pd.read_csv(file).drop(columns=["subject_id", "sofa"])
            validation = pd.read_csv(os.path.join(folds_dir, f"val_fold_{fold_number}.csv")).drop(columns=["subject_id", "sofa"])
            print(f"----- Training fold {fold_number} with {len(tabular_data)} rows. -----")
            model_name, syn_data =run_dp_cgans(tabular_data, f"{syn_path}/{exp_name}_syn_data_fold_{fold_number}", generated_model_path, config, save_output=True)
            # Evaluate Fidelity
            print(f"Evaluating fidelity for fold {fold_number}... - Model: {model_name}")
            syn_data_fi = postprocessing.postprocess_for_fidelity(syn_data)
            run_simple_evaluation(
                tabular_data, syn_data_fi, metadata,
                experiment_name=f"{exp_name}_fold_{fold_number}",
                results_csv=os.path.join(evaluation_path, 'fidelity_results.csv')
            )
            # Evaluate Utility - for now, run only experiment 1
            syn_data_util = postprocessing.postprocess_for_utility(syn_data)
            ExperimentRunner().run_experiment(
                train_data=syn_data_util, 
                test_data=validation,
                out_dir=f"{evaluation_path}/{exp_name}/fold_{fold_number}",
                exp_name=f"{exp_name}_fold_{fold_number}",
            )

    # Repeat for the final model training on the entire training set and evaluation on the test set.
    if run_final:
        print(f"----- Run final model with {len(train_data)} rows. -----")
        model_name, syn_data =run_dp_cgans(train_data, f"{syn_path}/{exp_name}_syn_data", generated_model_path, config, save_output=True)
        # Evaluate Fidelity
        print(f"Evaluating fidelity - Model: {model_name}")
        syn_data_fi = postprocessing.postprocess_for_fidelity(syn_data)
        run_simple_evaluation(
            train_data, syn_data_fi, metadata,
            experiment_name=f"{exp_name}_final",
            results_csv=os.path.join(evaluation_path, 'fidelity_results.csv')
        )
        # Evaluate Utility - for now, run only experiment 1
        syn_data_util = postprocessing.postprocess_for_utility(syn_data)
        ExperimentRunner().run_experiment(
            train_data=syn_data_util, 
            test_data=test_data,
            out_dir=f"{evaluation_path}/{exp_name}/final",
            exp_name=f"{exp_name}_final",
        )


RESOURCE_FOLDER = "./dp_cgans/resources/"
GENERATE_SYNTHETIC_DATA = True
RUN_UTILITY_EXPERIMENTS = False

if __name__ == "__main__":
    # real_data = pd.read_csv(os.path.join(RESOURCE_FOLDER, "icu_dka_dataset_20260415.csv"))
    train_data = pd.read_csv(os.path.join(RESOURCE_FOLDER, "icu_dka_train_data.csv"))
    test_data = pd.read_csv(os.path.join(RESOURCE_FOLDER, "icu_dka_test_data.csv"))

    output_dir = "dp_cgans/tests/output"
    generated_model_path = f'{output_dir}/generators'
    os.makedirs(generated_model_path, exist_ok=True)
    transformers_path = f'{output_dir}/transformer'
    os.makedirs(transformers_path, exist_ok=True)
    evaluation_path = f'{output_dir}/evaluation'
    os.makedirs(evaluation_path, exist_ok=True)
    syn_path = f'{output_dir}/synthetic_data'
    os.makedirs(syn_path, exist_ok=True)
    utility_path = f'{output_dir}/utility/'
    os.makedirs(utility_path, exist_ok=True)


    if GENERATE_SYNTHETIC_DATA:
        config = DPCGANConfig(
            epochs=2000,
            batch_size=60, # ~6% of 1086 (64) and ~6% of 1711 (100)
            generator_dim=(256, 256, 256),
            discriminator_dim=(256, 256, 256),
            generator_lr=2e-5,
            discriminator_lr=2e-5,
            discriminator_steps=5,
            private=False,
            focus_update_interval=25,
            focus_k_features=0,
            xai_weight=0,
            saved_transformer=transformers_path+'/fitted_transformer.pkl'
        )    
        # main(real_data=None, train_data=train_data, test_data=test_data, save_folds=False, config=config, exp_name=f"dp_baseline", generated_model_path=generated_model_path, syn_path=syn_path, evaluation_path=evaluation_path, skip_fold=[])
        # config.xai_weight = 1
        # config.focus_k_features = 10
        # main(real_data=None, train_data=train_data, test_data=test_data, save_folds=False, config=config, exp_name=f"dp_shap_focus", generated_model_path=generated_model_path, syn_path=syn_path, evaluation_path=evaluation_path, skip_fold=[])
        # main(real_data=real_data, save_folds=True)
        # features = [20, 10]
        # weight = [1, 2]
        exp_config =[ # (w,k)
            # (2.5, 15),
            # (10, 10),
            # (1, 5),
            # (2, 10),
            # (1, 10),
            # (2,20),
            # (1, 15),
            # (2, 15),
            # (1.5, 10),
            # (0, 10),
            # (1,10),
            # (1,5),
            (1,20)
        ]
        skip_folds = []
        main(real_data=None, train_data=train_data, test_data=test_data, save_folds=False, config=config, exp_name=f"cgans", generated_model_path=generated_model_path, syn_path=syn_path, evaluation_path=evaluation_path, skip_fold=skip_folds)
        for w, k in exp_config:
            print(f"Start experiment: weight ={w} num features={k}")
            config.xai_weight = w
            config.focus_k_features = k            
            main(real_data=None, train_data=train_data, test_data=test_data, save_folds=False, config=config, exp_name=f"shap_focus_{k}f", generated_model_path=generated_model_path, syn_path=syn_path, evaluation_path=evaluation_path, skip_fold=skip_folds)
            

    if RUN_UTILITY_EXPERIMENTS:
        # Run utility experiments list
        experiment_list = [
            # ("dp_cgans", 'dp_cgans_syn_data_fold_*.csv'),
            # ("dp_shap_focus", 'dp_shap_focus_10f_weight_1_syn_data_fold_*.csv'),
            # ("cgans_val", 'cgans_syn_data_fold_*.csv', False),
            # ("cgans_test", 'cgans_syn_data_fold_*.csv', True),
            ("shap_val", 'shap_focus_10f_syn_data_fold_5_*.csv', False),
            # ("shap_test", 'shap_focus_10f_syn_data_fold_*.csv', True),
            # ("shap_focus_val", 'shap_focus_10f_syn_data_fold_*.csv'),
            # ("baseline_testset", 'baseline_syn_data_fold_*.csv'),
            # ("shap_focus_testset", 'shap_focus_10f_weight_1_int25_syn_data_fold_*.csv'),
            # ("shap_focus_0.5_testset", 'shap_focus_10f_weight_0.5_int25_syn_data_fold_*.csv'),
            # ("baseline_balance", 'baseline__balance_fold_*.csv'),
            # ("shap_focus_pos", 'shap_focus_pos_fold_*'),
            # ("final_shap_focus", 'final_shap_focus_syn_data_2026_05_13_14_47_37.csv'),
            # ("shap_focus_balance", 'shap_focus_balance_fold_*'),
            # ("shap_focus_10", 'shap_focus10f_weight_10_int25_*'),
            # ("shap_focus_2", 'shap_focus10f_weight_2_int25_*'),
        ]
        
        final = False
        postprocess = True
        utility_start = time.perf_counter()
        thresholds=[0.0571, 0.1874, 0.0474, 0.0886, 0.5213]

        for data_name, syn_pattern, use_test_set in experiment_list:
            print(f"Running utility experiments for {data_name} with pattern {syn_pattern}... - Test set: {use_test_set}")
            if use_test_set:
                test_file = test_data
            else:
                test_file = None
            experiment_start = time.perf_counter()
            if final:
                real_file = f"icu_dka_train_data.csv"
                run_final_experiment_list(
                    result_path=utility_path,
                    real_path=RESOURCE_FOLDER,
                    syn_path=syn_path, 
                    real_file="icu_dka_train_data.csv", 
                    syn_file=syn_pattern, 
                    data_name=data_name,
                    threshold = 0.1526,
                    postprocess=postprocess
                )
            else: 
                run_experiment_list(
                    result_path=utility_path,
                    real_fold_path=os.path.join(RESOURCE_FOLDER, "folds"),
                    syn_fold_path=syn_path,
                    real_pattern="train_fold_*.csv",
                    syn_pattern=syn_pattern,
                    data_name=data_name,
                    thresholds=thresholds,
                    postprocess=postprocess,
                    test_file=test_file
                )
            # check if positive samples exist
            pos_pattern = f"{data_name}_pos*.csv"
            pos_files = glob.glob(os.path.join(syn_path, pos_pattern))
            if not pos_files:
                # If no pos_*.csv files, generate them by filtering the original syn_data_fold_*.csv files
                print(f"No files matching {pos_pattern} found. Generating positive samples...")
                generate_positive_samples(syn_path, syn_pattern, generated_model_path)
            else:
                print(f"found: {pos_files}")
                continue

            # Run positive experiments
            if final:
                real_file = f"icu_dka_train_data.csv"
                run_final_experiment_list(
                    result_path=utility_path,
                    real_path=RESOURCE_FOLDER,
                    syn_path=syn_path, 
                    real_file="icu_dka_train_data.csv", 
                    syn_file=f"{data_name}_pos.csv", 
                    data_name=f"{data_name}_pos",
                    threshold = 0.1526,
                    postprocess=postprocess
                )
            else: 
                run_experiment_list(
                    result_path=utility_path,
                    real_fold_path=os.path.join(RESOURCE_FOLDER, "folds"),
                    syn_fold_path=syn_path,
                    real_pattern="train_fold_*.csv",
                    syn_pattern=pos_pattern,
                    data_name=data_name+"_pos",
                    thresholds=thresholds,
                    postprocess=postprocess,
                    test_file=test_file
                )
                
            experiment_elapsed = time.perf_counter() - experiment_start
            print(f"Utility experiments for {data_name} finished in {experiment_elapsed / 60:.2f} minutes ({experiment_elapsed:.2f} seconds).")

        utility_elapsed = time.perf_counter() - utility_start
        print(f"All utility experiments finished in {utility_elapsed / 60:.2f} minutes ({utility_elapsed:.2f} seconds).")
