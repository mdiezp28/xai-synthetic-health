import os
import re
from typing import Dict, List
from pathlib import Path
from syn_data_evaluation.evaluation.fidelity_evaluator import run_full_evaluation, run_simple_evaluation
from sdv.metadata import SingleTableMetadata
import pandas as pd

from syn_data_evaluation.data import postprocessing


def get_submetadata(metadata_file: str, columns: List[str]) -> Dict:
    """Extract metadata for a subset of columns."""
    metadata = SingleTableMetadata.load_from_json(metadata_file).to_dict()
    submetadata = {
        'columns': {},
        'primary_key': metadata.get('primary_key', None),
        'METADATA_SPEC_VERSION': metadata['METADATA_SPEC_VERSION']
    }
    for col in columns:
        if col in metadata['columns']:
            submetadata['columns'][col] = metadata['columns'][col]
    return submetadata


def get_csv_data(data_dir, pattern = "*_syn_data_10f.csv"):
    files = sorted(Path(data_dir).glob(pattern))

    print(f"Found {len(files)} files")

    runs = []
    for f in files:
        df = pd.read_csv(f)
        df["source_file"] = f.name 
        runs.append(df)
    

    return runs

def run_experiment():
    dataset_path = "C:/Users/maria/OneDrive - Maastricht University/Maria Diez Perez/datasets/"
    results_path = "C:/Users/maria/iCloudDrive/Documents/Studies/AI/Thesis/full_results/"
    syn_path = os.path.join(results_path, "syn_data")

    # # Fidelity evaluation on full training set
    real_data = pd.read_csv(os.path.join(dataset_path, 'icu_dka_train_data.csv')).drop(columns=["sofa", "subject_id"], errors="ignore")
    # print("Real data shape:", real_data.shape)
    metadata = get_submetadata("c:/Users/Maria/Code/xai-synthetic-health/notebooks/icu_dka_metadata.json", real_data.columns)
    # syn_data_baseline = pd.read_csv(os.path.join(syn_path, '2026_02_03_11_14_40_baseline.csv'))
    
    # syn_data_baseline = postprocessing.postprocess_for_fidelity(syn_data_baseline)
    # results = run_simple_evaluation(
    #     real_data, syn_data_baseline, metadata,
    #     experiment_name="baseline",
    #     results_csv=os.path.join(results_path,'fidelity', 'fidelity_results.csv')
    # )

    
    # syn_data_shap = pd.read_csv(os.path.join(syn_path, '2026_02_03_12_22_58_shap2.csv'))
    # syn_data_shap = postprocessing.postprocess_for_fidelity(syn_data_shap)

    # results = run_full_evaluation(
    #     real_data, syn_data_shap, metadata,
    #     experiment_name="shap",
    #     results_csv=os.path.join(results_path,'fidelity', 'fidelity_results.csv')
    # )
        


    # ===== SIMPLE EVALUATION =====
    # evaluator = FidelityEvaluator(real_data, syn_data_baseline, metadata)
    # Fidelity on folds

    baseline_list = get_csv_data(syn_path, pattern="*_shap_*_fold_*.csv")
    # pattern="*_baseline_e_2500_fold_3.csv",
    # shap_list = get_csv_data(syn_path, pattern="*_shap2_fold_*.csv")

    for i, syn_data in enumerate(baseline_list):
        source_file = syn_data["source_file"].iloc[0]
        syn_data = syn_data.drop(columns=["source_file"])

        fold_id = re.match(r'.*fold_(\d+)\.csv', source_file).group(1)
        real_data = pd.read_csv(os.path.join(dataset_path+"folds/", f"train_fold_{fold_id}.csv")).drop(columns=["sofa", "subject_id"], errors="ignore")

        syn_data = postprocessing.postprocess_for_fidelity(syn_data)
        results = run_simple_evaluation(
            real_data, real_data, metadata,
            experiment_name=source_file.replace(".csv",""),
            results_csv=os.path.join(results_path,'fidelity', 'fidelity_results.csv')
        )
    # for i, syn_data in enumerate(shap_list):
    #     source_file = syn_data["source_file"].iloc[0]
    #     syn_data = syn_data.drop(columns=["source_file"])

    #     fold_id = re.match(r'.*_fold_(\d+)\.csv', source_file).group(1)
    #     real_data = pd.read_csv(os.path.join(dataset_path+"folds/", f"train_fold_{fold_id}.csv")).drop("sofa", axis=1)

    #     syn_data = postprocessing.postprocess_for_fidelity(syn_data)
    #     results = run_simple_evaluation(
    #         real_data, syn_data, metadata,
    #         experiment_name=source_file.replace(".csv",""),
    #         results_csv=os.path.join(results_path,'fidelity', 'fidelity_results.csv')
    #     )

    # # ===== FULL EVALUATION =====
    # # Specify which columns to plot
    # plot_cols = ["age", "lactate_max", "sofa"]
    # plot_pairs = [('lactate_max', 'lactate_min'), ('wbc_max', 'wbc_min')]

    # full_results = evaluator.full_evaluation(
    #     experiment_name="CTGAN_experiment_1",
    #     exp_dir="./experiments",

    
    # # # Or use convenience function
    # # full_results = run_full_evaluation(
    # #     real_df, synthetic_df, metadata,
    # #     experiment_name="TVAE_experiment_1",
    # #     exp_dir="./experiments",
    # #     plot_columns=plot_cols,
    # #     plot_pairs=plot_pairs
    # # )

if __name__ == "__main__":
    run_experiment()