import os
from typing import Dict, List
from pathlib import Path
from syn_data_evaluation.evaluation.fidelity_evaluator import run_simple_evaluation
from sdv.metadata import SingleTableMetadata
import pandas as pd

from syn_data_evaluation.data import postprocessing


def get_submetadata(metadata: Dict, columns: List[str]) -> Dict:
    """Extract metadata for a subset of columns."""
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
    # lista de archivos
    files = sorted(Path(data_dir).glob(pattern))

    print(f"Found {len(files)} files")

    # leer y concatenar
    runs = []
    for f in files:
        df = pd.read_csv(f)
        df["source_file"] = f.name   # opcional, útil para trazabilidad
        runs.append(df)
    

    return runs

def run_experiment():
    dataset_path = "C:/Users/maria/OneDrive - Maastricht University/Maria Diez Perez/datasets/"
    results_path = "C:/Users/maria/iCloudDrive/Documents/Studies/AI/Thesis/full_results/"
    real_data = pd.read_csv(os.path.join(dataset_path, 'icu_dka_dataset_simplify.csv')).drop("sofa", axis=1).drop("subject_id", axis=1)
    print("Real data shape:", real_data.shape)
    metadata = get_submetadata(SingleTableMetadata.load_from_json("c:/Users/maria/Code/master/xai-synthetic-health/notebooks/icu_dka_metadata.json").to_dict(), real_data.columns)
    # ===== SIMPLE EVALUATION =====
    # evaluator = FidelityEvaluator(real_data, syn_data_baseline, metadata)

    data_list = get_csv_data(results_path, pattern="2026*_syn_lime_*.csv")
    for i, syn_data in enumerate(data_list):
        source_file = syn_data["source_file"].iloc[0]
        syn_data = syn_data.drop(columns=["source_file"])

        syn_data = postprocessing.postprocess_for_fidelity(syn_data)
        results = run_simple_evaluation(
            real_data, syn_data, metadata,
            experiment_name=source_file.replace(".csv",""),
            results_csv=os.path.join(results_path,'fidelity', 'fidelity_results.csv')
        )

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