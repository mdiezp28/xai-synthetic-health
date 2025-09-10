import os
from tabnanny import verbose

import numpy as np
import pandas as pd
# from dp_cgans import DP_CGAN, __version__
from dp_cgans.dp_cgan_init import DP_CGAN


def run_dp_cgans(tabular_data, output_file):
    print(f'Testing DP_CGAN')

    model = DP_CGAN(
        epochs=10, # number of training epochs
        batch_size=100, # the size of each batch
        log_frequency=True,
        verbose=True,
        generator_dim=(128, 128, 128),
        discriminator_dim=(128, 128, 128),
        generator_lr=2e-4,
        discriminator_lr=2e-4,
        discriminator_steps=1,
        private=False,
        wandb=False,
        xai='SHAP'
    )

    model.fit(tabular_data)

    # Sample the generated synthetic data
    syn_data = model.sample(100)
    syn_data.to_csv(output_file)

    # sample = tabular_data.iloc[0]
    # explanation = model._xai_discriminator(sample)
    # print(explanation)

def get_datasets():
    folder = "C:/Users/maria/Code/master/xai-synthetic-health/notebooks/"
    # Read the dataset
    non_pre_dataset = pd.read_csv(os.path.join(folder, 'icu_dka_dataset_20250723.csv'))

    non_pre_dataset = non_pre_dataset.drop(columns=['subject_id'])
    outcome = non_pre_dataset.pop('in_hospital_death')
    non_pre_dataset['in_hospital_death'] = outcome
    pre_dataset = pd.read_csv(os.path.join(folder, 'imputed_dataset_0.csv'))
    outcome = pre_dataset.pop('in_hospital_death')
    pre_dataset['in_hospital_death'] = outcome
    # for col in ['aki', 'ckd', 'diabetes_without_cc', 'diabetes_with_cc', 'cardiac_disease', 'hypertension', 'copd',
    #             'ventilation', 'nitroprusside', 'vasopressor', 'albumin', 'emergency', 'in_hospital_death']:
    #     pre_dataset[col] = pre_dataset[col].astype(int)

    paper_features = [
        'apsiii', 'lactate_max', 'age', 'sofa', 'urineoutput', 'creatinine_max', 'wbc_max', 'bun_max',
        'temperature_max', 'calcium_max', 'chloride_max', 'los', 'ph_max', 'mbp_min', 'weight', 'hemoglobin_min',
        'platelet_min', 'osmolality_min', 'sodium_max', 'potassium_max', 'calcium_min', 'osmolality_max',
        'resp_rate_max', 'glucose_max', 'heart_rate_max', 'ph_min', 'vasopressor', 'cardiac_disease',
        'osmolality_initial', 'in_hospital_death']

    top_30 = [
        'race', 'sofa', 'lactate_max', 'vasopressor', 'age', 'wbc_min', 'diabetes_without_cc',
        'bun_min', 'calcium_max', 'potassium_initial', 'lactate_initial',
        'wbc_max', 'creatinine_min', 'osmolality_initial', 'apsiii', 'temperature_max',
        'creatinine_max', 'potassium_min', 'bicarbonate_initial', 'lactate_min', 'bun_initial', 'bicarbonate_max',
        'anion_gap_max', 'chloride_initial', 'osmolality_max', 'wbc_initial', 'bicarbonate_min', 'weight', 'ph_min',
        'platelet_max', 'in_hospital_death']
        # 'bun_max', 'potassium_max', 'resp_rate_initial', 'glucose_initial', 'sodium_max',
        # 'calcium_min', 'anion_gap_min', 'hemoglobin_max', 'mbp_initial', 'gcs']

    union_features = [col for col in pre_dataset.columns if col in set(paper_features) | set(top_30)]

    return non_pre_dataset, pre_dataset, pre_dataset[union_features]



def main():
    data_a, data_b, data_c = get_datasets()
    #
    # os.makedirs("outputs", exist_ok=True)
    # print("Running DP_CGAN test...")
    # print("A: No preprocessing of data.")
    # # tabular_data = pd.read_csv("../resources/icu_dka_dataset_20250723.csv")
    # run_dp_cgans(data_a, "outputs/syn_data_file_A.csv")
    # print("B: Preprocessing of data.")
    # run_dp_cgans(data_b, "outputs/syn_data_file_B.csv")
    # print("C: Preprocessing of data with limited features.")
    # run_dp_cgans(data_c, "outputs/syn_data_file_C.csv")
    # print("Done.")

    tabular_data = pd.read_csv("../resources/example_tabular_data_UCIAdult.csv")
    run_dp_cgans(tabular_data, "outputs/syn_data_file_UCIAdult.csv")


main()