import pandas as pd
from dp_cgans import DP_CGAN, __version__
import torch

def run_dp_cgans(tabular_data, output_file, dataset_name="dpcgan"):
    print(f'Testing DP_CGAN')

    model = DP_CGAN(
        epochs=10, # number of training epochs
        batch_size=400, # the size of each batch
        log_frequency=True,
        verbose=True,
        generator_dim=(128, 128, 128),
        discriminator_dim=(128, 128, 128),
        generator_lr=2e-4,
        discriminator_lr=2e-4,
        discriminator_steps=1,
        private=False,
        xai='SHAP',
        xai_weight=0.3,
        dataset_name=dataset_name,
        cuda="cuda:0",
    )

    model.fit(tabular_data)

    # Sample the generated synthetic data
    syn_data = model.sample(100)
    syn_data.to_csv(output_file)

def main():
    #  Example:
    tabular_data = pd.read_csv("../resources/example_tabular_data_UCIAdult.csv")
    run_dp_cgans(tabular_data, "outputs/syn_data_file_UCIAdult.csv", "example_UCIAdult")

    # 30 most important features according to DKA paper
    # tabular_data = pd.read_csv("../resources/paper_features_icu_dka_dataset.csv")
    # run_dp_cgans(tabular_data, "outputs/syn_data_paper_features.csv", "paper_features_icu_dka")

    # 30 most important features according to XGBOOST model
    # tabular_data = pd.read_csv("../resources/xgboost_30_important_features_icu_dka_dataset.csv")
    # run_dp_cgans(tabular_data, "outputs/syn_data_xgboost.csv", "xgboost_icu_dka")

    # 30 most important features according to SHAP values
    # tabular_data = pd.read_csv("../resources/shap_30_important_features_icu_dka_dataset.csv")
    # run_dp_cgans(tabular_data, "outputs/syn_data_shap.csv", "shap_icu_dka")




main()