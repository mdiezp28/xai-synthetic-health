from datetime import datetime
import pandas as pd
from dp_cgans import DP_CGAN
import os


def run_dp_cgans(tabular_data, output_file, generated_model_path, dataset_name="dpcgan"):
    print(f'Testing DP_CGAN')

    model = DP_CGAN(
        epochs=5, # number of training epochs
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
    print("Start training model")
    model.fit(tabular_data)

    ## Save the generative models
    now = datetime.now()
    current_time = now.strftime("%Y_%m_%d_%H_%M_%S")
    print('Training finished, saving the model')
    model_name = f'{generated_model_path}/{current_time}_dpcgans_icu_dka.pkl'
    model.save(model_name)

    print("load the trained file.")
    loaded_model=DP_CGAN.load(model_name)

    ## Sample the generated synthetic data
    nb_rows = len(tabular_data)
    print(f'Sampling {nb_rows} seen rows')
    loaded_model.sample(nb_rows).to_csv(output_file)
    
    # Sample the generated synthetic data
    # syn_data = model.sample(100)
    # syn_data.to_csv(output_file)

def main():
    generated_model_path = 'output/generators'
    if not os.path.exists(generated_model_path):
        os.makedirs(generated_model_path)

    #  Example:
    # tabular_data = pd.read_csv("../resources/example_tabular_data_UCIAdult.csv")
    # run_dp_cgans(tabular_data, "output/syn_data_file_UCIAdult.csv", "example_UCIAdult")

    # 30 most important features according to DKA paper
    # tabular_data = pd.read_csv("../resources/paper_features_icu_dka_dataset.csv")
    # run_dp_cgans(tabular_data, "output/syn_data_paper_features.csv", "paper_icu_dka")

    # 30 most important features according to XGBOOST model
    # tabular_data = pd.read_csv("../resources/xgboost_30_important_features_icu_dka_dataset.csv")
    # run_dp_cgans(tabular_data, "output/syn_data_xgboost.csv", "xgboost_icu_dka")

    # 30 most important features according to SHAP values
    tabular_data = pd.read_csv("../resources/shap_30_important_features_icu_dka_dataset.csv")
    run_dp_cgans(tabular_data, "output/syn_data_shap.csv", generated_model_path, "shap_icu_dka")


main()