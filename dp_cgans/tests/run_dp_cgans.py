from datetime import datetime
import os
import random

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
from dp_cgans import DP_CGAN
# from notebooks.sampling import post_process_synthetic_data

class DPCGANConfig: 
    def __init__(self, epochs=1000, batch_size=750, generator_dim=(128, 128, 128),
                 discriminator_dim=(128, 128, 128), generator_lr=5e-5, 
                 discriminator_lr=5e-5, discriminator_steps=5, private=False,
                 saved_transformer=os.getcwd()+'/fitted_transformer.pkl', pac=10, 
                 focus_k_features=0, focus_update_interval=50, xai_weight=0.0,
                 seed=42, deterministic=True):
        self.epochs = epochs
        self.batch_size = batch_size
        self.generator_dim = generator_dim
        self.discriminator_dim = discriminator_dim
        self.generator_lr = generator_lr
        self.discriminator_lr = discriminator_lr
        self.discriminator_steps = discriminator_steps
        self.private = private
        self.saved_transformer = saved_transformer
        self.pac = pac
        self.focus_k_features = focus_k_features
        self.focus_update_interval = focus_update_interval
        self.xai_weight = xai_weight
        self.seed = seed
        self.deterministic = deterministic


def set_global_seed(seed=42, deterministic=True):
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)

def run_dp_cgans(tabular_data, output_file, generated_model_path, config= None, save_output=False):
    print(f'Testing DP_CGAN')
    if config is None:
        config = DPCGANConfig()
    set_global_seed(config.seed, config.deterministic)
    print(f"Seed set to {config.seed}; deterministic algorithms enabled: {config.deterministic}")

    model = DP_CGAN(
        epochs=config.epochs, # number of training epochs
        batch_size=config.batch_size, # the size of each batch
        log_frequency=True,
        verbose=True,
        generator_dim=config.generator_dim,
        discriminator_dim=config.discriminator_dim,
        generator_lr=config.generator_lr,
        discriminator_lr=config.discriminator_lr,
        discriminator_steps=config.discriminator_steps,
        private=False,
        focus_k_features=config.focus_k_features,
        focus_update_interval=config.focus_update_interval,
        xai_weight=config.xai_weight,
        saved_transformer=config.saved_transformer,
        # dataset_name=config.dataset_name,
        pac=config.pac
    )
    print("Start training model")
    model.fit(tabular_data)

    ## Save the  model
    now = datetime.now()
    current_time = now.strftime("%Y_%m_%d_%H_%M_%S")
    print('Training finished, saving the model...')
    model_name = f'{generated_model_path}/{current_time}_dpcgans_icu_dka.pkl'
    model.save(model_name)
    print(f'Model saved to {model_name}')
    sample = None
    if save_output:
        sample = sample_dp_cgans(model_name, len(tabular_data), output_file, current_time)
    return  model_name, sample


def sample_dp_cgans(model_name, nb_rows, output_file, current_time=None, conditions=None, postprocess=False):
    print(f"Loading model from {model_name}")
    loaded_model=DP_CGAN.load(model_name)

    ## Sample the generated synthetic data
    sample_size = nb_rows + nb_rows//2 if postprocess else nb_rows

    if current_time is None:
        output_name = f'{output_file}.csv'
    else: 
        output_name = f'{output_file}_{current_time}.csv'

    print(f'Sampling {sample_size} seen rows')
    sample = loaded_model.sample(sample_size, conditions=conditions)

    if postprocess:
        # sample = post_process_synthetic_data(sample)
        print(f"Post-processing applied to synthetic data. Valid rows: {len(sample)}")
        if len(sample) > nb_rows:
            sample = sample[: nb_rows]
        #     sample = sample.sample(n=nb_rows // 3, random_state=42).reset_index(drop=True)
            print(f"Sampled down to {len(sample)} rows after post-processing.")
        
    sample.to_csv(output_name, index=False)
    print(f'Synthetic data saved to {output_name}')
    return sample
    # {"in_hospital_death":1}
    # loaded_model.sample(nb_rows).to_csv(f'output/{output_file}.csv')


def generate_balanced_samples(model_path, nb_rows, output_file, condition_col, postprocess=True):
    half_rows = nb_rows // 2

    print(f"Generating {half_rows} samples for {condition_col}=1")
    positive = sample_dp_cgans(model_path, half_rows, f"{output_file}_positive", 
                            postprocess=postprocess, conditions={condition_col: 1})
    
    print(f"Generating {half_rows} samples for {condition_col}=0")
    negative = sample_dp_cgans(model_path, half_rows, f"{output_file}_negative",
                            postprocess=postprocess, conditions={condition_col: 0})
    
    # Combine and shuffle
    combined = (
        pd.concat([positive, negative], ignore_index=True)
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )
    
    output_path = f'output/{output_file}_balanced.csv'
    combined.to_csv(output_path, index=False)
    print(f'Balanced synthetic data saved to {output_path}')
    return combined

def main():
    generated_model_path = 'output/generators'
    os.makedirs(generated_model_path, exist_ok=True)
    transformers_path = 'output/transformer'
    os.makedirs(transformers_path, exist_ok=True)

    print("cuda?", torch.cuda.is_available())

    # ===== SAMPLING FROM EXISTING MODELS =====
    # model_path = "output/generators/2026_01_03_20_13_38_dpcgans_icu_dka.pkl"
    # sample_dp_cgans(model_path, 1571, "syn_data_10f_xai_10_b", postprocess=False)

    # ---- Generate balanced samples ----
    # model_path = "output/generators/2026_01_03_20_13_38_dpcgans_icu_dka.pkl"
    # generate_balanced_conditional_samples(
    #     model_path, 1570, "shap_10f_750b_samples_postprocessed", 
    #     "in_hospital_death", postprocess=True
    # )

    # ===== TRAINING NEW MODELS =====
    #  Example:
    # tabular_data = pd.read_csv("../resources/example_tabular_data_UCIAdult.csv")
    # run_dp_cgans(tabular_data, "output/syn_data_file_UCIAdult.csv", "example_UCIAdult")

    tabular_data = pd.read_csv("../resources/folds/train_fold_1.csv").drop(columns=["sofa", "subject_id"], errors="ignore")
    # remove race column
    # tabular_data = tabular_data.drop("race", axis=1)
    print("Tabular data shape:", tabular_data.shape)
    print("Tabular data:", tabular_data.dtypes)
    
    # ===== Baseline =====
    config = DPCGANConfig(
        epochs=2000,
        batch_size=60, # ~6% of 1086 (64) and ~6% of 1711 (100)
        generator_dim=(128, 128, 128),
        discriminator_dim=(128, 128, 128),
        generator_lr=1e-4,
        discriminator_lr=1e-4,
        discriminator_steps=5,
        pac=10,
        private=False,
        saved_transformer=transformers_path+'/fitted_transformer.pkl'
    )
    # config = DPCGANConfig(
    #     epochs=3500,
    #     batch_size=130, # ~6% of 1086 (64) and ~6% of 1711 (100)
    #     generator_dim=(256, 256, 256),
    #     discriminator_dim=(256, 256, 256),
    #     generator_lr=2e-5,
    #     discriminator_lr=2e-5,
    #     discriminator_steps=5,
    #     pac=10,
    #     private=False,
    #     xai_type=None,
    #     xai_weight=0,
    #     saved_transformer=transformers_path+'/fitted_transformer.pkl'
    # )
    run_dp_cgans(tabular_data, "syn_data_fold_1", generated_model_path, config, save_output=True)
    # run_dp_cgans(tabular_data, "syn_data_1", generated_model_path, config_1, save_output=True)

    # ===== Synthetic data with SHAP =====
    # xai_list = [0.5, 1, 2, 5, 10, 15, 20]
    # config.xai_type = 'SHAP'
    # for weight in xai_list:
    #     config.xai_weight = weight
    #     run_dp_cgans(tabular_data, f"syn_data_10f_shap_{weight}", generated_model_path, config, save_output=True)
        

if __name__ == "__main__":
    main()