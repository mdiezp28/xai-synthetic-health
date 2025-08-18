import pandas as pd
from dp_cgans import DP_CGAN, __version__

def run_dp_cgans():
    print(f'Testing DP_CGAN {__version__}')

    tabular_data=pd.read_csv("../resources/icu_dka_dataset_20250723.csv")

    model = DP_CGAN(
        epochs=10, # number of training epochs
        batch_size=100, # the size of each batch
        log_frequency=True,
        verbose=False,
        generator_dim=(128, 128, 128),
        discriminator_dim=(128, 128, 128),
        generator_lr=2e-4,
        discriminator_lr=2e-4,
        discriminator_steps=1,
        private=False,
        wandb=False
    )

    model.fit(tabular_data)

    # Sample the generated synthetic data
    syn_data = model.sample(100)
    syn_data.to_csv("syn_data_file.csv")

run_dp_cgans()
