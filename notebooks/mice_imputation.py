# %%
import os

import pandas as pd
import miceforest as mf
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp

# %% Get the data to impute
data = pd.read_csv('C:/Users/maria/Code/master/xai-synthetic-health/notebooks/icu_dka_dataset_20250723.csv')
# data = pd.read_csv('../dp_cgans/resources/icu_dka_dataset.csv')
data_to_impute = data.copy()
# %%
# missing_columns = get_missing_columns(data_to_impute)
# cols_to_impute = missing_columns.index.tolist()
# exclude_cols = [col for col in data_to_impute.columns if col not in cols_to_impute]
# %% Exclude ids and change type for categorical variables
data_to_impute = data_to_impute.drop(columns=['subject_id'])
# convert categorical variable to category
categorical_col = ['race', 'gender', 'insurance']
# binary_col = ['aki', 'cardiac_disease', 'ckd', 'copd', 'hypertension']
for col in categorical_col:
    data_to_impute[col] = data_to_impute[col].astype('category')

# %% Create the MICE Imputation Kernel
kernel = mf.ImputationKernel(
    data_to_impute,
    random_state=42,
    num_datasets=5
)
# %% Impute the data
kernel.mice(iterations=5)

# %% Save the imputed data
print(kernel.complete_data)
for i in range(5):
    imputed_df = kernel.complete_data(i)
    print(f"\nImputed Dataset {i} Summary:")
    print(imputed_df.describe())

    imputed_df.to_csv(f"C:/Users/maria/Code/master/xai-synthetic-health/notebooks/imputed_dataset_{i}.csv", index=False)


# %% Check the imputed data
# %% Folder to save the plots
output_dir = "C:/Users/maria/Code/master/xai-synthetic-health/notebooks/imputation_analysis"
os.makedirs(output_dir, exist_ok=True)

# %% Check the imputed data
def compare_distributions(column):
    if column in categorical_col:
        return

    if data_to_impute[column].isnull().sum() == 0:
        return

    original = data_to_impute[column].dropna()

    plt.figure(figsize=(15, 6))
    sns.histplot(original, kde=True, label='Original', color='gray', stat='density')

    for i in range(5):
        imputed = kernel.complete_data(i)[column]
        sns.kdeplot(imputed, label=f'Imputed {i}', lw=1, linestyle='--')

    plt.title(f"Distribution before and after: {column}")
    plt.legend()
    plt.savefig(f"{output_dir}/{column}_distribution_comparison.png")
    plt.close()

    # Kolmogorov–Smirnov (KS) Test
    for i in range(5):
        imputed = kernel.complete_data(i)[column]
        stat, p = ks_2samp(original, imputed)
        print(f"{column} - Imputed {i}: KS p-value = {p:.4f}")


# %% Analyse the distributions of the columns with missing values
columns_with_missing = data_to_impute.columns[data_to_impute.isnull().any()]
for col in columns_with_missing:
    compare_distributions(col)

print("Done. Saved plots to:", output_dir)


# %% Check the imputed data
binary_col = ['aki', 'cardiac_disease', 'ckd', 'copd', 'hypertension']
for col in binary_col:
    print(f"{col} value counts in imputed dataset 0:")
    print(kernel.complete_data(0)[col].value_counts())

# %%
print(len(columns_with_missing))

