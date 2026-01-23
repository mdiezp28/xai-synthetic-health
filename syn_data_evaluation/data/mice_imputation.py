import pandas as pd
import miceforest as mf

class MICE_Imputer:    
    def __init__(self, train_data, categorical_cols):
        """Fit imputer on training data only"""
        data_to_impute = train_data.copy()
        self.categorical_cols = categorical_cols
        self.category_mappings = {}

        for col in self.categorical_cols:
            data_to_impute[col] = data_to_impute[col].astype('category')
            self.category_mappings[col] = data_to_impute[col].cat.categories
        
        self.kernel = mf.ImputationKernel(
            data_to_impute,
            random_state=42,
            num_datasets=1,
            mean_match_candidates=1,
        )
        self.kernel.mice(iterations=5)
        
    def transform_train(self):
        return self.kernel.complete_data(0)
    
    def transform_test(self, data):
        """Apply fitted imputer to new data"""
        data_to_transform = data.copy()
        
        for col in self.categorical_cols:
            if col in data_to_transform.columns:
                # Convert to categorical with the SAME categories as training
                data_to_transform[col] = pd.Categorical(
                    data_to_transform[col],
                    categories=self.category_mappings[col]
                )

        # Use impute_new_data as documented
        imputed_kernel = self.kernel.impute_new_data(new_data=data_to_transform)
        return imputed_kernel.complete_data(0)
    



# %% Get the data to impute
# data = pd.read_csv('C:/Users/maria/Code/master/xai-synthetic-health/notebooks/icu_dka_dataset_20250723.csv')
# # data = pd.read_csv('../dp_cgans/resources/icu_dka_dataset.csv')
# data_to_impute = data.copy()
# # %% Exclude ids and change type for categorical variables
# data_to_impute = data_to_impute.drop(columns=['subject_id'])
# %%
# missing_columns = get_missing_columns(data_to_impute)
# cols_to_impute = missing_columns.index.tolist()
# exclude_cols = [col for col in data_to_impute.columns if col not in cols_to_impute]

# def impute_missing_data(data, categorical_cols, save=False, save_path="C:/Users/maria/Code/master/xai-synthetic-health/notebooks/"):

#     # convert categorical variable to category
#     # categorical_col = ['race', 'gender', 'insurance']
#     # binary_col = ['aki', 'cardiac_disease', 'ckd', 'copd', 'hypertension']
#     data_to_impute = data.copy()
#     for col in categorical_cols:
#         data_to_impute[col] = data_to_impute[col].astype('category')

#     num_datasets = 1
#     kernel = mf.ImputationKernel(
#         data_to_impute,
#         random_state=42,
#         num_datasets=num_datasets,
#         mean_match_candidates=5,
#     )
#     kernel.mice(iterations=5)

    
#     print(kernel.complete_data)
#     for i in range(num_datasets):
#         imputed_df = kernel.complete_data(i)
#         print(f"\nImputed Dataset {i} Summary:")
#         print(imputed_df.describe())
#         if save:
#             imputed_df.to_csv(f"imputed_dataset_{i}.csv", index=False)
#     return kernel.complete_data(0)

# def analyse_imputed_data(data_to_impute, kernel, categorical_cols):
#     # %% Check the imputed data
#     # %% Folder to save the plots
#     output_dir = "C:/Users/maria/Code/master/xai-synthetic-health/notebooks/imputation_analysis"
#     os.makedirs(output_dir, exist_ok=True)

#     # %% Check the imputed data
#     def compare_distributions(column):
#         if column in categorical_cols:
#             return

#         if data_to_impute[column].isnull().sum() == 0:
#             return

#         original = data_to_impute[column].dropna()

#         plt.figure(figsize=(15, 6))
#         sns.histplot(original, kde=True, label='Original', color='gray', stat='density')

#         for i in range(5):
#             imputed = kernel.complete_data(i)[column]
#             sns.kdeplot(imputed, label=f'Imputed {i}', lw=1, linestyle='--')

#         plt.title(f"Distribution before and after: {column}")
#         plt.legend()
#         plt.savefig(f"{output_dir}/{column}_distribution_comparison.png")
#         plt.close()

#         # Kolmogorov–Smirnov (KS) Test
#         for i in range(5):
#             imputed = kernel.complete_data(i)[column]
#             stat, p = ks_2samp(original, imputed)
#             print(f"{column} - Imputed {i}: KS p-value = {p:.4f}")


#     # %% Analyse the distributions of the columns with missing values
#     columns_with_missing = data_to_impute.columns[data_to_impute.isnull().any()]
#     for col in columns_with_missing:
#         compare_distributions(col)

#     print("Done. Saved plots to:", output_dir)


#     # %% Check the imputed data
#     binary_col = ['aki', 'cardiac_disease', 'ckd', 'copd', 'hypertension']
#     for col in binary_col:
#         print(f"{col} value counts in imputed dataset 0:")
#         print(kernel.complete_data(0)[col].value_counts())

#     # %%
#     print(len(columns_with_missing))

