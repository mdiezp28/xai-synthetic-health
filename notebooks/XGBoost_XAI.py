# %% import libraries
import math
import os
from pathlib import Path

import pandas as pd
import numpy as np
import xgboost as xgb
from lime.lime_tabular import LimeTabularExplainer
from sklearn.calibration import calibration_curve, CalibrationDisplay

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, roc_auc_score, roc_curve, f1_score, \
    precision_score, average_precision_score
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import shap


# %% Clean the data (split the data from the outcome)
def get_clean_data(dataset, features=None, drop_na=False):
    # Prepare data
    x = dataset.drop('in_hospital_death', axis=1)  # Exclude outcome
    y = dataset['in_hospital_death']
    # if dataset has 'subject_id', drop it
    if 'subject_id' in x.columns:
        x = x.drop('subject_id', axis=1)
    x_encoded = pd.get_dummies(x, drop_first=True)
    if features is not None:
        x_encoded = x_encoded[features]
    if drop_na:
        x_encoded = x_encoded.dropna()

    y_aligned = y.loc[x_encoded.index]

    return  x_encoded.astype(float), y_aligned

# %%
def normalise_data(train, test):
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train)
    test_scaled = scaler.transform(test)
    return train_scaled, test_scaled


# %% # Calculate scale_pos_weight for class imbalance
def get_scale_pos_weight(y_train):
    scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
    return scale_pos_weight



# %% # Train XGBoost classifier
#  parameters in the paper:
#         n_estimators=100,
#         max_depth=3,
#         eta=0.1,
#         gamma=0.25,
#         colsample_bytree=1,
#         min_child_weight=1,
#         subsample=0.5,
#         scale_pos_weight=10,
#         # scale_pos_weight=get_scale_pos_weight(y_train),
#         eval_metric='auc'
#         threashold = 0.017
def find_best_params(x_train, y_train):
    param_grid = {
        'n_estimators': [100],
        'max_depth': [3],
        'eta': [0.1],
        'gamma': [0.25, 0.17],
        'colsample_bytree': [0.8, 1.0],
        'min_child_weight': [1],
        'subsample': [0.5, 0.6],
        'scale_pos_weight': [10,15,17,20],
    }
    grid = GridSearchCV(xgb.XGBClassifier(eval_metric='auc'), param_grid, scoring='recall', cv=5)
    grid.fit(x_train, y_train)
    print("Best params:", grid.best_params_)
    print("Best recall (sensitivity):", grid.best_score_)

def train_xgboost(x_train, y_train):
    model_xgb = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3, # paper 3 [3, 4]
        eta=0.1,
        gamma=0.1, # paper 0.25 [0, 0.1]
        colsample_bytree=1,
        min_child_weight=1,
        subsample=0.7, # paper 0.5 [0.6, 0.8]
        scale_pos_weight=get_scale_pos_weight(y_train),
        eval_metric='auc'
    )
    model_xgb.fit(x_train, y_train)
    return model_xgb


# %%
def evaluate_model(model, test_scaled, y_test):
    print("Evaluating model...")
    # Evaluate model
    y_pred = model.predict(test_scaled)
    y_pred_proba = model.predict_proba(test_scaled)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    sensitivity = recall_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    auc = roc_auc_score(y_test, y_pred_proba)
    auprc = average_precision_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)

    print(f"Accuracy: {accuracy:.2f}")
    print(f"Sensitivity: {sensitivity:.2f}")
    print(f"Specificity: {specificity:.2f}")
    print(f"True Negative: {tn} False Positive: {fp} False Negative: {fn} True Positive: {tp}")
    print(f"AUC: {auc:.2f}")
    print(f"AUPRC (Average Precision): {auprc:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"F1 Score: {f1:.2f}")

    threshold = get_optimal_threshold(y_test, y_pred_proba, auc)
    print(f"Optimal Threshold: {threshold:.3f}")
    add_threshold_to_predictions(y_pred_proba, y_test, threshold)

    # Compute calibration curve
    prob_true, prob_pred = calibration_curve(y_test, y_pred_proba, n_bins=10)

    # n_bootstraps = 100
    # boot_prob_true = []
    # boot_prob_pred = []
    #
    # for i in range(n_bootstraps):
    #     # Resample training data with replacement
    #     X_res, y_res = resample(X_train, y_train, random_state=i)
    #
    #     # Train model on the resampled dataset
    #     model.fit(X_res, y_res)
    #
    #     # Predict probabilities on the original test set
    #     y_prob_boot = model.predict_proba(X_test)[:, 1]
    #
    #     # Compute calibration curve for this bootstrap
    #     pt, pp = calibration_curve(y_test, y_prob_boot, n_bins=10)
    #
    #     # Store results
    #     boot_prob_true.append(pt)
    #     boot_prob_pred.append(pp)

    # Plot
    plt.figure(figsize=(7, 7))

    # Ideal line
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Ideal')

    # Apparent
    CalibrationDisplay.from_predictions(
        y_test, y_pred_proba, n_bins=10, name='Apparent', ax=plt.gca(), color='blue', marker='o'
    )

    # Bias-corrected
    # plt.plot(prob_pred_bias_corrected, prob_true_bias_corrected, marker='s', color='red', label='Bias-corrected')

    plt.xlabel('Predicted probability')
    plt.ylabel('Observed probability')
    plt.title('Calibration Plot')
    plt.legend()
    plt.grid(True)
    plt.show()


def get_optimal_threshold(y_test, y_pred_proba, auc):
    # https://towardsdatascience.com/optimal-threshold-for-imbalanced-classification-5884e870c293/
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

    # df_fpr_tpr = pd.DataFrame({'FPR': fpr, 'TPR': tpr, 'Threshold': thresholds})
    # print(df_fpr_tpr.head())

    youdensj = tpr - fpr
    idx = np.argmax(youdensj)
    best_threshold = thresholds[idx]

    # Plot the ROC curve
    plot_roc_curve(fpr, tpr, thresholds, idx, auc)

    return best_threshold

def plot_roc_curve(fpr, tpr, thresholds, idx, auc):
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="blue", label=f"ROC curve (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random guess")

    # Mark the optimal threshold point
    plt.scatter(fpr[idx], tpr[idx], color="red", label=f"Best threshold = {thresholds[idx]:.3f}")

    plt.xlabel("1 - Specificity (FPR)")
    plt.ylabel("Sensitivity (TPR)")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

def add_threshold_to_predictions(y_pred_proba, y_test, threshold=0.015):
    y_pred_adjusted = (y_pred_proba >= threshold).astype(int)
    accuracy = accuracy_score(y_test, y_pred_adjusted)
    sensitivity = recall_score(y_test, y_pred_adjusted)
    precision = precision_score(y_test, y_pred_adjusted)
    f1 = f1_score(y_test, y_pred_adjusted)

    tn_adj, fp_adj, fn_adj, tp_adj = confusion_matrix(y_test, y_pred_adjusted).ravel()
    specificity = tn_adj / (tn_adj + fp_adj) if (tn_adj + fp_adj) > 0 else 0

    # print(f"\nWith threshold {threshold}:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Sensitivity: {sensitivity:.2f}")
    print(f"Specificity: {specificity:.2f}")
    print(f"True Negative: {tn_adj} False Positive: {fp_adj} False Negative: {fn_adj} True Positive: {tp_adj}")
    print(f"Precision: {precision:.2f}")
    print(f"F1 Score: {f1:.2f}")

# %%
def get_feature_imp_df(imp, features):
    # Create a DataFrame for importance
    feat_imp_df = pd.DataFrame({'feature': features, 'importance': imp})
    feat_imp_df = feat_imp_df.sort_values(by='importance', ascending=False)
    # print(feat_imp_df.head(20))  # Top 20 features
    return feat_imp_df


# %%
def show_feature_importance(df, filename, top_n=40):
    top_feats = df.head(top_n)

    plt.figure(figsize=(10, 6))
    plt.barh(top_feats['feature'][::-1], top_feats['importance'][::-1])
    plt.xlabel('Importance Score')
    plt.title(f'Top {len(top_feats)} Feature Importances (XGBoost)')
    plt.tight_layout()
    plt.savefig(filename+'_feature_importance.png')
    plt.show()

# %%
def generate_shap_explanations(model_xgb, X_train, feat_imp_df, features, filename):
    # SHAP explainer
    explainer = shap.Explainer(model_xgb)
    shap_values = explainer(X_train)

    feature_importance_shap = pd.DataFrame({
        'feature': features,
        'importance': np.abs(shap_values.values).mean(axis=0)
    }).sort_values(by='importance', ascending=False)

    # summary plot
    shap.summary_plot(shap_values, X_train, feature_names=feat_imp_df['feature'].tolist(), show=False)
    plt.savefig(f'{filename}_shap_summary.png')
    plt.close()

    # Waterfall plot
    shap.plots.waterfall(shap_values[0], show=False)  # for the 1st observation
    plt.savefig(f'{filename}_shap_1st_observation.png')
    plt.close()
    # shap.plots.waterfall(shap_values[1], max_display=4)  # for the 2nd observation only display 4

    # Absolute Mean SHAP
    # Which features are more important to the model.
    shap.plots.bar(shap_values, show=False)
    plt.savefig(f'{filename}_mean_shap.png')
    plt.close()

    # Force plot
    # force_plot = shap.plots.force(shap_values[0], matplotlib=False, show=False)
    # shap.save_html('test.html', force_plot)
    # force_plot = shap.plots.force(shap_values[0:100], matplotlib=False, show=False)
    # shap.save_html('test_100.html', force_plot)
    # shap.plots.force(shap_values[0:100], matplotlib=False, show=False)

    # Beeswarm plot
    # All of the shap values.
    # shap.plots.beeswarm(shap_values)
    # # Dependence plots
    # shap.plots.scatter(shap_values[:, 'lactate_max'])
    # shap.plots.scatter(shap_values[:, 'lactate_max'], color=shap_values[:, 'wbc_min'])
    # shap.plots.scatter(shap_values[:, 'wbc_min'])
    # shap.summary_plot(shap_values, X_train_top)
    # shap.summary_plot(shap_values, X_train_top, plot_type="violin", feature_names=X_train_top.columns)

    return feature_importance_shap

# %%

def generate_lime_explanations(model_xgb, X_train, feat_imp_df, filename, num_samples=5):
    """
    Generate LIME explanations for the first few samples in X_train.
    Saves explanation plots as PNG files.
    """

    # Convert X_train to numpy if it's a DataFrame
    if hasattr(X_train, "values"):
        X_train_np = X_train.values
    else:
        X_train_np = X_train

    # Create LIME explainer
    explainer = LimeTabularExplainer(
        training_data=X_train_np,
        feature_names=feat_imp_df['feature'].tolist(),
        class_names=[str(c) for c in np.unique(model_xgb.predict(X_train_np))],
        mode="classification"
    )

    # Explain a few samples
    for i in range(num_samples):
        exp = explainer.explain_instance(
            X_train_np[i],
            model_xgb.predict_proba,  # use predict_proba for classification
            num_features=10
        )

        # Save plot
        fig = exp.as_pyplot_figure()
        fig.savefig(f'{filename}_lime_instance_{i}.png')
        plt.close(fig)

        exp.save_to_file(f'{filename}_lime_instance_{i}.html')

# %%
def get_feature_importance(dataset, folder, filename, features=None, normalise=False):
    X, y = get_clean_data(dataset, features)
    # Split data (stratify - ensures class balance is maintained)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42,
                                                        stratify=y)
    print(f"Num of train data: {len(X_train)}. Num of test data: {len(X_test)}")
    print(f"Train Outcome: {y_train.value_counts()}")
    print(f"Test Outcome: {y_test.value_counts()}")
    if normalise:
        print("Normalising data...")
        X_train, X_test = normalise_data(X_train, X_test)

    # find_best_params(X_train, y_train)
    print("Training XGBoost model...")
    model_xgb = train_xgboost(X_train, y_train)
    print("Model trained successfully.")
    evaluate_model(model_xgb, X_test, y_test)

    # Get feature importance
    imp = model_xgb.feature_importances_
    features = X.columns
    feat_imp_df = get_feature_imp_df(imp, features)

    # "weight": number of times a feature appears in trees
    # "gain": average gain from splits using the feature (often most informative)
    # "cover": number of observations related to splits
    xgb.plot_importance(model_xgb, max_num_features=30, importance_type='gain', height=0.9)
    plt.title("Feature Importance (XGBoost)")
    plt.tight_layout()
    plt.show()

    show_feature_importance(feat_imp_df, os.path.join(folder, filename))
    print("Feature importance plot saved")

    print("Genarating SHAP explanations...")
    feature_importance_shap = generate_shap_explanations(model_xgb, X_train, feat_imp_df, features, os.path.join(folder, filename))
    print("SHAP explanations generated and saved.")

    print("Generating LIME explanations...")
    nan_rows = X.isna().any(axis=1)
    num_rows_to_remove = nan_rows.sum()
    if num_rows_to_remove == 0:
        generate_lime_explanations(model_xgb, X_train, feat_imp_df, os.path.join(folder, filename), num_samples=3)

    return feat_imp_df, feature_importance_shap


# def get_plot_filename(filename, plot_name):
#     filename = Path(filename)
#     return filename.with_name(f"{filename.stem}_{plot_name}{filename.suffix}")

def main():
    folder = "C:/Users/maria/Code/master/xai-synthetic-health/notebooks/"
    # Read the datasets
    non_pre_dataset = pd.read_csv(os.path.join(folder, 'icu_dka_dataset_20250723.csv'))
    non_pre_dataset['min/max'] = np.where(
        (non_pre_dataset['osmolality_min'].isna()) |
        (non_pre_dataset['osmolality_max'].isna()) |
        (non_pre_dataset['osmolality_min'] == 0) |
        (non_pre_dataset['osmolality_max'] == 0),
        np.nan,
        non_pre_dataset['osmolality_min'] / non_pre_dataset['osmolality_max']
    )
    pre_dataset = pd.read_csv(os.path.join(folder, 'imputed_dataset_0.csv'))
    pre_dataset['min/max'] = pre_dataset['osmolality_min'] / pre_dataset['osmolality_max']
    paper_features = [
        'apsiii', 'lactate_max', 'age', 'sofa', 'urineoutput', 'creatinine_max', 'wbc_max', 'bun_max', 'min/max',
        'temperature_max', 'calcium_max', 'chloride_max', 'los', 'ph_max', 'mbp_min', 'weight', 'hemoglobin_min',
        'platelet_min', 'osmolality_min', 'sodium_max', 'potassium_max', 'calcium_min', 'osmolality_max',
        'resp_rate_max', 'glucose_max', 'heart_rate_max', 'ph_min', 'vasopressor', 'cardiac_disease',
        'osmolality_initial']

    # min/max, the ratio of the minimum and maximum plasma osmolality; ‘I’_max, maximum body temperature; map_min
    feature_imp_folder = "xai"
    num_top = 30

    # No preprocessing
    print("-----------A: Getting feature importance for non-preprocessed dataset...-----------")
    no_pre_feat_imp_df, feature_importance_shap = get_feature_importance(
        non_pre_dataset,
        os.path.join(folder, feature_imp_folder),
        'A'
    )
    # Take the first 30 rows
    feat_subset = no_pre_feat_imp_df['feature'].head(30)

    # Compare with first 30 features of other_df
    other_subset = feature_importance_shap['feature'].head(30)

    if set(feat_subset) == set(other_subset):
        print("The first 30 features match")
    else:
        print("The first 30 features are different")

    # Optional: see differences
    diff1 = set(feat_subset) - set(other_subset)
    diff2 = set(other_subset) - set(feat_subset)
    print("In feat_imp_df but not in other_df:", diff1)
    print("In other_df but not in feat_imp_df:", diff2)
    # XGBoost with top features
    top_features = no_pre_feat_imp_df['feature'].head(num_top).tolist()
    print(f"---Top {num_top} Features Selected:---")
    print(top_features)

    get_feature_importance(
        non_pre_dataset,
        os.path.join(folder, feature_imp_folder),
        f'A_top_{num_top}',
        features=top_features,
    )

    # XGBoost with top features according to SHAP
    top_features = feature_importance_shap['feature'].head(num_top).tolist()
    print(f"---Top {num_top} SHAP Features Selected:---")
    print(top_features)

    get_feature_importance(
        non_pre_dataset,
        os.path.join(folder, feature_imp_folder),
        f'A_top_shap_{num_top}',
        features=top_features,
    )

    # Preprocessed dataset
    print("--------- B: Getting feature importance for preprocessed dataset...---------")
    pre_feat_imp_df, feature_importance_shap = get_feature_importance(
        pre_dataset,
        os.path.join(folder, feature_imp_folder),
        'B',
        normalise=True)

    # XGBoost with top features
    top_features = pre_feat_imp_df['feature'].head(num_top).tolist()
    print(f"---Top {num_top} Features Selected:---")
    print(top_features)
    pre_feat_imp_df, feature_importance_shap = get_feature_importance(
        pre_dataset,
        os.path.join(folder, feature_imp_folder),
        f'B_top_{num_top}',
        features=top_features,
        normalise=True)

    # XGBoost with top features according to SHAP
    top_features = feature_importance_shap['feature'].head(num_top).tolist()
    print(f"---Top {num_top} SHAP Features Selected:---")
    print(top_features)

    get_feature_importance(
        non_pre_dataset,
        os.path.join(folder, feature_imp_folder),
        f'B_top_shap_{num_top}',
        features=top_features,
    )

    # Paper dataset
    print("--------- C: Getting feature importance for paper dataset...---------")
    feature_importance, feature_importance_shap = get_feature_importance(
        pre_dataset,
        os.path.join(folder, feature_imp_folder),
        'C',
        features=paper_features,
        normalise=True)

    print(feature_importance_shap)


if __name__ == "__main__":
    main()
