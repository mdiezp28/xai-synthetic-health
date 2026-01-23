# %% import libraries
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
from sklearn.utils import resample
import seaborn as sns
from mice_imputation import MICE_Imputer
import evaluation.utils as utils
import sampling


# %%
def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

# %% Clean the data (split the data from the outcome)
def get_raw_xy(dataset, target_col="in_hospital_death"):
    # Prepare data
    x = dataset.drop(target_col, axis=1)  # Exclude outcome
    y = dataset[target_col]

    x = x.loc[:, ~x.columns.str.contains(
        r'^Unnamed|subject_id', 
        case=False, regex=True
    )]
    return  x, y

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
        n_estimators=150,
        max_depth=4, # paper 3 [3, 4]
        learning_rate=0.1,
        gamma=0.1, # paper 0.25 [0, 0.1]
        colsample_bytree=0.7,
        min_child_weight=5,
        subsample=0.7, # paper 0.5 [0.6, 0.8]
        scale_pos_weight=get_scale_pos_weight(y_train),
        eval_metric='auc',
        max_delta_step=1,
        reg_alpha=0.3,
        reg_lambda=2.0,
        # colsample_bytree=0.7,  # Only use 50% of features per tree
        # colsample_bylevel=0.8,  # Additional sampling at each level
    )
    model_xgb.fit(x_train, y_train)
    return model_xgb


# %%
def evaluate_model(model, x_test, y_test):
    print("Evaluating model...")
    # Evaluate model
    y_pred = model.predict(x_test)
    y_pred_proba = model.predict_proba(x_test)[:, 1]

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


    return {
        "y_pred": y_pred,
        "y_pred_proba": y_pred_proba,
        "metrics": dict(
            accuracy=accuracy, 
            sensitivity=sensitivity, 
            specificity=specificity, 
            precision=precision, 
            f1=f1,
            auc=auc,
            auprc=auprc, 
            confusion_matrix= f"tn:{tn} fp:{fp} fn:{fn} tp:{tp}"
        ),
    }

# %%
def get_optimal_threshold(y_test, y_pred_proba):
    # https://towardsdatascience.com/optimal-threshold-for-imbalanced-classification-5884e870c293/
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

    # df_fpr_tpr = pd.DataFrame({'FPR': fpr, 'TPR': tpr, 'Threshold': thresholds})
    # print(df_fpr_tpr.head())

    youdensj = tpr - fpr
    idx = np.argmax(youdensj)
    best_threshold = thresholds[idx]

    return {
        "threshold": best_threshold,
        "metrics": {
            "fpr": fpr,
            "tpr": tpr,
            "thresholds": thresholds,
            "idx": idx,
        }
    }


# %%
def compute_bias_corrected_curve(model, test_scaled, y_test, n_boot, n_bins):
    """Compute bias-corrected calibration curve using bootstrap."""
    all_prob_true, all_prob_pred = [], []

    # Define fixed bin edges
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    for i in range(n_boot):
        X_res, y_res = resample(test_scaled, y_test, replace=True, random_state=i)

        # Refit model on bootstrap
        model_res = xgb.XGBClassifier(
            n_estimators=model.n_estimators,
            max_depth=model.max_depth,
            learning_rate=model.learning_rate,
            gamma=model.gamma,
            colsample_bytree=model.colsample_bytree,
            min_child_weight=model.min_child_weight,
            subsample=model.subsample,
            scale_pos_weight=model.scale_pos_weight,
            eval_metric='auc',
            use_label_encoder=False,
            verbosity=0
        )
        model_res.fit(X_res, y_res)

        # Predict on original dataset
        y_pred_res = model_res.predict_proba(test_scaled)[:, 1]

        # Compute fraction of positives per fixed bin
        true_bin = []
        for start, end in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (y_pred_res >= start) & (y_pred_res < end)
            if mask.sum() > 0:
                true_bin.append(y_test[mask].mean())
            else:
                true_bin.append(np.nan)  # empty bin

        all_prob_true.append(true_bin)
        all_prob_pred.append(bin_centers)

    # Convert to array and compute mean ignoring NaNs
    mean_prob_true = np.nanmean(np.array(all_prob_true), axis=0)
    mean_prob_pred = np.array(bin_centers)

    return mean_prob_pred, mean_prob_true

# %%
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

    return {
        "metrics": dict(
            accuracy_adj=accuracy,
            sensitivity_adj=sensitivity,
            specificity_adj=specificity,
            precision_adj=precision,
            f1_adj=f1,
            confusion_matrix_adj= f"tn:{tn_adj} fp:{fp_adj} fn:{fn_adj} tp:{tp_adj}"
        ),
    }

# %%
def get_feature_imp_df(imp, features):
    # Create a DataFrame for importance
    feat_imp_df = pd.DataFrame({'feature': features, 'importance': imp})
    feat_imp_df = feat_imp_df.sort_values(by='importance', ascending=False)
    # print(feat_imp_df.head(20))  # Top 20 features
    return feat_imp_df


# %%
def show_feature_importance(df, filename, top_n=30):
    top_feats = df.head(top_n)

    plt.figure(figsize=(10, 6))
    plt.barh(top_feats['feature'][::-1], top_feats['importance'][::-1])
    plt.xlabel('Importance Score')
    plt.title(f'Top {len(top_feats)} Feature Importances (XGBoost)')
    plt.tight_layout()
    plt.savefig(filename+'_feature_importance.png')
    # plt.show()
    plt.close()

# %%
def generate_shap_explanations(model_xgb, x_train_df, filename):
    # SHAP explainer
    explainer = shap.TreeExplainer(model_xgb)
    shap_values = explainer(x_train_df)

    feature_importance_shap = pd.DataFrame({
        'feature': x_train_df.columns,
        'importance': np.abs(shap_values.values).mean(axis=0)
    }).sort_values(by='importance', ascending=False)
    
    # summary plot
    shap.summary_plot(shap_values, x_train_df, feature_names=x_train_df.columns, show=False)
    plt.savefig(f'{filename}_shap_summary.png')
    plt.close()

    # Waterfall plot
    shap.plots.waterfall(shap_values[0], show=False)  # for the 1st observation
    plt.savefig(f'{filename}_shap_1st_observation.png')
    plt.close()
    # shap.plots.waterfall(shap_values[1], max_display=4)  # for the 2nd observation only display 4

    # Absolute Mean SHAP
    # Which features are more important to the model.
    shap.plots.bar(shap_values, show=False, max_display=30)
    plt.savefig(f'{filename}_mean_shap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    feature_importance_shap_grouped = group_dummy_feature_importance(feature_importance_shap)
    importance_df = feature_importance_shap_grouped.sort_values('importance', ascending=True)

    plt.figure(figsize=(10, 8))
    plt.barh(importance_df['feature'], importance_df['importance'], color='#ff0050', edgecolor='black')
    plt.xlabel('Mean |SHAP Value|')
    plt.title('Global Feature Importance (Grouped Categorical Features)')
    plt.tight_layout()
    plt.savefig(f'{filename}_mean_shap_grouped_.png', dpi=300, bbox_inches='tight')
    plt.close()

    # # Create a proper SHAP Explanation object with grouped values
    # grouped_shap_values_for_plotting = shap.Explanation(
    #     values=feature_importance_shap_grouped['importance'].values.reshape(1, -1),  # 1 row for bar plot
    #     feature_names=feature_importance_shap_grouped['feature'].tolist()
    # )

    # shap.plots.bar(grouped_shap_values_for_plotting, max_display=30, show=False)
    # plt.savefig(f'{filename}_mean_shap_grouped.png', dpi=300, bbox_inches='tight')
    # plt.close()

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

    return feature_importance_shap, feature_importance_shap_grouped

# %%
def generate_lime_explanations(model_xgb, X_train_df, feat_imp_df, filename, num_samples=5):
    """
    Generate LIME explanations for the first few samples in X_train.
    Saves explanation plots as PNG files.
    """
    X_train_np = X_train_df.values
    feature_names = X_train_df.columns.tolist()
    if hasattr(model_xgb, "classes_"):
        class_names = [str(c) for c in model_xgb.classes_]
    else:
        class_names = ['0', '1']

    # Create LIME explainer
    explainer = LimeTabularExplainer(
        training_data=X_train_np,
        feature_names=feature_names,
        class_names=class_names,
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
def plot_calibration(eva, model, x_test_np, y_test, filename):
    y_pred_proba = eva["y_pred_proba"]

    # Apparent calibration
    prob_true_app, prob_pred_app = calibration_curve(y_test, y_pred_proba, n_bins=10, strategy='uniform')

    # Bias-corrected calibration
    mean_prob_pred, mean_prob_true = compute_bias_corrected_curve(model, x_test_np, y_test, n_boot=200, n_bins=10)

    # Plot
    plt.figure(figsize=(7, 7))
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Ideal')
    plt.plot(prob_pred_app, prob_true_app, "o-", label='Apparent')
    plt.plot(mean_prob_pred, mean_prob_true, "o-", label='Bias-corrected')
    plt.xlabel('Predicted probability')
    plt.ylabel('Observed probability')
    plt.title('Calibration Plot')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename + '_calibration.png')
    # plt.show()
    plt.close()

def plot_predicted_probabilities(eva, threshold, y_test, filename):
    y_pred_proba = eva["y_pred_proba"]

    plt.figure(figsize=(8, 5))
    plt.scatter(range(len(y_test)), y_pred_proba, c=y_test, cmap='bwr', alpha=0.6)
    plt.axhline(0.5, color='gray', linestyle='--')  # default threshold
    plt.axhline(threshold, color='red', linestyle='--')  # optimal threshold
    plt.xlabel('Patient index')
    plt.ylabel('Predicted probability')
    plt.title('Predicted Probabilities vs True Outcome')
    plt.colorbar(label='True outcome (0=Survived, 1=Died)')
    plt.savefig(filename + '_predicted_prob.png')
    # plt.show()
    plt.close()


def plot_roc_curve(fpr, tpr, thresholds, idx, auc, filename):
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
    plt.savefig(filename + '_roc_curve.png')
    # plt.show()
    plt.close()


def group_dummy_feature_importance(feat_imp_df, sep = "__"):
    df = feat_imp_df.copy()

    # Detect dummy features by separator
    is_dummy = df["feature"].str.contains(sep)

    # Extract original feature name for dummy columns, leave others unchanged
    df["original_feature"] = df["feature"].where(~is_dummy, df["feature"].str.split(sep, n=1).str[0])

    # Group by original feature and sum importance
    grouped = df.groupby("original_feature")["importance"].sum().sort_values(ascending=False).reset_index()
    grouped = grouped.rename(columns={"original_feature": "feature"})
    return grouped

    
# %%
def run_experiment(train_data, test_data=None, impute=False, categorical_cols=None, normalise=False, out_dir="outputs", exp_name="experiment"):
    ensure_dir(out_dir)
    filename = os.path.join(out_dir, exp_name)

    X, y = get_raw_xy(train_data)
    if test_data is None:
        # Split data (stratify - ensures class balance is maintained)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    else:
        X_test, y_test = get_raw_xy(test_data)
        X_train, y_train = X, y
       
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    print(f"Num of train data: {len(X_train)}. Num of test data: {len(X_test)}")
    print(f"Train Outcome: {y_train.value_counts()}")
    print(f"Test Outcome: {y_test.value_counts()}")

    if impute:
        print("Imputing missing data...")
        imputer = MICE_Imputer(X_train, categorical_cols)
        X_train = imputer.transform_train()
        X_test = imputer.transform_test(X_test)
        print("Imputation complete.")

    # Encode categorical variables
    X_train = pd.get_dummies(X_train, prefix_sep='__')
    X_test = pd.get_dummies(X_test, prefix_sep='__')

    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)     

    if normalise:
        print("Normalising data...")
        x_train_np, x_test_np = normalise_data(X_train, X_test)

        X_train_model = pd.DataFrame(x_train_np, columns=X_train.columns, index=X_train.index)
        X_test_model  = pd.DataFrame(x_test_np,  columns=X_test.columns,  index=X_test.index)
    else:
        X_train_model = X_train.copy()
        X_test_model  = X_test.copy()
        x_train_np = X_train.values
        x_test_np = X_test.values

   

    # find_best_params(X_train, y_train)
    print("Training XGBoost model...")
    model_xgb = train_xgboost(x_train_np, y_train)
    print("Model trained successfully.")

    eva = evaluate_model(model_xgb, x_test_np, y_test)
    optimal_metrics = get_optimal_threshold(y_test, eva["y_pred_proba"])
    best_threshold = optimal_metrics["threshold"]
    # Plot the ROC curve
    plot_roc_curve(optimal_metrics["metrics"]["fpr"], optimal_metrics["metrics"]["tpr"], optimal_metrics["metrics"]["thresholds"], optimal_metrics["metrics"]["idx"], eva["metrics"]["auc"], filename)
    print(f"Optimal Threshold: {best_threshold:.3f}")
    eva_adj = add_threshold_to_predictions(eva["y_pred_proba"], y_test, best_threshold)
    # Save evaluation metrics
    evaluation = pd.concat([pd.DataFrame([eva["metrics"]]), pd.DataFrame([eva_adj["metrics"]])], axis=1)
    evaluation["threshold"] = best_threshold
    evaluation.insert(0, "run_name", exp_name)

    csv_path = Path(out_dir).parent / "evaluation_metrics.csv"
    evaluation.to_csv(csv_path,
                       mode='a',
                       header = not os.path.exists(csv_path),
                       index=False)
    plot_calibration(eva, model_xgb, x_test_np, y_test, filename)
    plot_predicted_probabilities(eva, best_threshold, y_test, filename)

    # Get feature importance
    imp = model_xgb.feature_importances_
    features = X_train.columns
    feat_imp_df = get_feature_imp_df(imp, features)
    feat_imp_grouped = group_dummy_feature_importance(feat_imp_df)
    # Save importance to CSV
    feat_imp_df.to_csv(filename + '_feature_importance.csv', index=False)
    feat_imp_grouped.to_csv(filename + '_feature_importance_grouped.csv', index=False)
    # "weight": number of times a feature appears in trees
    # "gain": average gain from splits using the feature (often most informative)
    # "cover": number of observations related to splits
    # model_xgb.get_booster().feature_names = features.tolist()
    # xgb.plot_importance(model_xgb.get_booster(), max_num_features=30, importance_type='gain', height=0.9)
    # plt.title("Feature Importance (XGBoost)")
    # plt.tight_layout()
    # plt.show()

    show_feature_importance(feat_imp_df, filename)
    show_feature_importance(feat_imp_grouped, filename + "_grouped")
    print("Feature importance plot saved")

    print("Genarating SHAP explanations...")
    feature_importance_shap, feature_importance_shap_grouped = generate_shap_explanations(model_xgb, X_train_model, filename)
    feature_importance_shap.to_csv(filename + '_feature_importance_shap.csv', index=False)
    feature_importance_shap_grouped.to_csv(filename + '_feature_importance_shap_grouped.csv', index=False)
    print("SHAP explanations generated and saved.")

    print("Generating LIME explanations...")
    nan_rows = X_train.isna().any(axis=1)
    num_rows_to_remove = nan_rows.sum()
    if num_rows_to_remove == 0:
        generate_lime_explanations(model_xgb, X_train_model, feat_imp_df, filename, num_samples=3)
    else:
        print("Skipping LIME (found NaNs).")

    return feat_imp_df, feat_imp_grouped, feature_importance_shap, feature_importance_shap_grouped




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
    synthetic_folder = "syn"
    num_top = 30

    # Synthetic dataset
    print("----------- SYNTHETIC DATA 1: Getting feature importance for non-preprocessed dataset...-----------")
    syn_dataset_1 = pd.read_csv(os.path.join(folder, synthetic_folder, '2025_09_18_12_46_41_dpcgans_1571_rows_icu_dka.csv'))
    no_pre_feat_imp_df, feat_imp_grouped, feature_importance_shap = run_experiment(
        syn_dataset_1,
        os.path.join(folder, synthetic_folder),
        'syn_1'
    )

    print("-----------SYNTHETIC DATA 2: Getting feature importance for non-preprocessed dataset...-----------")
    syn_dataset_2 = pd.read_csv(os.path.join(folder, synthetic_folder, '2025_09_18_14_56_11_dpcgans_1571_rows_icu_dka.csv'))
    no_pre_feat_imp_df, feat_imp_grouped, feature_importance_shap = run_experiment(
        train_data=syn_dataset_2,
        out_dir=os.path.join(folder, synthetic_folder),
        exp_name='syn_2'
    )

    # No preprocessing
    # print("-----------A: Getting feature importance for non-preprocessed dataset...-----------")
    # no_pre_feat_imp_df, feat_imp_grouped , feature_importance_shap = get_feature_importance(
    #     non_pre_dataset,
    #     os.path.join(folder, feature_imp_folder),
    #     'A'
    # )

    # # Take the first 30 rows
    # feat_subset = no_pre_feat_imp_df['feature'].head(30)
    # # Compare with first 30 features of other_df
    # other_subset = feature_importance_shap['feature'].head(30)
    #
    # if set(feat_subset) == set(other_subset):
    #     print("The first 30 features match")
    # else:
    #     print("The first 30 features are different")
    #
    # diff1 = set(feat_subset) - set(other_subset)
    # diff2 = set(other_subset) - set(feat_subset)
    # print("In feat_imp_df but not in other_df:", diff1)
    # print("In other_df but not in feat_imp_df:", diff2)
    #
    # XGBoost with top features
    # top_features = no_pre_feat_imp_df['feature'].head(num_top).tolist()
    # print(f"---Top {num_top} Features Selected:---")
    # print(top_features)
    #
    # get_feature_importance(
    #     non_pre_dataset,
    #     os.path.join(folder, feature_imp_folder),
    #     f'A_top_{num_top}',
    #     features=top_features,
    # )
    #
    # # XGBoost with top features according to SHAP
    # top_features = feature_importance_shap['feature'].head(num_top).tolist()
    # print(f"---Top {num_top} SHAP Features Selected:---")
    # print(top_features)
    #
    # get_feature_importance(
    #     non_pre_dataset,
    #     os.path.join(folder, feature_imp_folder),
    #     f'A_top_shap_{num_top}',
    #     features=top_features,
    # )

    # Preprocessed dataset
    # print("--------- B: Getting feature importance for preprocessed dataset...---------")
    # pre_feat_imp_df, feat_imp_grouped, feature_importance_shap = get_feature_importance(
    #     pre_dataset,
    #     os.path.join(folder, feature_imp_folder),
    #     'B',
    #     normalise=True)
    #
    # # XGBoost with top features
    # top_features = pre_feat_imp_df['feature'].head(num_top).tolist()
    # print(f"---Top {num_top} Features Selected:---")
    # print(top_features)
    # pre_feat_imp_df, feat_imp_grouped, feature_importance_shap = get_feature_importance(
    #     pre_dataset,
    #     os.path.join(folder, feature_imp_folder),
    #     f'B_top_{num_top}',
    #     features=top_features,
    #     normalise=True)
    #
    # # XGBoost with top features according to SHAP
    # top_features = feature_importance_shap['feature'].head(num_top).tolist()
    # print(f"---Top {num_top} SHAP Features Selected:---")
    # print(top_features)
    #
    # get_feature_importance(
    #     non_pre_dataset,
    #     os.path.join(folder, feature_imp_folder),
    #     f'B_top_shap_{num_top}',
    #     features=top_features,
    # )
    #
    # # Paper dataset
    # print("--------- C: Getting feature importance for paper dataset...---------")
    # feature_importance, feat_imp_grouped, feature_importance_shap = get_feature_importance(
    #     pre_dataset,
    #     os.path.join(folder, feature_imp_folder),
    #     'C',
    #     features=paper_features,
    #     normalise=True)
    #
    # print(feature_importance_shap)
def run_experiment_list(synthetic_data, train_data, test_data, result_path, data_name, impute=False):
    synthetic_data = sampling.post_process_synthetic_data(synthetic_data, True)
    # Before running the experiment, check for inf values
    normalise_data = True
    categorical_cols=["gender", "race", "insurance"]
    result_path = result_path + data_name + "/"

    print("Experiment 1: \n Training data: Synthetic data.\n Test data: real test data.")
    exp_name = "exp1_" + data_name
    run_experiment(
        train_data=synthetic_data, 
        test_data=test_data,
        out_dir=result_path+exp_name+"/",
        exp_name=exp_name,
        impute=impute,
        normalise=normalise_data,
        categorical_cols=categorical_cols,
    )
    
    print("Experiment 2: \n Training data: Hybrid data.\n Test data: real test data.")
    percentages = [0.1, 0.3, 0.5, 0.7]
    for perc in percentages:
        exp_name = f"exp2_{data_name}_{int(perc*100)}perc"

        hybrid_data = utils.get_hybrid_data_constant_size(
            real_data=train_data,
            syn_data=synthetic_data,
            syn_data_percentage=perc
        )
        run_experiment(
            train_data=hybrid_data, 
            test_data=test_data,
            out_dir=result_path+exp_name+"/",
            exp_name=exp_name,
            impute=impute,
            normalise=normalise_data,
            categorical_cols=categorical_cols,
        )


    print("Experiment 3: \n Training data: Hybrid data - Augmentation.\n Test data: real test data.")
    percentages = [0.1, 0.3, 0.5, 0.7]
    for perc in percentages:
        exp_name = f"exp3_{data_name}_{int(perc*100)}perc"

        hybrid_data = utils.get_hybrid_data_augmentation(
            real_data=train_data,
            syn_data=synthetic_data,
            syn_data_percentage=perc
        )
        run_experiment(
            train_data=hybrid_data, 
            test_data=test_data,
            out_dir=result_path+exp_name+"/",
            exp_name=exp_name,
            impute=impute,
            normalise=normalise_data,
            categorical_cols=categorical_cols,
        )


if __name__ == "__main__":
    main()