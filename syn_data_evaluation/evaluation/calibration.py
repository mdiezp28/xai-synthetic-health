

from sklearn.utils import resample
import numpy as np
import xgboost as xgb


def compute_bias_corrected_curve(model, test_scaled, y_test, n_boot, n_bins):
    """Compute bias-corrected calibration curve using bootstrap."""
    all_prob_true = []

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
        

    # Convert to array and compute mean ignoring NaNs
    mean_prob_true = np.nanmean(np.array(all_prob_true), axis=0)
    mean_prob_pred = np.array(bin_centers)

    return mean_prob_pred, mean_prob_true