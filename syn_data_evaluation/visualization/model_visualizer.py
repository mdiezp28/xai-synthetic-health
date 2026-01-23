

from matplotlib import pyplot as plt
from sklearn.calibration import calibration_curve

from syn_data_evaluation.evaluation.calibration import compute_bias_corrected_curve

from syn_data_evaluation.evaluation.calibration import compute_bias_corrected_curve


class ModelVisualizer:
    
    @staticmethod
    def plot_feature_importance(feat_imp_df, filename, top_n=30):
        
        top_feats = feat_imp_df.head(top_n)

        plt.figure(figsize=(10, 6))
        plt.barh(top_feats['feature'][::-1], top_feats['importance'][::-1])
        plt.xlabel('Importance Score')
        plt.title(f'Top {len(top_feats)} Feature Importances (XGBoost)')
        plt.tight_layout()
        plt.savefig(filename+'_feature_importance.png')
        plt.close()

    @staticmethod
    def plot_calibration(model, x_test, y_test, y_pred_proba, filename):
        # Apparent calibration
        prob_true_app, prob_pred_app = calibration_curve(y_test, y_pred_proba, n_bins=10, strategy='uniform')

        # Bias-corrected calibration
        mean_prob_pred, mean_prob_true = compute_bias_corrected_curve(model, x_test, y_test, n_boot=200, n_bins=10)

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
        plt.close()

    @staticmethod
    def plot_predicted_probabilities(y_pred_proba, y_test, threshold, filename):
        plt.figure(figsize=(8, 5))
        plt.scatter(range(len(y_test)), y_pred_proba, c=y_test, cmap='bwr', alpha=0.6)
        plt.axhline(0.5, color='gray', linestyle='--')  # default threshold
        plt.axhline(threshold, color='red', linestyle='--')  # optimal threshold
        plt.xlabel('Patient index')
        plt.ylabel('Predicted probability')
        plt.title('Predicted Probabilities vs True Outcome')
        plt.colorbar(label='True outcome (0=Survived, 1=Died)')
        plt.savefig(filename + '_predicted_prob.png')
        plt.close()

    @staticmethod
    def plot_roc_curve(fpr, tpr, thresholds, idx, auc, filename):
        """Plot ROC curve with optimal threshold."""
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
        plt.close()