import numpy as np
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, roc_auc_score, roc_curve, f1_score, \
    precision_score, average_precision_score


class ModelEvaluator:
    
    @staticmethod
    def evaluate_model(model, x_test, y_test):
        print("Evaluating model...")
        # Evaluate model
        y_pred = model.predict(x_test)
        y_pred_proba = model.predict_proba(x_test)

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
                auc=auc,
                auprc=auprc,
                accuracy=accuracy, 
                f1=f1,
                precision=precision,
                sensitivity=sensitivity, 
                specificity=specificity, 
                confusion_matrix= f"tn:{tn} fp:{fp} fn:{fn} tp:{tp}"
            ),
        }

    @staticmethod
    def get_optimal_threshold(fpr, tpr, thresholds):
        # https://towardsdatascience.com/optimal-threshold-for-imbalanced-classification-5884e870c293/
        youdensj = tpr - fpr
        idx = np.argmax(youdensj)
        best_threshold = thresholds[idx]

        return idx, best_threshold
    
    @staticmethod
    def find_threshold_idx(roc_thresholds, fixed_threshold):
        # roc thresholds are sorted from high to low; closest match is fine for plotting
        return int(np.argmin(np.abs(roc_thresholds - fixed_threshold)))

    @staticmethod
    def evaluate_with_threshold(y_pred_proba, y_test, threshold):
        """Evaluate model with threshold."""
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
                f1_adj=f1,
                precision_adj=precision,
                sensitivity_adj=sensitivity,
                specificity_adj=specificity,               
                confusion_matrix_adj= f"tn:{tn_adj} fp:{fp_adj} fn:{fn_adj} tp:{tp_adj}"
            ),
        }
