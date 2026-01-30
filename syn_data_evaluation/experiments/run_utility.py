from dataclasses import dataclass
import os
from pathlib import Path
from typing import List

import pandas as pd
from sklearn.model_selection import StratifiedKFold
from syn_data_evaluation.data import dataset_mixing, postprocessing
from syn_data_evaluation.data.preprocessing import DataPreprocessor
from syn_data_evaluation.evaluation.model_evaluator import ModelEvaluator
from syn_data_evaluation.models.xgboost import XGBoostModel
from syn_data_evaluation.visualization.explainability import ModelExplainer
from syn_data_evaluation.visualization.model_visualizer import ModelVisualizer

@dataclass
class DataConfig:
    target_col: str = "in_hospital_death"
    categorical_cols: List[str] = None
    test_size: float = 0.3
    
    def __post_init__(self):
        if self.categorical_cols is None:
            self.categorical_cols = ["gender", "race", "insurance"]

@dataclass
class ExperimentConfig:
    impute: bool = False
    normalize: bool = True


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

class ExperimentRunner:

    def __init__(self):
        self.data_config = DataConfig() 
        self.exp_config = ExperimentConfig()
        
        self.preprocessor = DataPreprocessor()
        self.model_wrapper = XGBoostModel()
        self.evaluator = ModelEvaluator()
        self.visualizer = ModelVisualizer()
        self.explainer = ModelExplainer()

    def run_experiment(self, train_data, test_data=None, out_dir="outputs", exp_name="experiment"):
        ensure_dir(out_dir)
        filename = os.path.join(out_dir, exp_name)

        # Data preparation
        data = self.preprocessor.prepare_data(
            train_data=train_data,
            test_data=test_data,
            impute=self.exp_config.impute,
            categorical_cols=self.data_config.categorical_cols,
            normalize=self.exp_config.normalize,
            test_size=self.data_config.test_size,
            )
        X_train_model = data['X_train_model']
        X_test_model = data['X_test_model']
        y_train = data['y_train']
        y_test = data['y_test']
        
        print(f"Num of train data: {len(X_train_model)}. "
              f"Num of test data: {len(X_test_model)}")
        print(f"Train Outcome: {y_train.value_counts()}")
        print(f"Test Outcome: {y_test.value_counts()}")

    
        # find_best_params(X_train, y_train)
        print("Training XGBoost model...")
        model = self.model_wrapper.train(X_train_model.values, y_train)
        
        # Evaluate
        metrics = self.evaluator.evaluate_model(self.model_wrapper, X_test_model.values, y_test)
        # Find optimal threshold
        optimal_metrics = self.evaluator.get_optimal_threshold(y_test, metrics["y_pred_proba"])
        best_threshold = optimal_metrics["threshold"]
        print(f"Optimal Threshold: {best_threshold:.3f}") 

        # Plot the ROC curve
        self.visualizer.plot_roc_curve(
            optimal_metrics["metrics"]["fpr"],
            optimal_metrics["metrics"]["tpr"],
            optimal_metrics["metrics"]["thresholds"], 
            optimal_metrics["metrics"]["idx"], 
            metrics["metrics"]["auc"], 
            filename
        )

        # Evaluate with optimal threashold
        metrics_thr = self.evaluator.evaluate_with_threshold(metrics["y_pred_proba"], y_test, best_threshold)

        # Save evaluation metrics
        self._save_evaluation_metrics(metrics, metrics_thr, best_threshold, exp_name, out_dir)

        # Visualization
        self.visualizer.plot_calibration(model, X_test_model.values, y_test, metrics["y_pred_proba"], filename)
        self.visualizer.plot_predicted_probabilities(metrics["y_pred_proba"], y_test, best_threshold, filename)

        # Get feature importance
        feat_imp_df = self.explainer.get_feat_importance(model, X_train_model.columns)
        feat_imp_grouped = self.explainer.group_dummy_feature_importance(feat_imp_df)

        # Save importance to CSV
        feat_imp_df.to_csv(filename + '_feature_importance.csv', index=False)
        feat_imp_grouped.to_csv(filename + '_feature_importance_grouped.csv', index=False)

        # Plot importance
        self.visualizer.plot_feature_importance(feat_imp_df, filename)
        self.visualizer.plot_feature_importance(feat_imp_grouped, filename + "_grouped")

        # SHAP explanations
        shap_imp, shap_imp_grouped = self.explainer.generate_shap_explanations(model, X_train_model, filename)

        shap_imp.to_csv(filename + '_feature_importance_shap.csv', index=False)
        shap_imp_grouped.to_csv(filename + '_feature_importance_shap_grouped.csv', index=False)
    

        # LIME explanations
        if not data['X_train'].isna().any().any():
            self.explainer.generate_lime_explanations(model, X_train_model, filename, num_samples=3)
        else:
            print("Skipping LIME (found NaNs).")

        return {
            "model": model,
            "metrics": metrics,
            "metrics_thr": metrics_thr,
            "threshold": best_threshold,
            'feat_imp_df': feat_imp_df,
            'feat_imp_grouped': feat_imp_grouped,
            'shap_imp': shap_imp,
            'shap_imp_grouped': shap_imp_grouped
        }

    def run_stratified_kfold(self, real_data: pd.DataFrame, syn_data: pd.DataFrame =None, syn_percentage: float=0.3, n_splits: int = 5, out_dir: str = "outputs", exp_name: str = "cv_experiment", augmentation: bool = False):
        """
        Run stratified k-fold cross-validation on full dataset.
        
        All real data is used.
        
        Args:
            data: Full dataset (will be split into k folds)
            n_splits: Number of folds (default: 5)
            out_dir: Output directory
            exp_name: Base name for experiment
            augmentation: Whether to use data augmentation for hybrid data
            
        Returns:
            Dictionary with aggregated results across folds
        """
        ensure_dir(out_dir)
        
        # Extract features and target
        X, y = self.preprocessor.get_raw_xy(real_data)
        
        # Initialize stratified k-fold
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, 
                             random_state=42)
        
        # Storage for results
        cv_metrics = []
        # cv_feature_importance = []
        
        print(f"\n{'='*80}")
        print(f"Running {n_splits}-Fold Stratified Cross-Validation")
        print(f"Total samples: {len(X)}")
        print(f"{'='*80}\n")
        
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
            print(f"\n{'='*60}\nFOLD {fold_idx}/{n_splits}\n{'='*60}")

            real_train_fold = real_data.iloc[train_idx].reset_index(drop=True)
            real_test_fold  = real_data.iloc[test_idx].reset_index(drop=True)

            
            if syn_data is None:
                hybrid_train_fold = real_train_fold
            elif not augmentation:
                hybrid_train_fold = dataset_mixing.get_hybrid_data_stratified(
                    real_data=real_train_fold,
                    syn_data=syn_data,
                    syn_data_percentage=syn_percentage,
                    seed=1000+fold_idx,
                )
            else:
                hybrid_train_fold = dataset_mixing.get_hybrid_data_augmentation(
                    real_data=real_train_fold,
                    syn_data=syn_data,
                    syn_data_percentage=syn_percentage,
                    stratify=True,
                    seed=1000+fold_idx
                )

            fold_dir = os.path.join(out_dir, f"fold_{fold_idx}")
            ensure_dir(fold_dir)

            out = self.run_experiment(
                train_data=hybrid_train_fold,
                test_data=real_test_fold,
                out_dir=fold_dir,
                exp_name=f"{exp_name}_fold_{fold_idx}",
            )
            
            # Extract metrics for aggregation
            base_metrics = out["metrics"]["metrics"]          # dict from evaluate_model
            thr_metrics = out["metrics_thr"]["metrics"]       # dict from evaluate_with_threshold

            fold_row = {
                "fold": fold_idx,
                "threshold": out["threshold"],
                **base_metrics,
                **thr_metrics,
            }
            cv_metrics.append(fold_row)

            print(f"Fold {fold_idx} done. AUC={base_metrics.get('auc', None)}")

        # Aggregate + save
        metrics_df = pd.DataFrame(cv_metrics)
        metrics_path = os.path.join(out_dir, f"{exp_name}_all_folds_metrics.csv")
        metrics_df.to_csv(metrics_path, index=False)

        # Summary stats (mean/std) for numeric columns
        numeric_cols = [c for c in metrics_df.columns if c not in ["fold"] and metrics_df[c].dtype != "object"]
        summary = []
        for c in numeric_cols:
            summary.append({
                "metric": c,
                "mean": float(metrics_df[c].mean()),
                "std": float(metrics_df[c].std()),
                "min": float(metrics_df[c].min()),
                "max": float(metrics_df[c].max()),
            })
        summary_df = pd.DataFrame(summary).sort_values("metric")
        summary_path = os.path.join(out_dir, f"{exp_name}_summary_metrics.csv")
        summary_df.to_csv(summary_path, index=False)

        print(f"\n{'='*80}")
        print("CROSS-VALIDATION COMPLETE")
        print(f"Saved: {metrics_path}")
        print(f"Saved: {summary_path}")
        print(f"{'='*80}\n")

        return {
            "metrics_df": metrics_df,
            "summary_df": summary_df,
        }
    
    
    def _save_evaluation_metrics(self, metrics, metrics_thr, threshold, exp_name, out_dir):
        """ Save metrics to CSV """
        evaluation = pd.concat([pd.DataFrame([metrics["metrics"]]), pd.DataFrame([metrics_thr["metrics"]])], axis=1)
        evaluation["threshold"] = threshold
        evaluation.insert(0, "run_name", exp_name)

        csv_path = Path(out_dir).parent / "evaluation_metrics.csv"
        evaluation.to_csv(csv_path,
                        mode='a',
                        header = not os.path.exists(csv_path),
                        index=False)



def run_stratified_experiments(syn_data, real_data, result_path, data_name):
    """
    Run complete list of experiments comparing synthetic and real data.
    
    Experiments:
    1. Train on synthetic, test on real
    2. Train on hybrid (constant size), test on real
    3. Train on hybrid (augmented), test on real
    """
    
    syn_data = postprocessing.postprocess_for_utility(syn_data)

    # # Configuration
    # data_config = DataConfig()
    # exp_config = ExperimentConfig()

    result_path = result_path + data_name + "/"
    percentages = [0.1, 0.3, 0.5, 0.7]

    runner = ExperimentRunner()

    exp_name = "exp1_" + data_name
    n_classes = syn_data['in_hospital_death'].nunique(dropna=True)
    if n_classes > 1:
        print("\n" + "="*80)
        print("Experiment 1: \n Training data: Synthetic data.\n Test data: real test data.")
        print("="*80 + "\n")
        runner.run_stratified_kfold(
            real_data=real_data,
            syn_data=syn_data,
            syn_percentage=1,
            n_splits=5,
            out_dir=result_path+exp_name+"/",
            exp_name=exp_name,
        )

    
    print("\n" + "="*80)
    print("Experiment 2: \n Training data: Hybrid data.\n Test data: real test data.")
    print("="*80 + "\n")

    for perc in percentages:
        exp_name = f"exp2_{data_name}_{int(perc*100)}perc"
        runner.run_stratified_kfold(
            real_data=real_data,
            syn_data=syn_data,
            syn_percentage=perc,
            n_splits=5,
            out_dir=result_path+exp_name+"/",
            exp_name=exp_name,
        )

    print("\n" + "="*80)
    print("Experiment 3: \n Training data: Hybrid data - Augmentation.\n Test data: real test data.")
    print("="*80 + "\n")

    for perc in percentages:
        exp_name = f"exp3_{data_name}_{int(perc*100)}perc"
        runner.run_stratified_kfold(
            real_data=real_data,
            syn_data=syn_data,
            syn_percentage=perc,
            n_splits=5,
            out_dir=result_path+exp_name+"/",
            exp_name=exp_name,
            augmentation=True
        )
        

def run_experiment_list(synthetic_data, train_data, test_data, result_path, data_name):
    """
    Run complete list of experiments comparing synthetic and real data.
    
    Experiments:
    1. Train on synthetic, test on real
    2. Train on hybrid (constant size), test on real
    3. Train on hybrid (augmented), test on real
    """
    
    synthetic_data = postprocessing.postprocess_for_utility(synthetic_data)

    # # Configuration
    # data_config = DataConfig()
    # exp_config = ExperimentConfig()

    result_path = result_path + data_name + "/"
    percentages = [0.1, 0.3, 0.5, 0.7]

    runner = ExperimentRunner()

    n_classes = synthetic_data['in_hospital_death'].nunique(dropna=True)
    if n_classes > 1:
        print("\n" + "="*80)
        print("Experiment 1: \n Training data: Synthetic data.\n Test data: real test data.")
        print("="*80 + "\n")
        
        exp_name = "exp1_" + data_name
        runner.run_experiment(
            train_data=synthetic_data, 
            test_data=test_data,
            out_dir=result_path+exp_name+"/",
            exp_name=exp_name,
        )
        
    print("\n" + "="*80)
    print("Experiment 2: \n Training data: Hybrid data.\n Test data: real test data.")
    print("="*80 + "\n")

    for perc in percentages:
        exp_name = f"exp2_{data_name}_{int(perc*100)}perc"

        hybrid_data = dataset_mixing.get_hybrid_data_constant_size(
            real_data=train_data,
            syn_data=synthetic_data,
            syn_data_percentage=perc
        )
        runner.run_experiment(
            train_data=hybrid_data, 
            test_data=test_data,
            out_dir=result_path+exp_name+"/",
            exp_name=exp_name,
        )

    print("\n" + "="*80)
    print("Experiment 3: \n Training data: Hybrid data - Augmentation.\n Test data: real test data.")
    print("="*80 + "\n")

    for perc in percentages:
        exp_name = f"exp3_{data_name}_{int(perc*100)}perc"

        hybrid_data = dataset_mixing.get_hybrid_data_augmentation(
            real_data=train_data,
            syn_data=synthetic_data,
            syn_data_percentage=perc
        )
        runner.run_experiment(
            train_data=hybrid_data, 
            test_data=test_data,
            out_dir=result_path+exp_name+"/",
            exp_name=exp_name,
        )