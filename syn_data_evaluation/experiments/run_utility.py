from dataclasses import dataclass
import glob
import os
import re
from pathlib import Path
from typing import List

import pandas as pd
from sklearn.metrics import roc_curve
from sklearn.model_selection import PredefinedSplit
from syn_data_evaluation.data import dataset_mixing, postprocessing
from syn_data_evaluation.data.preprocessing import DataPreprocessor
from syn_data_evaluation.evaluation.model_evaluator import ModelEvaluator
from syn_data_evaluation.visualization.explainability import ModelExplainer
from syn_data_evaluation.visualization.model_visualizer import ModelVisualizer
from syn_data_evaluation.models.xgboost import XGBoostModel
import numpy as np

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

    def run_experiment(self, train_data, test_data, out_dir="outputs", exp_name="experiment", threshold=None):
        ensure_dir(out_dir)
        filename = os.path.join(out_dir, exp_name)

        # Data preparation
        data = self.preprocessor.prepare_data(
            train_data=train_data,
            test_data=test_data,
            impute=self.exp_config.impute,
            categorical_cols=self.data_config.categorical_cols,
            normalize=self.exp_config.normalize,
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

        # ROC curve
        fpr, tpr, thresholds = roc_curve(y_test, metrics["y_pred_proba"])
        if threshold is not None:
            best_threshold = threshold
            idx = self.evaluator.find_threshold_idx(thresholds, best_threshold)
            print(f"Using provided threshold: {best_threshold:.3f}")
        else:
            # Find optimal threshold
            idx, best_threshold = self.evaluator.get_optimal_threshold(fpr, tpr, thresholds)
            print(f"Optimal Threshold: {best_threshold:.3f}") 

        # Plot the ROC curve
        self.visualizer.plot_roc_curve(
            fpr,
            tpr, 
            thresholds, 
            idx, 
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
       
    
    def _save_evaluation_metrics(self, metrics, metrics_thr, threshold, exp_name, out_dir):
        """ Save metrics to CSV """
        evaluation = pd.concat([pd.DataFrame([metrics["metrics"]]), pd.DataFrame([metrics_thr["metrics"]])], axis=1)
        evaluation["threshold"] = threshold
        evaluation.insert(0, "run_name", exp_name)

        csv_path = Path(out_dir).parent.parent / "evaluation_metrics.csv"
        evaluation.to_csv(csv_path,
                        mode='a',
                        header = not os.path.exists(csv_path),
                        index=False)


def save_folds_metrics(experiment_results, out_dir: str = "outputs", exp_name: str = "cv_experiment"):
    cv_metrics = []
    for fold_idx, experiment_result in enumerate(experiment_results, 1):
        # Extract metrics for aggregation
        base_metrics = experiment_result["metrics"]["metrics"]          # dict from evaluate_model
        thr_metrics = experiment_result["metrics_thr"]["metrics"]       # dict from evaluate_with_threshold

        fold_row = {
            "fold": fold_idx,
            "threshold": experiment_result["threshold"],
            **base_metrics,
            **thr_metrics,
        }
        # Storage for results
        cv_metrics.append(fold_row)

        print(f"Fold {fold_idx} done. AUC={base_metrics.get('auc', None)}")

    # Aggregate + save
    metrics_df = pd.DataFrame(cv_metrics)
    # metrics_path = os.path.join(out_dir, f"{exp_name}_all_folds_metrics.csv")
    # metrics_df.to_csv(metrics_path, index=False)

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
    print(f"Saved: {summary_path}")
    print(f"{'='*80}\n")

    return {
        "metrics_df": metrics_df,
        "summary_df": summary_df,
    }
    
def run_experiment_list(synthetic_data, train_data, test_data, result_path, data_name, threshold=None, fold_num=0, postprocess=True):
    """
    Run complete list of experiments comparing synthetic and real data.
    
    Experiments:
    1. Train on synthetic, test on real
    2. Train on hybrid (constant size, syn data with same proportion as real), test on real
    3. Train on hybrid (augmented, syn data with same proportion as real), test on real
    4. Train on hybrid (constant size, random syn data), test on real
    5. Train on hybrid (augmented, random syn data), test on real
    """
    if postprocess:
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
        if fold_num == 0:
            out_dir = result_path+exp_name+"/"
        else:
            out_dir = result_path+exp_name+f"/fold_{fold_num}/"
            exp_name = exp_name + f"_fold_{fold_num}"
        runner.run_experiment(
            train_data=synthetic_data, 
            test_data=test_data,
            out_dir=out_dir,
            exp_name=exp_name,
            threshold=threshold
        )
        
        print("\n" + "="*80)
        print("Experiment 2.1: \n Training data: Hybrid data.\n Test data: real test data.")
        print("="*80 + "\n")

        for perc in percentages:
            exp_name = f"exp2.1_{int(perc*100)}perc_{data_name}"
            if fold_num == 0:
                out_dir = result_path+exp_name+"/"
            else:
                out_dir = result_path+exp_name+f"/fold_{fold_num}/"
                exp_name = exp_name + f"_fold_{fold_num}"
            # catch error if syn_size is larger than available synthetic data
            try:
                hybrid_data = dataset_mixing.get_hybrid_data(
                    train_data=train_data,
                    syn_data=synthetic_data,
                    syn_pct=perc,
                    constant_size=True,
                    match_real=True
                )
                runner.run_experiment(
                    train_data=hybrid_data, 
                    test_data=test_data,
                    out_dir=out_dir,
                    exp_name=exp_name,
                    threshold=threshold
                )
            except ValueError as e:
                print(f"Error occurred while creating hybrid data for {exp_name}: {e}")
                continue


        print("\n" + "="*80)
        print("Experiment 3.1: \n Training data: Hybrid data - Augmentation.\n Test data: real test data.")
        print("="*80 + "\n")

        for perc in percentages:
            exp_name = f"exp3.1_{int(perc*100)}perc_{data_name}"
            if fold_num == 0:
                out_dir = result_path+exp_name+"/"
            else:
                out_dir = result_path+exp_name+f"/fold_{fold_num}/"
                exp_name = exp_name + f"_fold_{fold_num}"
            try:
                hybrid_data = dataset_mixing.get_hybrid_data(
                    train_data=train_data,
                    syn_data=synthetic_data,
                    syn_pct=perc,
                    constant_size=False,
                    match_real=True
                )
                runner.run_experiment(
                    train_data=hybrid_data, 
                    test_data=test_data,
                    out_dir=out_dir,
                    exp_name=exp_name,
                    threshold=threshold
                )
            except ValueError as e:
                print(f"Error occurred while creating hybrid data for {exp_name}: {e}")
                continue
            


    print("\n" + "="*80)
    print("Experiment 2.2: \n Training data: Hybrid data. Random selection.\n Test data: real test data.")
    print("="*80 + "\n")

    for perc in percentages:
        exp_name = f"exp2.2_{int(perc*100)}perc_{data_name}"
        if fold_num == 0:
            out_dir = result_path+exp_name+"/"
        else:
            out_dir = result_path+exp_name+f"/fold_{fold_num}/"
            exp_name = exp_name + f"_fold_{fold_num}"

        try:
            hybrid_data = dataset_mixing.get_hybrid_data(
                train_data=train_data,
                syn_data=synthetic_data,
                syn_pct=perc,
                constant_size=True,
                match_real=False
            )
            runner.run_experiment(
                train_data=hybrid_data, 
                test_data=test_data,
                out_dir=out_dir,
                exp_name=exp_name,
                threshold=threshold
            )
        except ValueError as e:
            print(f"Error occurred while creating hybrid data for {exp_name}: {e}")
            continue
        

    print("\n" + "="*80)
    print("Experiment 3.2: \n Training data: Hybrid data - Augmentation. Random selection.\n Test data: real test data.")
    print("="*80 + "\n")

    for perc in percentages:
        exp_name = f"exp3.2_{int(perc*100)}perc_{data_name}"
        if fold_num == 0:
            out_dir = result_path+exp_name+"/"
        else:
            out_dir = result_path+exp_name+f"/fold_{fold_num}/"
            exp_name = exp_name + f"_fold_{fold_num}"

        try:
            hybrid_data = dataset_mixing.get_hybrid_data(
                train_data=train_data,
                syn_data=synthetic_data,
                syn_pct=perc,
                constant_size=False,
                match_real=False
            )
            runner.run_experiment(
                train_data=hybrid_data, 
                test_data=test_data,
                out_dir=out_dir,
                exp_name=exp_name,
                threshold=threshold
            )
        except ValueError as e:
            print(f"Error occurred while creating hybrid data for {exp_name}: {e}")
            continue
        


def run_exp_syn(result_path, real_path, syn_path, real_file, syn_file, data_name, threshold=None, postprocess=True):
    syn_data = pd.read_csv(os.path.join(syn_path, syn_file))

    real_data = pd.read_csv(os.path.join(real_path, real_file)).drop(columns=["sofa", "subject_id"], errors="ignore")
    test_data = pd.read_csv(os.path.join(real_path, "icu_dka_test_data.csv")).drop(columns=["sofa", "subject_id"], errors="ignore")
    run_experiment_list(
            synthetic_data=syn_data,
            train_data=real_data, 
            test_data=test_data,
            result_path=result_path,
            data_name=data_name,
            threshold=threshold,
            postprocess=postprocess
    )
        

def run_exp_syn_folds(result_path, real_fold_path, syn_fold_path, real_pattern, syn_pattern, data_name, thresholds=[0.5, 0.5, 0.5, 0.5, 0.5], postprocess=True, test_file=None):
    sorted_files = sorted(glob.glob( os.path.join(syn_fold_path, syn_pattern)))
    for syn_file in sorted_files:
        fold_idx = int(re.search(r"fold_(\d+)", syn_file).group(1))

        real_file = os.path.join(real_fold_path, real_pattern.replace("*", f"{fold_idx}"))
        print("\n" + "="*80)
        print(f"Start {syn_file}")
        print("="*80 + "\n")
        syn_data = pd.read_csv(syn_file)
        real_data = pd.read_csv(real_file).drop(columns=["sofa", "subject_id"], errors="ignore")
        if test_file is None:
            test_data = pd.read_csv(os.path.join(real_fold_path, f"val_fold_{fold_idx}.csv")).drop(columns=["sofa", "subject_id"], errors="ignore")
        else: 
            test_data = test_file

        run_experiment_list(
                synthetic_data=syn_data,
                train_data=real_data, 
                test_data=test_data,
                result_path=result_path,
                data_name=data_name,
                threshold=thresholds[fold_idx-1],
                fold_num=fold_idx,
                postprocess=postprocess
        )
        print(f"Finished {syn_data}")

    # save_folds_metrics(results, out_dir=result_path, exp_name="real_data_folds")

def run_real_folds(fold_path, result_path, pattern,data_name):
    runner = ExperimentRunner()
    results = []
    for i in range(5):
        csv_file = os.path.join(fold_path, pattern.replace("*", f"{i+1}"))
        print("\n" + "="*80)
        print(f"Start {csv_file}")
        print("="*80 + "\n")
        train_data = pd.read_csv(csv_file).drop(columns=["sofa", "subject_id"], errors="ignore")
        test_data = pd.read_csv(os.path.join(fold_path, f"val_fold_{i+1}.csv")).drop(columns=["sofa", "subject_id"], errors="ignore")
        exp_name = f"real_fold_{i+1}"
        results.append(runner.run_experiment(
                train_data=train_data, 
                test_data=test_data,
                out_dir=result_path+data_name+f"/fold_{i+1}/",
                exp_name=exp_name
        ))
        print(f"Finished {csv_file}")

    save_folds_metrics(results, out_dir=result_path, exp_name="real_data_folds")

def run_real_train(result_path, data_path, threshold=None):
    runner = ExperimentRunner()

    train_data = pd.read_csv(os.path.join(data_path, "icu_dka_train_data.csv")).drop(columns=["sofa", "subject_id"], errors="ignore")
    test_data = pd.read_csv(os.path.join(data_path, "icu_dka_test_data.csv")).drop(columns=["sofa", "subject_id"], errors="ignore")
    data_name = f"real_train"
    runner.run_experiment(
            train_data=train_data, 
            test_data=test_data,
            out_dir=result_path+data_name+"/",
            exp_name=data_name,
            threshold=threshold
    )


def run_hyperparameters_tuning(fold_path):
    X_data, y_data, test_fold = dataset_mixing.build_dataset_from_folds(fold_path=fold_path)
    # ps = PredefinedSplit(test_fold)

    # print("Total samples:", len(test_fold))
    # print("Unique fold labels:", np.unique(test_fold))  # should include -1 and 0..4

    # for k, (train_idx, test_idx) in enumerate(ps.split()):
    #     print(f"\nSplit {k}:")
    #     print("  train size:", len(train_idx))
    #     print("  test size :", len(test_idx))
    #     print("  min/max test_fold in test:", test_fold[test_idx].min(), test_fold[test_idx].max())
    #     print("  unique test_fold in test:", np.unique(test_fold[test_idx]))
    #     print("  overlap(train,test):", len(np.intersect1d(train_idx, test_idx)))
        

    model_wrapper = XGBoostModel()
    best_params, best_score, grid = model_wrapper.find_best_params(X_data, y_data, test_fold)
 
if __name__ == "__main__":
    """ 
        Documentation:
            syn_fold_path = ""
            real_fold_path = ""
            data_path = ""
            result_path = ""
            ----- HYPERPARAMETERS TUNING -----
            run_hyperparameters_tuning(fold_path)

            ----- RUN REAL TRAIN -----
            pattern = f"train_fold_*.csv"
            data_name = "real"
            run_real_train(result_path, data_path, threshold=0.1526)

            ----- RUN REAL FOLDS -----
            run_real_folds(fold_path, result_path, pattern, data_name)

            ----- RUN SYNTHETIC -----
            real_file = f"icu_dka_train_data.csv"
            run_exp_syn(result_path, data_path, syn_path, real_file, syn_file, data_name, threshold = 0.1526)

            ----- RUN SYNTHETIC FOLDS -----    
            real_pattern = f"train_fold_*.csv"
            run_exp_syn_folds(result_path, real_fold_path, syn_fold_path, real_pattern, syn_pattern, data_name, thresholds=[0.0689, 0.1243, 0.0865, 0.1692, 0.3140])
        
    """
    print("Starting utility experiments...")
    experiment_list = [
        ("baseline", 'config_3_syn_data_fold_*.csv'),
    ]
    for data_name, syn_data in experiment_list:
        run_exp_syn_folds(
            result_path= "", 
            real_fold_path="",
            syn_fold_path="",
            real_pattern=f"train_fold_*.csv",
            syn_pattern=syn_data,
            data_name=data_name,
            thresholds=[0.0689, 0.1243, 0.0865, 0.1692, 0.3140]
        )


# Train XGBoost classifier
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
