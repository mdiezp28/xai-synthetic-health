import pandas as pd
import os
import json
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from sdv.metadata import SingleTableMetadata
from sdmetrics.reports.single_table import DiagnosticReport, QualityReport
from sdmetrics.single_table import NewRowSynthesis
from sdmetrics.visualization import get_column_plot, get_column_pair_plot
from sdmetrics.single_column import TVComplement
from sdmetrics.single_table import (
    ContinuousKLDivergence,
    DiscreteKLDivergence,
    KSComplement,
    CSTest,
    CorrelationSimilarity,
    ContingencySimilarity
)

from syn_data_evaluation.data.clinical_rules import NON_NEGATIVE_COLUMNS, RANGE_CONSTRAINTS, BINARY_COLUMNS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FidelityEvaluator:
    """
    Comprehensive evaluator for synthetic data fidelity.
    
    Supports two evaluation modes:
    - Simple: Quick metrics saved to CSV
    - Full: Comprehensive evaluation with plots and detailed reports
    """
    
    def __init__(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame, 
                 metadata: SingleTableMetadata, verbose: bool = True):
        """
        Initialize the FidelityEvaluator.
        
        Args:
            real_data: Original real dataset
            synthetic_data: Generated synthetic dataset
            metadata: SDV metadata object
            verbose: Whether to print detailed logs
        """
        self.real_data = real_data.copy()
        self.synthetic_data = synthetic_data.copy()
        self.metadata = metadata
        self.verbose = verbose
        
        # Validate inputs
        self._validate_inputs()
    
    def _validate_inputs(self):
        """Validate that inputs are properly formatted."""
        if self.real_data.empty:
            raise ValueError("Real data is empty")
        if self.synthetic_data.empty:
            raise ValueError("Synthetic data is empty")
        
        # Check column alignment
        real_cols = set(self.real_data.columns)
        synth_cols = set(self.synthetic_data.columns)
        
        if real_cols != synth_cols:
            missing_in_synth = real_cols - synth_cols
            missing_in_real = synth_cols - real_cols
            
            if missing_in_synth:
                logger.warning(f"Columns in real but not synthetic: {missing_in_synth}")
            if missing_in_real:
                logger.warning(f"Columns in synthetic but not real: {missing_in_real}")
    
    def simple_evaluation(self, experiment_name: str, results_csv: str = "experiments_results.csv") -> pd.DataFrame:
        """
        Run simple evaluation and append results to CSV.
        
        Args:
            experiment_name: Name of the experiment
            results_csv: Path to CSV file for storing results
            
        Returns:
            DataFrame with the evaluation row
        """
        logger.info(f"Running simple evaluation for: {experiment_name}")
        
        # Compute core metrics
        kl_categorical = self._safe_compute(DiscreteKLDivergence.compute, "KL Categorical")
        kl_continuous = self._safe_compute(ContinuousKLDivergence.compute, "KL Continuous")
        cs_categorical = self._safe_compute(CSTest.compute, "CS Test")
        ks_continuous = self._safe_compute(KSComplement.compute, "KS Test")
        
        # Correlation metrics
        corr_similarity = self._safe_compute(CorrelationSimilarity.compute, "Correlation Similarity")
        pearson_diff = 1 - corr_similarity if corr_similarity is not None else None
        
        # Categorical dependency
        cramers_v = self._safe_compute(ContingencySimilarity.compute, "Cramer's V")
        cramers_v_diff = 1 - cramers_v if cramers_v is not None else None
        
        # Semantic validity
        validity_results = self.evaluate_semantic_validity(self.synthetic_data)
        print(validity_results["column_results"].head())
        validity_rate = validity_results['row_summary']['row_validity_rate'].values[0]
        
        # Create results row
        results_row = {
            'Experiment_name': experiment_name,
            'KL_divergence_categorical': kl_categorical,
            'KL_divergence_continuous': kl_continuous,
            'CS_test_categorical': cs_categorical,
            'KS_test_continuous': ks_continuous,
            'Cramers_V_diff': cramers_v_diff,
            'Pearson_correlation_diff': pearson_diff,
            'Validity_rate': validity_rate
        }
        
        # Convert to DataFrame
        results_df = pd.DataFrame([results_row])
        
        # Append to CSV (or create if doesn't exist)
        if os.path.exists(results_csv):
            existing_df = pd.read_csv(results_csv)
            # Check if experiment already exists
            if experiment_name in existing_df['Experiment_name'].values:
                logger.warning(f"Experiment '{experiment_name}' already exists. Updating row.")
                existing_df = existing_df[existing_df['Experiment_name'] != experiment_name]
            combined_df = pd.concat([existing_df, results_df], ignore_index=True)
        else:
            combined_df = results_df
        
        combined_df.to_csv(results_csv, index=False)
        logger.info(f"Results saved to {results_csv}")
        
        # Print summary
        if self.verbose:
            self._print_simple_results(results_row)
        
        return results_df
    
    def _print_simple_results(self, results: Dict):
        """Print formatted simple evaluation results."""
        def fmt(x):
            return "N/A" if x is None else f"{x:.4f}"
        
        print("\n" + "="*70)
        print(f"SIMPLE EVALUATION RESULTS: {results['Experiment_name']}")
        print("="*70)
        print(f"KL Divergence (Categorical) [similarity]:     {fmt(results['KL_divergence_categorical'])}")
        print(f"KL Divergence (Continuous) [similarity]:      {fmt(results['KL_divergence_continuous'])}")
        print(f"CS Test (Categorical) [similarity]:           {fmt(results['CS_test_categorical'])}")
        print(f"KS Test (Continuous) [similarity]:            {fmt(results['KS_test_continuous'])}")
        print(f"Cramér's V (categorical dependency) [diff]:   {fmt(results['Cramers_V_diff'])}")
        print(f"Pearson Correlation (continuous dep.) [diff]: {fmt(results['Pearson_correlation_diff'])}")
        print(f"Validity Rate:                                {fmt(results['Validity_rate'])}")
        print("="*70 + "\n")
    
    def full_evaluation(self, experiment_name: str, exp_dir: str, 
                       plot_columns: Optional[List[str]] = None,
                       plot_pairs: Optional[List[Tuple[str, str]]] = None,
                       save_to_csv: bool = True,
                       results_csv: str = "experiments_results.csv") -> Dict:
        """
        Run full evaluation with detailed reports and visualizations.
        
        Args:
            experiment_name: Name of the experiment
            exp_dir: Directory to save all results
            plot_columns: List of columns to plot (None = all)
            plot_pairs: List of column pairs to plot
            save_to_csv: Whether to also save to the simple CSV
            results_csv: Path to CSV file for storing simple results
            
        Returns:
            Dictionary containing all evaluation results
        """
        logger.info(f"Running full evaluation for: {experiment_name}")
        
        # Create experiment directory
        exp_path = Path(exp_dir) / experiment_name
        exp_path.mkdir(parents=True, exist_ok=True)
        
        all_results = {
            'experiment_name': experiment_name,
            'exp_dir': str(exp_path)
        }
        
        # 1. Core statistical metrics
        logger.info("Computing statistical similarity metrics...")
        all_results['statistical_metrics'] = self.get_report_metrics()
        
        # 2. Diagnostic Report
        logger.info("Generating diagnostic report...")
        try:
            diagnostic_report = DiagnosticReport()
            diagnostic_results = diagnostic_report.generate(
                self.real_data, self.synthetic_data, self.metadata
            )
            all_results['diagnostic_report'] = diagnostic_results
            
            
            # Save diagnostic details
            for prop in ["Coverage", "Boundary", "Synthesis"]:
                try:
                    details = diagnostic_report.get_details(prop)
                    details.to_csv(exp_path / f"diagnostic_{prop.lower()}_details.csv", index=False)
                except Exception as e:
                    logger.warning(f"Could not save {prop} details: {e}")
        except Exception as e:
            logger.warning(f"Diagnostic report failed: {e}")
            all_results['diagnostic_report'] = None
        
        # 3. Quality Report
        logger.info("Generating quality report...")
        try:
            quality_report = QualityReport()
            quality_results = quality_report.generate(
                self.real_data, self.synthetic_data, self.metadata
            )
            all_results['quality_report'] = quality_results
            
            # Save quality details
            for prop, filename in [("Column Shapes", "distribution_similarity.csv"),
                                   ("Column Pair Trends", "correlation_similarity.csv")]:
                try:
                    details = quality_report.get_details(prop)
                    details.to_csv(exp_path / filename, index=False)
                    quality_report.get_visualization(prop).write_html(os.path.join(exp_path, f"quality_pair_trends.html"))
                except Exception as e:
                    logger.warning(f"Could not save {prop} details: {e}")
        except Exception as e:
            logger.warning(f"Quality report failed: {e}")
            all_results['quality_report'] = None
        
        # 4. New Row Synthesis
        logger.info("Computing new row synthesis...")
        try:
            nrs_metric = NewRowSynthesis()
            nrs_score = nrs_metric.compute(self.real_data, self.synthetic_data, self.metadata)
            all_results['new_row_synthesis'] = nrs_score
            with open(exp_path / "new_row_synthesis.txt", 'w') as f:
                f.write(f"New Row Synthesis Score: {nrs_score:.4f}\n")
        except Exception as e:
            logger.warning(f"New row synthesis failed: {e}")
            all_results['new_row_synthesis'] = None
        
        # 5. Column-level metrics
        logger.info("Computing column-level metrics...")
        tv_scores = {}
        for column in self.real_data.columns:
            try:
                score = TVComplement.compute(
                    self.real_data[column],
                    self.synthetic_data[column]
                )
                tv_scores[column] = score
            except Exception as e:
                logger.warning(f"Failed to compute TVComplement for {column}: {str(e)}")
                tv_scores[column] = None

        all_results['tv_complement'] = tv_scores
        
        try:
            continuous_cols = [col for col, col_meta in self.metadata['columns'].items()
                              if col_meta.get('sdtype') == 'numerical']
        except Exception as e:
            logger.warning(f"Could not identify continuous columns: {e}. Using all columns.")
            continuous_cols = list(self.real_data.columns)

        """Compute a metric for each column."""
        ks_scores = {}
        for column in continuous_cols:
            try:
                score = KSComplement.compute(
                    self.real_data,
                    self.synthetic_data,
                    self.metadata
                )
                ks_scores[column] = score
            except Exception as e:
                logger.warning(f"Failed to compute KSComplement for {column}: {str(e)}")
                ks_scores[column] = None

        all_results['ks_complement'] = ks_scores
        
        # Save column metrics
        pd.DataFrame({
            'column': list(tv_scores.keys()),
            'tv_complement': list(tv_scores.values())
        }).to_csv(exp_path / "tv_complement_scores.csv", index=False)
        
        if ks_scores:
            pd.DataFrame({
                'column': list(ks_scores.keys()),
                'ks_complement': list(ks_scores.values())
            }).to_csv(exp_path / "ks_complement_scores.csv", index=False)
        
        # 6. Semantic Validity
        logger.info("Evaluating semantic validity...")
        validity_results = self.evaluate_semantic_validity(self.synthetic_data)
        all_results['semantic_validity'] = validity_results
        
        # Save validity results
        if not validity_results['column_results'].empty:
            validity_results['column_results'].to_csv(
                exp_path / "semantic_violations_by_column.csv"
            )
        validity_results['row_summary'].to_csv(
            exp_path / "semantic_violations_summary.csv", index=False
        )
        
        # 7. Generate visualizations
        logger.info("Generating visualizations...")
        plots_dir = exp_path / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Determine which columns to plot
        if plot_columns is None:
            # Default: plot up to 10 columns
            plot_columns = list(self.real_data.columns)[:10]
            if len(self.real_data.columns) > 10:
                logger.info(f"Plotting first 10 columns. Specify plot_columns to customize.")
        
        for column in plot_columns:
            if column in self.real_data.columns:
                try:
                    self.save_column_plot(str(plots_dir), column)
                except Exception as e:
                    logger.warning(f"Failed to plot {column}: {e}")
        
        # Plot column pairs if specified
        if plot_pairs:
            for col1, col2 in plot_pairs:
                try:
                    self.save_column_pair_plot(str(plots_dir), [col1, col2])
                except Exception as e:
                    logger.warning(f"Failed to plot {col1}-{col2}: {e}")
        
        # 8. Save comprehensive summary
        logger.info("Saving comprehensive summary...")
        self._save_full_summary(all_results, exp_path / "full_evaluation_summary.json")
        
        # 9. Optionally save to simple CSV as well
        if save_to_csv:
            logger.info("Saving to simple results CSV...")
            self.simple_evaluation(experiment_name, results_csv)
        
        logger.info(f"Full evaluation complete! Results saved to {exp_path}")
        return all_results
    
    def get_report_metrics(self) -> Dict[str, Optional[float]]:
        """
        Compute statistical similarity metrics.
        
        Returns:
            Dictionary of metric scores
        """
        results = {}
        
        # KL Divergence for continuous variables
        results["kl_continuous"] = self._safe_compute(
            ContinuousKLDivergence.compute,
            "Continuous KL Divergence"
        )
        
        # KL Divergence for categorical variables
        results["kl_categorical"] = self._safe_compute(
            DiscreteKLDivergence.compute,
            "Discrete KL Divergence"
        )
        
        # KS Test for continuous marginals
        results["ks_continuous"] = self._safe_compute(
            KSComplement.compute,
            "KS Complement"
        )
        
        # Chi-square test for categorical marginals
        results["cs_categorical"] = self._safe_compute(
            CSTest.compute,
            "Chi-Square Test"
        )
        
        # Pearson correlation similarity
        corr_score = self._safe_compute(
            CorrelationSimilarity.compute,
            "Correlation Similarity"
        )
        results["corr_continuous"] = corr_score
        results["pearson_error"] = 1 - corr_score if corr_score is not None else None
        
        # Categorical dependency similarity
        results["categorical_dependency"] = self._safe_compute(
            ContingencySimilarity.compute,
            "Contingency Similarity"
        )
        
        if self.verbose:
            self._print_report_metrics(results)
        
        return results
    
    def _safe_compute(self, compute_func, metric_name: str) -> Optional[float]:
        """
        Safely compute a metric with error handling.
        
        Args:
            compute_func: The metric computation function
            metric_name: Name of the metric for logging
            
        Returns:
            Computed score or None if computation fails
        """
        try:
            score = compute_func(
                real_data=self.real_data,
                synthetic_data=self.synthetic_data,
                metadata=self.metadata
            )
            return score
        except Exception as e:
            logger.warning(f"{metric_name} computation failed: {str(e)}")
            return None
    
    def _print_report_metrics(self, results: Dict[str, Optional[float]]):
        """Print formatted metric results."""
        def fmt(x):
            return "N/A" if x is None else f"{x:.4f}"
        
        print("\n" + "="*60)
        print("STATISTICAL SIMILARITY METRICS")
        print("="*60)
        print(f"KL Divergence (Categorical):        {fmt(results['kl_categorical'])}")
        print(f"KL Divergence (Continuous):         {fmt(results['kl_continuous'])}")
        print(f"Chi-Square Test (Categorical):      {fmt(results['cs_categorical'])}")
        print(f"KS Test (Continuous):               {fmt(results['ks_continuous'])}")
        print(f"Categorical Dependency:             {fmt(results['categorical_dependency'])}")
        print(f"Pearson Correlation Error:          {fmt(results['pearson_error'])}")
        print("="*60 + "\n")
    
    
    def save_column_plot(self, exp_dir: str, column_name: str):
        """Save individual column distribution plot."""
        fig = get_column_plot(
            real_data=self.real_data,
            synthetic_data=self.synthetic_data,
            column_name=column_name
        )
        filepath = os.path.join(exp_dir, f"column_plot_{column_name}.html")
        fig.write_html(filepath)
        logger.debug(f"Saved column plot for {column_name}")
    
    def save_column_pair_plot(self, exp_dir: str, column_names: List[str]):
        """Save column pair relationship plot."""
        fig = get_column_pair_plot(
            real_data=self.real_data,
            synthetic_data=self.synthetic_data,
            column_names=column_names
        )
        filepath = os.path.join(exp_dir, f"column_plot_{column_names[0]}_{column_names[1]}.html")
        fig.write_html(filepath)
        logger.debug(f"Saved column pair plot for {column_names[0]}-{column_names[1]}")
    
    @staticmethod
    def check_non_negative(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Check that specified columns have non-negative values."""
        violations = {}
        for col in columns:
            if col in df.columns:
                violations[col] = df[col] < 0
        return pd.DataFrame(violations) if violations else pd.DataFrame(index=df.index)
    
    @staticmethod
    def check_ranges(df: pd.DataFrame, ranges: Dict[str, Tuple[Optional[float], Optional[float]]]) -> pd.DataFrame:
        """Check that columns are within specified ranges."""
        violations = {}
        for col, (min_val, max_val) in ranges.items():
            if col not in df.columns:
                continue
            series = df[col]
            mask = pd.Series(False, index=df.index)
            if min_val is not None:
                mask |= series < min_val
            if max_val is not None:
                mask |= series > max_val
            violations[col] = mask
        return pd.DataFrame(violations) if violations else pd.DataFrame(index=df.index)
    
    @staticmethod
    def check_binary(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Check that specified columns contain only binary values {0, 1}."""
        violations = {}
        for col in columns:
            if col in df.columns:
                violations[col] = ~df[col].isin([0, 1])
        return pd.DataFrame(violations) if violations else pd.DataFrame(index=df.index)
    
    @staticmethod
    def evaluate_semantic_validity(df: pd.DataFrame) -> Dict:
        """
        Evaluate semantic validity and clinical constraints.
        
        Returns:
            Dictionary with violation statistics
        """
        # Gather violations
        nn_viol = FidelityEvaluator.check_non_negative(df, NON_NEGATIVE_COLUMNS)
        range_viol = FidelityEvaluator.check_ranges(df, RANGE_CONSTRAINTS)
        bin_viol = FidelityEvaluator.check_binary(df, BINARY_COLUMNS)
        
        # Combine all violations
        all_viol = pd.concat([nn_viol, range_viol, bin_viol], axis=1)
        
        if all_viol.empty:
            return {
                "column_results": pd.DataFrame(columns=["violation_rate"]),
                "row_summary": pd.DataFrame([{
                    "n_rows": len(df),
                    "rows_with_any_violation": 0,
                    "row_violation_rate": 0.0,
                    "rows_fully_valid": len(df),
                    "row_validity_rate": 1.0,
                }]),
                "violations_matrix": all_viol,
            }
        
        all_viol = all_viol.fillna(False).astype(bool)
        
        # Column-level statistics
        col_violation_rate = all_viol.mean(axis=0)
        column_results = col_violation_rate.to_frame(name="violation_rate")
        column_results.index.name = "column"
        column_results = column_results.sort_values("violation_rate", ascending=False)
        
        # Row-level statistics
        any_viol = all_viol.any(axis=1)
        row_summary = pd.DataFrame([{
            "n_rows": int(len(df)),
            "rows_with_any_violation": int(any_viol.sum()),
            "row_violation_rate": float(any_viol.mean()),
            "rows_fully_valid": int((~any_viol).sum()),
            "row_validity_rate": float((~any_viol).mean()),
        }])
        
        return {
            "column_results": column_results,
            "row_summary": row_summary,
            "violations_matrix": all_viol,
        }
    
    @staticmethod
    def _save_full_summary(results: Dict, filepath: Path):
        """Save comprehensive evaluation summary to JSON file."""
        summary = {
            'experiment_name': results.get('experiment_name'),
            'exp_dir': results.get('exp_dir'),
        }
        
        # Add statistical metrics
        if 'statistical_metrics' in results:
            summary['statistical_metrics'] = results['statistical_metrics']
        
        # Add scores from reports
        if results.get('diagnostic_report'):
            try:
                summary['diagnostic_score'] = results['diagnostic_report'].get_score()
            except:
                pass
        
        if results.get('quality_report'):
            try:
                summary['quality_score'] = results['quality_report'].get_score()
            except:
                pass
        
        if 'new_row_synthesis' in results:
            summary['new_row_synthesis'] = results['new_row_synthesis']
        
        # Add validity summary
        if 'semantic_validity' in results:
            validity = results['semantic_validity']['row_summary'].iloc[0].to_dict()
            summary['semantic_validity'] = validity
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved comprehensive summary to {filepath}")
    



# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def run_simple_evaluation(real_data: pd.DataFrame, 
                         synthetic_data: pd.DataFrame,
                         metadata: SingleTableMetadata,
                         experiment_name: str,
                         results_csv: str = "experiments_results.csv") -> pd.DataFrame:
    """
    Convenience function for simple evaluation.
    
    Args:
        real_data: Real dataset
        synthetic_data: Synthetic dataset
        metadata: SDV metadata
        experiment_name: Name for this experiment
        results_csv: Path to results CSV file
        
    Returns:
        DataFrame with evaluation results
    """
    evaluator = FidelityEvaluator(real_data, synthetic_data, metadata, verbose=True)
    return evaluator.simple_evaluation(experiment_name, results_csv)


def run_full_evaluation(real_data: pd.DataFrame,
                       synthetic_data: pd.DataFrame,
                       metadata: SingleTableMetadata,
                       experiment_name: str,
                       exp_dir: str = "./experiments",
                       plot_columns: Optional[List[str]] = None,
                       plot_pairs: Optional[List[Tuple[str, str]]] = None,
                       save_to_csv: bool = True,
                       results_csv: str = "experiments_results.csv") -> Dict:
    """
    Convenience function for full evaluation.
    
    Args:
        real_data: Real dataset
        synthetic_data: Synthetic dataset
        metadata: SDV metadata
        experiment_name: Name for this experiment
        exp_dir: Base directory for experiments
        plot_columns: Columns to plot (None = first 10)
        plot_pairs: Column pairs to plot
        save_to_csv: Also save to simple CSV
        results_csv: Path to results CSV file
        
    Returns:
        Dictionary with all evaluation results
    """
    evaluator = FidelityEvaluator(real_data, synthetic_data, metadata, verbose=True)
    return evaluator.full_evaluation(
        experiment_name=experiment_name,
        exp_dir=exp_dir,
        plot_columns=plot_columns,
        plot_pairs=plot_pairs,
        save_to_csv=save_to_csv,
        results_csv=results_csv
    )

