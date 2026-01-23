
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer

class ModelExplainer:

    @staticmethod
    def group_dummy_feature_importance(feat_imp, sep = "__"):
        importances = feat_imp.copy()

        # Detect dummy features by separator
        is_dummy = importances["feature"].str.contains(sep)

        # Extract original feature name for dummy columns, leave others unchanged
        importances["original_feature"] = importances["feature"].where(~is_dummy, importances["feature"].str.split(sep, n=1).str[0])

        # Group by original feature and sum importance
        grouped = importances.groupby("original_feature")["importance"].sum().sort_values(ascending=False).reset_index()
        grouped = grouped.rename(columns={"original_feature": "feature"})
        return grouped


    # XGBOOST
    @staticmethod
    def get_feat_importance(model, features):
        imp = model.feature_importances_
        # Create a DataFrame for importance
        feat_imp = pd.DataFrame({'feature': features, 'importance': imp})
        feat_imp = feat_imp.sort_values(by='importance', ascending=False)
        # print(feat_imp_df.head(20))  # Top 20 features
        return feat_imp

    # SHAP
    def generate_shap_explanations(self, model_xgb, x_train, filename):
        print("Generating SHAP explanations...")

        explainer = shap.TreeExplainer(model_xgb)
        shap_values = explainer(x_train)

        feature_importance_shap = pd.DataFrame({
            'feature': x_train.columns,
            'importance': np.abs(shap_values.values).mean(axis=0)
        }).sort_values(by='importance', ascending=False)
        
        # summary plot
        shap.summary_plot(shap_values, x_train, feature_names=x_train.columns, show=False)
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
        
        feature_importance_shap_grouped = self.group_dummy_feature_importance(feature_importance_shap)
        importances = feature_importance_shap_grouped.sort_values('importance', ascending=True)

        plt.figure(figsize=(10, 8))
        plt.barh(importances['feature'], importances['importance'], color='#ff0050', edgecolor='black')
        plt.xlabel('Mean |SHAP Value|')
        plt.title('Global Feature Importance (Grouped Categorical Features)')
        plt.tight_layout()
        plt.savefig(f'{filename}_mean_shap_grouped_.png', dpi=300, bbox_inches='tight')
        plt.close()

        return feature_importance_shap, feature_importance_shap_grouped

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

    # LIME 
    def generate_lime_explanations(model_xgb, X_train, filename, num_samples=5):
        """
        Generate LIME explanations for the first few samples in X_train.
        Saves explanation plots as PNG files.
        """
        print("Generating LIME explanations...")

        X_train_np = X_train.values
        feature_names = X_train.columns.tolist()
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

        print("LIME explanations generated successfully.")

