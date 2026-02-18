from matplotlib.pyplot import grid
from sklearn.model_selection import GridSearchCV, PredefinedSplit
import xgboost as xgb

class XGBoostModel:
    def __init__(self):
        self.model = None

    # Calculate scale_pos_weight for class imbalance
    @staticmethod
    def get_scale_pos_weight(y_train):
        scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
        return scale_pos_weight


    def find_best_params(self, x_train, y_train, test_fold):
        scale_pos_weight = self.get_scale_pos_weight(y_train[test_fold == -1])
        param_grid = {
            'n_estimators': [100, 150],
            'max_depth': [3, 4],
            'gamma': [0, 0.25],
            'colsample_bytree': [ 0.7],
            'subsample': [0.5],
            'reg_alpha': [0.1, 0.3],
            'reg_lambda': [1.0, 2.0],
            'max_delta_step': [0],
        }
        grid = GridSearchCV(
            estimator= xgb.XGBClassifier(
                learning_rate=0.05,
                min_child_weight=1,
                eval_metric='aucpr',
                scale_pos_weight=scale_pos_weight,
                # random_state=42,
                # tree_method='hist',  # Use histogram-based algorithm for faster training
            ), 
            param_grid=param_grid,
            scoring='recall',
            cv=PredefinedSplit(test_fold),
            n_jobs=-1,
            refit=False,
            verbose=2,
        )

        grid.fit(x_train, y_train)

        best_params = grid.best_params_
        best_score = grid.best_score_

        print("Best params:", best_params)
        print(f"Best mean CV score: {best_score}")

        return best_params, best_score, grid


    def train(self, x_train, y_train):
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=3, # paper 3 [3, 4]
            learning_rate=0.05,
            gamma=0, # paper 0.25 [0, 0.1]
            colsample_bytree=0.7,
            min_child_weight=1,
            subsample=0.5, # paper 0.5 [0.6, 0.8]
            scale_pos_weight=self.get_scale_pos_weight(y_train),
            eval_metric='aucpr',
            max_delta_step=0,
            reg_alpha=0.1,
            reg_lambda=1.0,
            # colsample_bytree=0.7,  # Only use 50% of features per tree
            # colsample_bylevel=0.8,  # Additional sampling at each level
        )
        self.model.fit(x_train, y_train)
        return self.model
        
    
    def predict(self, X):
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict probabilities."""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        return self.model.predict_proba(X)[:, 1]