from sklearn.model_selection import GridSearchCV
import xgboost as xgb

class XGBoostModel:
    def __init__(self):
        self.model = None

    # Calculate scale_pos_weight for class imbalance
    @staticmethod
    def get_scale_pos_weight(y_train):
        scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
        return scale_pos_weight
    
    @staticmethod
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


    def train(self, x_train, y_train):
        self.model = xgb.XGBClassifier(
            n_estimators=150,
            max_depth=4, # paper 3 [3, 4]
            learning_rate=0.1,
            gamma=0.1, # paper 0.25 [0, 0.1]
            colsample_bytree=0.7,
            min_child_weight=5,
            subsample=0.7, # paper 0.5 [0.6, 0.8]
            scale_pos_weight=self.get_scale_pos_weight(y_train),
            eval_metric='auc',
            max_delta_step=1,
            reg_alpha=0.3,
            reg_lambda=2.0,
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