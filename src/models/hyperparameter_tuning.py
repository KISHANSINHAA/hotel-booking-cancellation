import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

def hyperparameter_tuning(X_train, y_train, model_name="Random Forest", n_iter=10, cv=3):
    """
    Perform hyperparameter tuning for specified model with reduced search space
    
    Args:
        X_train (pd.DataFrame or np.array): Training features
        y_train (pd.Series or np.array): Training target
        model_name (str): Name of model to tune
        n_iter (int): Number of iterations for RandomizedSearchCV
        cv (int): Number of cross-validation folds
        
    Returns:
        dict: Dictionary containing best model and parameters
    """
    print(f"=== TUNING {model_name.upper()} HYPERPARAMETERS ===")
    
    # Define models and parameter grids with smaller search spaces
    if model_name == "Random Forest":
        model = RandomForestClassifier(random_state=42)
        param_dist = {
            'n_estimators': [25, 50, 75],  # Smaller range
            'max_depth': [5, 10, None],
            'min_samples_split': [5, 10, 15],
            'min_samples_leaf': [2, 5, 10],
            'max_features': ['sqrt', 'log2']
        }
    elif model_name == "Gradient Boosting":
        model = GradientBoostingClassifier(random_state=42)
        param_dist = {
            'n_estimators': [25, 50, 75],  # Smaller range
            'max_depth': [3, 5, 7],
            'learning_rate': [0.05, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0]
        }
    elif model_name == "Logistic Regression":
        model = LogisticRegression(random_state=42, max_iter=1000)
        param_dist = {
            'C': [0.1, 1.0, 10.0],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear']  # Compatible with both penalties
        }
    else:
        # Default to Random Forest if unknown model
        print(f"Warning: Unknown model '{model_name}'. Using Random Forest as default.")
        model = RandomForestClassifier(random_state=42)
        param_dist = {
            'n_estimators': [25, 50, 75],
            'max_depth': [5, 10, None],
            'min_samples_split': [5, 10, 15],
            'min_samples_leaf': [2, 5, 10],
            'max_features': ['sqrt', 'log2']
        }
    
    # Use RandomizedSearchCV with smaller parameter space to reduce computation time and model size
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=min(n_iter, 10),  # Limit iterations for smaller search
        cv=cv,
        scoring='f1_weighted',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    # Fit the random search
    random_search.fit(X_train, y_train)
    
    print(f"{model_name} tuning completed.")
    print(f"Best parameters: {random_search.best_params_}")
    print(f"Best cross-validation score: {random_search.best_score_:.4f}")
    
    return {
        'best_model': random_search.best_estimator_,
        'best_params': random_search.best_params_,
        'best_score': random_search.best_score_,
        'search_results': random_search.cv_results_
    }

def quick_hyperparameter_tuning(X_train, y_train, model_name="Random Forest"):
    """
    Quick hyperparameter tuning with minimal parameter space for faster execution and smaller models
    
    Args:
        X_train (pd.DataFrame or np.array): Training features
        y_train (pd.Series or np.array): Training target
        model_name (str): Name of model to tune
        
    Returns:
        dict: Dictionary containing best model and parameters
    """
    print(f"=== QUICK TUNING {model_name.upper()} HYPERPARAMETERS ===")
    
    # Define models and parameter grids with minimal search spaces
    if model_name == "Random Forest":
        model = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [25],  # Minimal value
            'max_depth': [5],
            'min_samples_split': [10],
            'min_samples_leaf': [5],
            'max_features': ['sqrt']
        }
    elif model_name == "Gradient Boosting":
        model = GradientBoostingClassifier(random_state=42)
        param_grid = {
            'n_estimators': [25],  # Minimal value
            'max_depth': [3],
            'learning_rate': [0.1],
            'subsample': [0.8]
        }
    elif model_name == "Logistic Regression":
        model = LogisticRegression(random_state=42, max_iter=1000)
        param_grid = {
            'C': [1.0],
            'penalty': ['l2'],
            'solver': ['liblinear']
        }
    else:
        # Default to Random Forest if unknown model
        print(f"Warning: Unknown model '{model_name}'. Using Random Forest as default.")
        model = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [25],
            'max_depth': [5],
            'min_samples_split': [10],
            'min_samples_leaf': [5],
            'max_features': ['sqrt']
        }
    
    # Use GridSearchCV with minimal parameter space for fastest execution
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=3,  # Minimal CV folds
        scoring='f1_weighted',
        n_jobs=1,  # Use single thread to reduce memory usage
        verbose=1
    )
    
    # Fit the grid search
    grid_search.fit(X_train, y_train)
    
    print(f"{model_name} quick tuning completed.")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    return {
        'best_model': grid_search.best_estimator_,
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'search_results': grid_search.cv_results_
    }