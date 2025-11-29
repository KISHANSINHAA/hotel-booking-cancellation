import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets
    
    Args:
        X (pd.DataFrame or np.array): Feature matrix
        y (pd.Series or np.array): Target variable
        test_size (float): Proportion of data for testing
        random_state (int): Random state for reproducibility
        
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    print("=== SPLITTING DATA ===")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Testing set size: {X_test.shape[0]} samples")
    print(f"Feature count: {X_train.shape[1]}")
    
    return X_train, X_test, y_train, y_test

def train_logistic_regression(X_train, y_train, random_state=42):
    """
    Train Logistic Regression model
    
    Args:
        X_train (pd.DataFrame or np.array): Training features
        y_train (pd.Series or np.array): Training target
        random_state (int): Random state for reproducibility
        
    Returns:
        LogisticRegression: Trained model
    """
    print("=== TRAINING LOGISTIC REGRESSION ===")
    
    # Create and train model with optimized parameters for smaller size
    lr_model = LogisticRegression(random_state=random_state, max_iter=1000, C=1.0)
    lr_model.fit(X_train, y_train)
    
    print("Logistic Regression model trained successfully")
    return lr_model

def train_random_forest(X_train, y_train, random_state=42):
    """
    Train Random Forest model with optimized parameters for smaller size
    
    Args:
        X_train (pd.DataFrame or np.array): Training features
        y_train (pd.Series or np.array): Training target
        random_state (int): Random state for reproducibility
        
    Returns:
        RandomForestClassifier: Trained model
    """
    print("=== TRAINING RANDOM FOREST ===")
    
    # Create and train model with reduced complexity for smaller file size
    rf_model = RandomForestClassifier(
        n_estimators=50,  # Reduced from default
        max_depth=10,     # Limit depth to reduce model size
        min_samples_split=10,  # Increase to reduce overfitting and size
        min_samples_leaf=5,    # Increase to reduce overfitting and size
        max_features='sqrt',   # Reduce features considered at each split
        random_state=random_state
    )
    rf_model.fit(X_train, y_train)
    
    print("Random Forest model trained successfully")
    return rf_model

def train_gradient_boosting(X_train, y_train, random_state=42):
    """
    Train Gradient Boosting model with optimized parameters for smaller size
    
    Args:
        X_train (pd.DataFrame or np.array): Training features
        y_train (pd.Series or np.array): Training target
        random_state (int): Random state for reproducibility
        
    Returns:
        GradientBoostingClassifier: Trained model
    """
    print("=== TRAINING GRADIENT BOOSTING ===")
    
    # Create and train model with reduced complexity for smaller file size
    gb_model = GradientBoostingClassifier(
        n_estimators=50,      # Reduced from default
        max_depth=5,          # Limit depth to reduce model size
        learning_rate=0.1,    # Standard learning rate
        subsample=0.8,        # Use subset of samples to reduce overfitting
        random_state=random_state
    )
    gb_model.fit(X_train, y_train)
    
    print("Gradient Boosting model trained successfully")
    return gb_model

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluate model performance
    
    Args:
        model: Trained model
        X_test (pd.DataFrame or np.array): Test features
        y_test (pd.Series or np.array): Test target
        model_name (str): Name of the model for reporting
        
    Returns:
        dict: Evaluation metrics
    """
    print(f"=== EVALUATING {model_name.upper()} ===")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
    
    # Print results
    print(f"{model_name} Performance:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    if roc_auc:
        print(f"  ROC-AUC:   {roc_auc:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"  Confusion Matrix:")
    print(f"    {cm}")
    
    # Classification Report
    print(f"  Classification Report:")
    print(classification_report(y_test, y_pred))
    
    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'predictions': y_pred,
        'prediction_probabilities': y_pred_proba
    }

def compare_models(models_dict, X_test, y_test):
    """
    Compare multiple models
    
    Args:
        models_dict (dict): Dictionary of trained models
        X_test (pd.DataFrame or np.array): Test features
        y_test (pd.Series or np.array): Test target
        
    Returns:
        pd.DataFrame: Comparison results
    """
    print("=== COMPARING MODELS ===")
    
    results = []
    
    for model_name, model in models_dict.items():
        metrics = evaluate_model(model, X_test, y_test, model_name)
        results.append(metrics)
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame([
        {
            'Model': result['model_name'],
            'Accuracy': result['accuracy'],
            'Precision': result['precision'],
            'Recall': result['recall'],
            'F1-Score': result['f1_score'],
            'ROC-AUC': result['roc_auc']
        }
        for result in results if result['roc_auc'] is not None
    ])
    
    # If ROC-AUC is not available for all models, exclude it
    if comparison_df['ROC-AUC'].isna().all():
        comparison_df = comparison_df.drop('ROC-AUC', axis=1)
    
    # Sort by F1-Score
    comparison_df = comparison_df.sort_values('F1-Score', ascending=False)
    
    print("\nModel Comparison (sorted by F1-Score):")
    print(comparison_df.to_string(index=False))
    
    return comparison_df, results

def train_and_compare_models(X_train, y_train, X_test, y_test):
    """
    Complete model training and comparison pipeline
    
    Args:
        X_train (pd.DataFrame or np.array): Training features
        y_train (pd.Series or np.array): Training target
        X_test (pd.DataFrame or np.array): Test features
        y_test (pd.Series or np.array): Test target
        
    Returns:
        dict: Dictionary containing trained models and comparison results
    """
    print("=== STARTING MODEL TRAINING AND COMPARISON PIPELINE ===")
    
    # Train models
    lr_model = train_logistic_regression(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)
    gb_model = train_gradient_boosting(X_train, y_train)
    
    # Store models in dictionary
    models = {
        'Logistic Regression': lr_model,
        'Random Forest': rf_model,
        'Gradient Boosting': gb_model
    }
    
    # Compare models
    comparison_df, evaluation_results = compare_models(models, X_test, y_test)
    
    # Identify best model
    best_model_name = comparison_df.iloc[0]['Model']
    best_model = models[best_model_name]
    
    print(f"\nBest Model: {best_model_name}")
    
    return {
        'models': models,
        'best_model': best_model,
        'best_model_name': best_model_name,
        'comparison_results': comparison_df,
        'evaluation_results': evaluation_results
    }

def get_all_trained_models(X_train, y_train):
    """
    Train all three models and return them
    
    Args:
        X_train (pd.DataFrame or np.array): Training features
        y_train (pd.Series or np.array): Training target
        
    Returns:
        dict: Dictionary containing all trained models
    """
    print("=== TRAINING ALL THREE MODELS ===")
    
    # Train all models
    lr_model = train_logistic_regression(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)
    gb_model = train_gradient_boosting(X_train, y_train)
    
    # Store models in dictionary
    models = {
        'Logistic Regression': lr_model,
        'Random Forest': rf_model,
        'Gradient Boosting': gb_model
    }
    
    print("All models trained successfully")
    return models