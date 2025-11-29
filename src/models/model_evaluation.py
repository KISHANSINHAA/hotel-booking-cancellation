import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, roc_curve
import warnings
warnings.filterwarnings('ignore')

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance
    
    Args:
        model: Trained model
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test target
        
    Returns:
        dict: Evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }

def plot_confusion_matrix(y_test, y_pred, model_name, save_path=None):
    """
    Plot confusion matrix
    
    Args:
        y_test (pd.Series or np.array): True labels
        y_pred (pd.Series or np.array): Predicted labels
        model_name (str): Name of the model
        save_path (str): Path to save the plot
        
    Returns:
        plt: Matplotlib plot object
    """
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Canceled', 'Not Canceled'],
                yticklabels=['Canceled', 'Not Canceled'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt

def plot_roc_curve(y_test, y_pred_proba, model_name, save_path=None):
    """
    Plot ROC curve
    
    Args:
        y_test (pd.Series or np.array): True labels
        y_pred_proba (pd.Series or np.array): Prediction probabilities
        model_name (str): Name of the model
        save_path (str): Path to save the plot
        
    Returns:
        plt: Matplotlib plot object
    """
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt

def get_feature_importance(model, feature_names, top_n=10):
    """
    Get feature importance from model
    
    Args:
        model: Trained model
        feature_names (list): List of feature names
        top_n (int): Number of top features to return
        
    Returns:
        pd.DataFrame: Feature importance dataframe
    """
    # Check if model has feature_importances_ attribute
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    # For Logistic Regression, use coefficients
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_[0])
    else:
        # Return empty dataframe if no importance attribute
        return pd.DataFrame(columns=['feature', 'importance'])
    
    # Create dataframe
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False).head(top_n)
    
    return feature_importance

def plot_feature_importance(feature_importance, model_name, save_path=None):
    """
    Plot feature importance
    
    Args:
        feature_importance (pd.DataFrame): Feature importance dataframe
        model_name (str): Name of the model
        save_path (str): Path to save the plot
        
    Returns:
        plt: Matplotlib plot object
    """
    if feature_importance.empty:
        return None
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.barplot(data=feature_importance, x='importance', y='feature', palette='viridis')
    plt.title(f'Top Feature Importance - {model_name}')
    plt.xlabel('Importance')
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt

def comprehensive_model_evaluation(model, X_test, y_test, feature_names=None, 
                                 model_name="Model", class_names=None, save_path=None):
    """
    Comprehensive model evaluation with visualizations
    
    Args:
        model: Trained model
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test target
        feature_names (list): List of feature names
        model_name (str): Name of the model
        class_names (list): List of class names
        save_path (str): Path to save visualizations
        
    Returns:
        dict: Comprehensive evaluation results
    """
    print(f"=== STARTING COMPREHENSIVE MODEL EVALUATION FOR {model_name.upper()} ===")
    
    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Print metrics
    print(f"{model_name} Performance Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1_score']:.4f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix:")
    print(f"  {cm}")
    
    # Classification Report
    print(f"Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # Generate visualizations if save_path is provided
    if save_path:
        import os
        
        # Create graphs directory if it doesn't exist
        graphs_dir = os.path.join(save_path, "..", "graphs")
        os.makedirs(graphs_dir, exist_ok=True)
        
        # Save confusion matrix
        cm_path = os.path.join(graphs_dir, f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png")
        plot_confusion_matrix(y_test, y_pred, model_name, cm_path)
        plt.close()
        print(f"Confusion matrix saved to: {cm_path}")
        
        # Save ROC curve
        roc_path = os.path.join(graphs_dir, f"{model_name.lower().replace(' ', '_')}_roc_curve.png")
        plot_roc_curve(y_test, y_pred_proba, model_name, roc_path)
        plt.close()
        print(f"ROC curve saved to: {roc_path}")
        
        # Save feature importance if feature names provided
        if feature_names:
            feature_importance = get_feature_importance(model, feature_names)
            if not feature_importance.empty:
                fi_path = os.path.join(graphs_dir, f"{model_name.lower().replace(' ', '_')}_feature_importance.png")
                plot_feature_importance(feature_importance, model_name, fi_path)
                plt.close()
                print(f"Feature importance plot saved to: {fi_path}")
                
                # Print top features
                print(f"\n=== BUSINESS INTERPRETATION OF IMPORTANT FEATURES ===")
                top_features = feature_importance.head(10)
                interpretations = {
                    'lead_time': "Longer lead times may indicate more uncertainty in plans",
                    'avg_price_per_room': "Room pricing may affect cancellation decisions",
                    'no_of_special_requests': "Special requests may indicate higher engagement",
                    'adr_per_person': "Room pricing may affect cancellation decisions",
                    'special_requests_per_guest': "Party size may influence booking commitment",
                    'arrival_month': "Seasonal factors may influence booking behavior",
                    'is_weekend_booking': "Booking patterns differ between weekdays and weekends",
                    'total_stay_nights': "Length of stay may impact cancellation likelihood",
                    'is_peak_season': "Peak season bookings may have different cancellation patterns",
                    'previous_cancellation_rate': "Past cancellation behavior influences future cancellations"
                }
                
                for i, (idx, row) in enumerate(top_features.iterrows(), 1):
                    feature = row['feature']
                    importance = row['importance']
                    interpretation = interpretations.get(feature, "Feature importance for prediction")
                    print(f" {i}. {feature}: {importance:.4f}")
                    print(f"    -> {interpretation}")
    
    print(f"=== COMPREHENSIVE EVALUATION COMPLETED FOR {model_name.upper()} ===")
    
    # Add predictions to metrics
    metrics['confusion_matrix'] = cm
    metrics['predictions'] = y_pred
    metrics['prediction_probabilities'] = y_pred_proba
    
    return metrics