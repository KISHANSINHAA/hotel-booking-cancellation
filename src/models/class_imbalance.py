import pandas as pd
import numpy as np
from collections import Counter
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

def analyze_class_distribution(y):
    """
    Analyze the distribution of classes in the target variable
    
    Args:
        y (pd.Series or np.array): Target variable
        
    Returns:
        dict: Class distribution information
    """
    print("=== ANALYZING CLASS DISTRIBUTION ===")
    
    class_counts = Counter(y)
    total_samples = len(y)
    
    print("Class distribution:")
    for class_label, count in class_counts.items():
        percentage = (count / total_samples) * 100
        print(f"  Class {class_label}: {count} samples ({percentage:.2f}%)")
    
    # Calculate imbalance ratio
    counts = list(class_counts.values())
    imbalance_ratio = max(counts) / min(counts) if min(counts) > 0 else float('inf')
    print(f"Imbalance ratio: {imbalance_ratio:.2f}:1")
    
    return {
        'class_counts': class_counts,
        'total_samples': total_samples,
        'imbalance_ratio': imbalance_ratio
    }

def apply_smote(X, y, random_state=42):
    """
    Apply SMOTE (Synthetic Minority Oversampling Technique) to handle class imbalance
    
    Args:
        X (pd.DataFrame or np.array): Feature matrix
        y (pd.Series or np.array): Target variable
        random_state (int): Random state for reproducibility
        
    Returns:
        tuple: Resampled X and y
    """
    print("=== APPLYING SMOTE ===")
    
    # Check if we have enough samples for SMOTE
    min_class_count = min(Counter(y).values())
    
    if min_class_count < 2:
        print("Warning: Not enough samples in minority class for SMOTE. Using random oversampling instead.")
        return apply_random_oversampling(X, y, random_state)
    
    # Adjust k_neighbors if necessary
    k_neighbors = min(5, min_class_count - 1) if min_class_count > 1 else 1
    
    try:
        smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        print(f"SMOTE applied successfully.")
        print(f"Original shape: {X.shape} -> Resampled shape: {X_resampled.shape}")
        
        # Show new distribution
        new_counts = Counter(y_resampled)
        print("New class distribution after SMOTE:")
        for class_label, count in new_counts.items():
            print(f"  Class {class_label}: {count} samples")
            
        return X_resampled, y_resampled
    except Exception as e:
        print(f"SMOTE failed: {str(e)}. Using random oversampling instead.")
        return apply_random_oversampling(X, y, random_state)

def apply_random_oversampling(X, y, random_state=42):
    """
    Apply random oversampling to handle class imbalance
    
    Args:
        X (pd.DataFrame or np.array): Feature matrix
        y (pd.Series or np.array): Target variable
        random_state (int): Random state for reproducibility
        
    Returns:
        tuple: Resampled X and y
    """
    print("=== APPLYING RANDOM OVERSAMPLING ===")
    
    # Convert to DataFrame if needed for easier handling
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    
    # Combine X and y for resampling
    df = pd.concat([X, y], axis=1)
    target_col = y.name if y.name else 'target'
    
    # Separate majority and minority classes
    class_counts = Counter(y)
    majority_class = max(class_counts, key=class_counts.get)
    minority_class = min(class_counts, key=class_counts.get)
    
    df_majority = df[df[target_col] == majority_class]
    df_minority = df[df[target_col] == minority_class]
    
    # Upsample minority class
    df_minority_upsampled = resample(df_minority,
                                     replace=True,
                                     n_samples=len(df_majority),
                                     random_state=random_state)
    
    # Combine majority class with upsampled minority class
    df_resampled = pd.concat([df_majority, df_minority_upsampled])
    
    # Shuffle the data
    df_resampled = df_resampled.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Separate X and y
    X_resampled = df_resampled.drop(target_col, axis=1)
    y_resampled = df_resampled[target_col]
    
    print(f"Random oversampling applied successfully.")
    print(f"Original shape: {X.shape} -> Resampled shape: {X_resampled.shape}")
    
    # Show new distribution
    new_counts = Counter(y_resampled)
    print("New class distribution after random oversampling:")
    for class_label, count in new_counts.items():
        print(f"  Class {class_label}: {count} samples")
    
    return X_resampled, y_resampled

def apply_undersampling(X, y, random_state=42):
    """
    Apply random undersampling to handle class imbalance
    
    Args:
        X (pd.DataFrame or np.array): Feature matrix
        y (pd.Series or np.array): Target variable
        random_state (int): Random state for reproducibility
        
    Returns:
        tuple: Resampled X and y
    """
    print("=== APPLYING RANDOM UNDERSAMPLING ===")
    
    # Convert to DataFrame if needed for easier handling
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    
    # Combine X and y for resampling
    df = pd.concat([X, y], axis=1)
    target_col = y.name if y.name else 'target'
    
    # Separate majority and minority classes
    class_counts = Counter(y)
    majority_class = max(class_counts, key=class_counts.get)
    minority_class = min(class_counts, key=class_counts.get)
    
    df_majority = df[df[target_col] == majority_class]
    df_minority = df[df[target_col] == minority_class]
    
    # Downsample majority class
    df_majority_downsampled = resample(df_majority,
                                       replace=False,
                                       n_samples=len(df_minority),
                                       random_state=random_state)
    
    # Combine minority class with downsampled majority class
    df_resampled = pd.concat([df_majority_downsampled, df_minority])
    
    # Shuffle the data
    df_resampled = df_resampled.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Separate X and y
    X_resampled = df_resampled.drop(target_col, axis=1)
    y_resampled = df_resampled[target_col]
    
    print(f"Random undersampling applied successfully.")
    print(f"Original shape: {X.shape} -> Resampled shape: {X_resampled.shape}")
    
    # Show new distribution
    new_counts = Counter(y_resampled)
    print("New class distribution after random undersampling:")
    for class_label, count in new_counts.items():
        print(f"  Class {class_label}: {count} samples")
    
    return X_resampled, y_resampled

def handle_class_imbalance(X, y, method='smote', random_state=42):
    """
    Handle class imbalance using specified method
    
    Args:
        X (pd.DataFrame or np.array): Feature matrix
        y (pd.Series or np.array): Target variable
        method (str): Method to use ('smote', 'oversample', 'undersample')
        random_state (int): Random state for reproducibility
        
    Returns:
        tuple: Resampled X and y
    """
    print(f"=== HANDLING CLASS IMBALANCE USING {method.upper()} ===")
    
    # Analyze current distribution
    dist_info = analyze_class_distribution(y)
    
    # Check if resampling is needed
    imbalance_ratio = dist_info['imbalance_ratio']
    if imbalance_ratio < 1.5:
        print("Class distribution is relatively balanced. No resampling needed.")
        return X, y
    
    # Apply selected method
    if method == 'smote':
        return apply_smote(X, y, random_state)
    elif method == 'oversample':
        return apply_random_oversampling(X, y, random_state)
    elif method == 'undersample':
        return apply_undersampling(X, y, random_state)
    else:
        print(f"Warning: Unknown method '{method}'. Using SMOTE as default.")
        return apply_smote(X, y, random_state)

if __name__ == "__main__":
    print("Class imbalance handling module ready for use")