import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def get_hybrid_data_constant_size(real_data, syn_data, syn_data_percentage):
    """
    Create hybrid dataset with constant size by replacing real with synthetic.
    
    Args:
        real_data: Real training data
        syn_data: Synthetic data
        syn_data_percentage: Percentage of synthetic data (0.0 to 1.0)
    
    Returns:
        Hybrid dataset with same size as real_data
    
    Example:
        - real_data has 1000 samples
        - syn_data_percentage = 0.3
        - Result: 700 real + 300 synthetic = 1000 total
    """
    total_size = len(real_data)
    syn_size = int(total_size * syn_data_percentage)
    real_size = total_size - syn_size
    
    print(f"Total size: {total_size}")
    print(f"Real data size: {real_size} ({(1-syn_data_percentage)*100:.0f}%)")
    print(f"Synthetic data size: {syn_size} ({syn_data_percentage*100:.0f}%)")
    
    # Sample from real data
    real_data_sampled = real_data.sample(n=real_size, random_state=42)
    
    # Sample from synthetic data
    if syn_size > len(syn_data):
        print(f"Warning: Not enough synthetic data ({len(syn_data)} available, {syn_size} needed)")
        print(f"Using all available synthetic data and adjusting real data size")
        syn_data_sampled = syn_data
        real_size = total_size - len(syn_data)
        real_data_sampled = real_data.sample(n=real_size, random_state=42)
    else:
        syn_data_sampled = syn_data.sample(n=syn_size, random_state=42)
    
    # Combine and shuffle
    train_data = pd.concat([real_data_sampled, syn_data_sampled], ignore_index=True)
    train_data = train_data.sample(frac=1, random_state=42).reset_index(drop=True)

    
    print(f"Final train data size: {len(train_data)}")
    print(f"Class distribution: {train_data['in_hospital_death'].value_counts().to_dict()}")
    
    return train_data


# Stratified replacement (better for imbalanced data)
def get_hybrid_data_stratified(real_data, syn_data, syn_data_percentage, seed=42, target_col='in_hospital_death'):
    """
    Create hybrid dataset maintaining class balance.
    Replaces real data with synthetic while preserving class distribution.
    """
    total_size = len(real_data)
    syn_size = int(total_size * syn_data_percentage)
    real_size = total_size - syn_size
    
    # Get class distribution from real data
    real_class_dist = real_data[target_col].value_counts(normalize=True)
    
    print(f"Total size: {total_size}")
    print(f"Target class distribution: {real_class_dist.to_dict()}")
    
    # Stratified sampling from real data
    real_data_sampled = real_data.groupby(target_col, group_keys=False).apply(
        lambda x: x.sample(n=int(len(x) * (real_size / total_size)), random_state=seed)
    ).reset_index(drop=True)
    
    # Stratified sampling from synthetic data
    syn_data_sampled = syn_data.groupby(target_col, group_keys=False).apply(
        lambda x: x.sample(n=min(len(x), int(len(real_data[real_data[target_col] == x.name]) * 
                                              (syn_size / total_size))), 
                          random_state=seed)
    ).reset_index(drop=True)
    
    # Combine
    train_data = pd.concat([real_data_sampled, syn_data_sampled], ignore_index=True)
    
    print(f"Real data: {len(real_data_sampled)} samples")
    print(f"  Class distribution: {real_data_sampled[target_col].value_counts().to_dict()}")
    print(f"Synthetic data: {len(syn_data_sampled)} samples")
    print(f"  Class distribution: {syn_data_sampled[target_col].value_counts().to_dict()}")
    print(f"Final train data: {len(train_data)} samples")
    print(f"  Class distribution: {train_data[target_col].value_counts().to_dict()}")
    
    return train_data


def get_hybrid_data_augmentation(real_data, syn_data, syn_data_percentage, stratify=False, seed=42, target_col='in_hospital_death'):
    # Hybrid approach: train on real + synthetic data with given synthetic data percentage
    syn_size = int(len(real_data) * syn_data_percentage)
    print(f"Synthetic data size for {syn_data_percentage*100}%: {syn_size}")
    if syn_size > len(syn_data):
        syn_size = len(syn_data)
        print(f"Adjusted synthetic data size to available data: {syn_size}")

    n_classes = syn_data[target_col].nunique(dropna=True)
    if stratify and n_classes > 1:
        print("Using stratified sampling for synthetic data.")
        syn_data_sampled, _ = train_test_split(
            syn_data,
            train_size=syn_size,
            stratify=syn_data[target_col],
            random_state=seed
        )
    else:
        syn_data_sampled = syn_data.sample(n=syn_size, random_state=seed)
    print(f"Synthetic data: {syn_data_sampled[target_col].value_counts()}")
    train_data = pd.concat([real_data, syn_data_sampled], ignore_index=True)
    return train_data
