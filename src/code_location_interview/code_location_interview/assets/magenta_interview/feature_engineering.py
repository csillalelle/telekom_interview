import numpy as np
import pandas as pd
from dagster import AssetOut, Output, multi_asset, asset, get_dagster_logger
import logging
import sys

log_fmt = "[%(asctime)s] %(message)s"
log_datefmt = "%Y-%m-%d %H:%M:%S"
logging.basicConfig(stream=sys.stdout, format=log_fmt, datefmt=log_datefmt, level=logging.INFO)
logger = get_dagster_logger(__name__)
group_name = "feature_engineering"


@asset(group_name=group_name)
def usage_features_aggregated(usage_info):
    """
    Aggregate usage_info from monthly to account level
    Creates features like avg, sum, min, max of usage metrics
    """
    logger.info("Aggregating usage features...")
    
    # Group by rating_account_id and aggregate
    usage_agg = usage_info.groupby('rating_account_id').agg({
        'used_gb': ['mean', 'sum', 'min', 'max', 'std'],
        'has_used_roaming': ['sum', 'mean']  # sum gives total months with roaming, mean gives proportion
    }).reset_index()
    
    # Flatten column names
    usage_agg.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                         for col in usage_agg.columns.values]
    
    # Rename for clarity
    usage_agg.columns = [
        'rating_account_id',
        'used_gb_avg',      # Average GB used per month
        'used_gb_total',    # Total GB used across all months
        'used_gb_min',      # Minimum GB used in any month
        'used_gb_max',      # Maximum GB used in any month
        'used_gb_std',      # Standard deviation of GB usage (consistency)
        'roaming_months_count',  # Number of months with roaming
        'roaming_rate'      # Proportion of months with roaming
    ]
    
    # Additional derived features
    usage_agg['usage_variability'] = usage_agg['used_gb_std'] / (usage_agg['used_gb_avg'] + 1)  # Coefficient of variation
    usage_agg['usage_trend'] = usage_agg['used_gb_max'] - usage_agg['used_gb_min']  # Range of usage
    
    logger.info(f"Created {len(usage_agg)} aggregated usage records with {usage_agg.shape[1]} features")
    logger.info(f"Usage features: {list(usage_agg.columns)}")
    
    return usage_agg


@asset(group_name=group_name)
def interaction_features_aggregated(customer_interactions):
    """
    Aggregate customer_interactions to customer level
    Creates features about interaction patterns
    """
    logger.info("Aggregating interaction features...")
    
    # Overall interaction statistics per customer
    overall_agg = customer_interactions.groupby('customer_id').agg({
        'n': ['sum', 'mean', 'max'],
        'days_since_last': ['min', 'mean']
    }).reset_index()
    
    overall_agg.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                           for col in overall_agg.columns.values]
    
    overall_agg.columns = [
        'customer_id',
        'total_interactions',        # Total number of interactions across all types
        'avg_interactions_per_type', # Average interactions per type
        'max_interactions_type',     # Max interactions for any single type
        'days_since_last_contact',   # Most recent contact (min days since last)
        'avg_days_since_contact'     # Average days since last contact
    ]
    
    # Pivot to get interaction counts by type
    type_pivot = customer_interactions.pivot_table(
        index='customer_id',
        columns='type_subtype',
        values='n',
        aggfunc='sum',
        fill_value=0
    ).reset_index()
    
    # Rename columns to be more descriptive
    type_pivot.columns = ['customer_id'] + [f'interactions_{col}' for col in type_pivot.columns[1:]]
    
    # Merge overall and type-specific features
    interaction_agg = overall_agg.merge(type_pivot, on='customer_id', how='left')
    
    # Fill NaN for customers without specific interaction types
    interaction_cols = [col for col in interaction_agg.columns if col.startswith('interactions_')]
    interaction_agg[interaction_cols] = interaction_agg[interaction_cols].fillna(0)
    
    # Additional derived features
    interaction_agg['has_any_interaction'] = 1  # Flag for customers with any interaction
    interaction_agg['interaction_diversity'] = (interaction_agg[interaction_cols] > 0).sum(axis=1)  # Number of different interaction types
    interaction_agg['recent_contact'] = (interaction_agg['days_since_last_contact'] < 30).astype(int)  # Contact in last 30 days
    
    logger.info(f"Created {len(interaction_agg)} aggregated interaction records with {interaction_agg.shape[1]} features")
    logger.info(f"Interaction features: {list(interaction_agg.columns)}")
    
    return interaction_agg


@multi_asset(
    group_name=group_name,
    outs={
        "df_model_input": AssetOut(description="Complete feature set for modeling"),
        "feature_names": AssetOut(description="List of feature names for modeling")
    }
)
def create_model_features(core_data, usage_features_aggregated, interaction_features_aggregated):
    """
    Combine all data sources and create final feature set for modeling
    """
    logger.info("Creating complete feature set...")
    
    # Start with core data
    df = core_data.copy()
    
    # Merge usage features
    df = df.merge(usage_features_aggregated, on='rating_account_id', how='left')
    logger.info(f"After merging usage features: {df.shape}")
    
    # Merge interaction features
    df = df.merge(interaction_features_aggregated, on='customer_id', how='left')
    logger.info(f"After merging interaction features: {df.shape}")
    
    # Fill NaN for customers without interactions
    interaction_cols = [col for col in df.columns if 'interaction' in col or 'contact' in col]
    df[interaction_cols] = df[interaction_cols].fillna(0)
    
    # Create additional engineered features
    logger.info("Creating engineered features...")
    
    # 1. Usage vs Available ratio
    df['usage_to_available_ratio'] = df['used_gb_avg'] / (df['available_gb'].replace(0, np.nan) + 1)
    df['usage_to_available_ratio'] = df['usage_to_available_ratio'].fillna(0)
    
    # 2. Contract status features
    df['is_out_of_binding'] = (df['remaining_binding_days'] < 0).astype(int)
    df['binding_ending_soon'] = ((df['remaining_binding_days'] >= 0) & 
                                  (df['remaining_binding_days'] < 90)).astype(int)
    
    # 3. Customer value indicators
    df['is_high_value'] = (df['gross_mrc'] > df['gross_mrc'].quantile(0.75)).astype(int)
    df['is_low_usage'] = (df['used_gb_avg'] < df['used_gb_avg'].quantile(0.25)).astype(int)
    
    # 4. Age groups
    df['age_young'] = (df['age'] < 35).astype(int)
    df['age_middle'] = ((df['age'] >= 35) & (df['age'] < 55)).astype(int)
    df['age_senior'] = (df['age'] >= 55).astype(int)
    
    # 5. Contract tenure groups
    df['contract_new'] = (df['contract_lifetime_days'] < 365).astype(int)
    df['contract_established'] = ((df['contract_lifetime_days'] >= 365) & 
                                   (df['contract_lifetime_days'] < 1095)).astype(int)
    df['contract_long_term'] = (df['contract_lifetime_days'] >= 1095).astype(int)
    
    # 6. Interaction recency
    df['has_recent_interaction'] = (df['days_since_last_contact'] < 60).astype(int)
    df['has_recent_interaction'] = df['has_recent_interaction'].fillna(0)
    
    # One-hot encode smartphone brand
    brand_dummies = pd.get_dummies(df['smartphone_brand'], prefix='brand', drop_first=True)
    df = pd.concat([df, brand_dummies], axis=1)
    
    # Define feature columns (exclude identifiers and target)
    exclude_cols = ['rating_account_id', 'customer_id', 'has_done_upselling', 'smartphone_brand']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    logger.info(f"Final dataset shape: {df.shape}")
    logger.info(f"Number of features: {len(feature_cols)}")
    logger.info(f"Features: {feature_cols}")
    
    # Check for any remaining NaN values
    nan_counts = df[feature_cols].isnull().sum()
    if nan_counts.sum() > 0:
        logger.warning(f"NaN values found in features:\n{nan_counts[nan_counts > 0]}")
        # Fill remaining NaN with 0 or median
        for col in feature_cols:
            if df[col].isnull().sum() > 0:
                if df[col].dtype in ['float64', 'int64']:
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(0)
    
    return Output(df, output_name="df_model_input"), Output(feature_cols, output_name="feature_names")


@asset(group_name=group_name)
def feature_importance_analysis(df_model_input, feature_names):
    """
    Quick analysis of feature distributions and basic statistics
    """
    logger.info("Analyzing feature importance and distributions...")
    
    X = df_model_input[feature_names]
    y = df_model_input['has_done_upselling']
    
    # Basic statistics
    feature_stats = {
        'mean': X.mean().to_dict(),
        'std': X.std().to_dict(),
        'missing': X.isnull().sum().to_dict()
    }
    
    # Correlation with target
    correlations = {}
    for col in feature_names:
        if X[col].dtype in ['float64', 'int64']:
            corr = X[col].corr(y)
            correlations[col] = corr
    
    # Sort by absolute correlation
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    
    logger.info("\nTop 15 Features by Correlation with Target:")
    for feat, corr in sorted_corr[:15]:
        logger.info(f"{feat}: {corr:.4f}")
    
    return {
        'feature_stats': feature_stats,
        'correlations': correlations,
        'top_features': sorted_corr[:15]
    }