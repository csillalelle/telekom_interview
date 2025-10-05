import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dagster import asset, get_dagster_logger

logger = get_dagster_logger(__name__)
group_name = "exploration"


@asset(group_name=group_name)
def data_exploration_summary(core_data, usage_info, customer_interactions):
    """
    Generate comprehensive exploratory data analysis summary
    """
    exploration_results = {}
    
    # ============ CORE DATA EXPLORATION ============
    logger.info("=" * 50)
    logger.info("CORE DATA EXPLORATION")
    logger.info("=" * 50)
    
    exploration_results['core_data'] = {
        'shape': core_data.shape,
        'null_counts': core_data.isnull().sum().to_dict(),
        'dtypes': core_data.dtypes.to_dict()
    }
    
    logger.info(f"Shape: {core_data.shape}")
    logger.info(f"\nNull values:\n{core_data.isnull().sum()}")
    logger.info(f"\nData types:\n{core_data.dtypes}")
    
    # Target variable distribution
    target_dist = core_data['has_done_upselling'].value_counts(normalize=True)
    exploration_results['target_distribution'] = target_dist.to_dict()
    logger.info(f"\nTarget Variable (has_done_upselling) Distribution:")
    logger.info(f"No upsell (0): {target_dist.get(0, 0):.2%}")
    logger.info(f"Upsell (1): {target_dist.get(1, 0):.2%}")
    
    # Numerical features summary
    numerical_cols = ['age', 'contract_lifetime_days', 'remaining_binding_days', 
                      'available_gb', 'gross_mrc']
    logger.info(f"\nNumerical Features Summary:")
    logger.info(core_data[numerical_cols].describe())
    
    # Categorical features
    logger.info(f"\nSmartphone Brand Distribution:")
    logger.info(core_data['smartphone_brand'].value_counts(normalize=True))
    
    logger.info(f"\nSpecial Offer Distribution:")
    logger.info(core_data['has_special_offer'].value_counts(normalize=True))
    
    logger.info(f"\nMagenta1 Customer Distribution:")
    logger.info(core_data['is_magenta1_customer'].value_counts(normalize=True))
    
    # ============ USAGE INFO EXPLORATION ============
    logger.info("\n" + "=" * 50)
    logger.info("USAGE INFO EXPLORATION")
    logger.info("=" * 50)
    
    exploration_results['usage_info'] = {
        'shape': usage_info.shape,
        'unique_accounts': usage_info['rating_account_id'].nunique(),
        'billing_periods': usage_info['billed_period_month_d'].nunique()
    }
    
    logger.info(f"Shape: {usage_info.shape}")
    logger.info(f"Unique rating_account_ids: {usage_info['rating_account_id'].nunique()}")
    logger.info(f"Billing periods: {usage_info['billed_period_month_d'].unique()}")
    
    logger.info(f"\nUsage Statistics:")
    logger.info(usage_info[['has_used_roaming', 'used_gb']].describe())
    
    logger.info(f"\nRoaming Usage Distribution:")
    logger.info(usage_info['has_used_roaming'].value_counts(normalize=True))
    
    # ============ CUSTOMER INTERACTIONS EXPLORATION ============
    logger.info("\n" + "=" * 50)
    logger.info("CUSTOMER INTERACTIONS EXPLORATION")
    logger.info("=" * 50)
    
    exploration_results['customer_interactions'] = {
        'shape': customer_interactions.shape,
        'unique_customers': customer_interactions['customer_id'].nunique(),
        'interaction_types': customer_interactions['type_subtype'].unique().tolist()
    }
    
    logger.info(f"Shape: {customer_interactions.shape}")
    logger.info(f"Unique customer_ids: {customer_interactions['customer_id'].nunique()}")
    
    logger.info(f"\nInteraction Type Distribution:")
    logger.info(customer_interactions['type_subtype'].value_counts())
    
    logger.info(f"\nInteraction Count (n) Statistics:")
    logger.info(customer_interactions['n'].describe())
    
    logger.info(f"\nDays Since Last Interaction Statistics:")
    logger.info(customer_interactions['days_since_last'].describe())
    
    # ============ CORRELATIONS WITH TARGET ============
    logger.info("\n" + "=" * 50)
    logger.info("CORRELATIONS WITH TARGET VARIABLE")
    logger.info("=" * 50)
    
    # Select numerical columns for correlation
    corr_cols = ['age', 'contract_lifetime_days', 'remaining_binding_days', 
                 'has_special_offer', 'is_magenta1_customer', 'available_gb', 
                 'gross_mrc', 'has_done_upselling']
    
    correlation_matrix = core_data[corr_cols].corr()
    target_correlations = correlation_matrix['has_done_upselling'].sort_values(ascending=False)
    
    logger.info("\nCorrelations with has_done_upselling:")
    logger.info(target_correlations)
    
    exploration_results['target_correlations'] = target_correlations.to_dict()
    
    # ============ DATA QUALITY CHECKS ============
    logger.info("\n" + "=" * 50)
    logger.info("DATA QUALITY CHECKS")
    logger.info("=" * 50)
    
    # Check for duplicates
    logger.info(f"Duplicate rating_account_ids in core_data: {core_data['rating_account_id'].duplicated().sum()}")
    logger.info(f"Duplicate customer_ids in core_data: {core_data['customer_id'].duplicated().sum()}")
    
    # Check join keys
    core_accounts = set(core_data['rating_account_id'])
    usage_accounts = set(usage_info['rating_account_id'])
    core_customers = set(core_data['customer_id'])
    interaction_customers = set(customer_interactions['customer_id'])
    
    logger.info(f"\nAccounts in core_data but not in usage_info: {len(core_accounts - usage_accounts)}")
    logger.info(f"Accounts in usage_info but not in core_data: {len(usage_accounts - core_accounts)}")
    logger.info(f"Customers in core_data but not in customer_interactions: {len(core_customers - interaction_customers)}")
    logger.info(f"Customers in customer_interactions but not in core_data: {len(interaction_customers - core_customers)}")
    
    return exploration_results


@asset(group_name=group_name)
def upsell_analysis_by_segments(core_data):
    """
    Analyze upselling rates across different customer segments
    """
    logger.info("=" * 50)
    logger.info("UPSELL ANALYSIS BY SEGMENTS")
    logger.info("=" * 50)
    
    analysis_results = {}
    
    # By age groups
    core_data['age_group'] = pd.cut(core_data['age'], 
                                     bins=[0, 30, 45, 60, 100], 
                                     labels=['18-30', '31-45', '46-60', '60+'])
    
    age_upsell = core_data.groupby('age_group')['has_done_upselling'].agg(['mean', 'count'])
    logger.info("\nUpsell Rate by Age Group:")
    logger.info(age_upsell)
    analysis_results['age_group'] = age_upsell.to_dict()
    
    # By contract lifetime
    core_data['contract_group'] = pd.cut(core_data['contract_lifetime_days'], 
                                          bins=[0, 365, 730, 1095, 5000], 
                                          labels=['0-1yr', '1-2yr', '2-3yr', '3yr+'])
    
    contract_upsell = core_data.groupby('contract_group')['has_done_upselling'].agg(['mean', 'count'])
    logger.info("\nUpsell Rate by Contract Lifetime:")
    logger.info(contract_upsell)
    analysis_results['contract_group'] = contract_upsell.to_dict()
    
    # By remaining binding days
    core_data['binding_status'] = pd.cut(core_data['remaining_binding_days'], 
                                          bins=[-1000, 0, 180, 365, 1000], 
                                          labels=['Past binding', '0-6mo', '6-12mo', '12mo+'])
    
    binding_upsell = core_data.groupby('binding_status')['has_done_upselling'].agg(['mean', 'count'])
    logger.info("\nUpsell Rate by Remaining Binding Days:")
    logger.info(binding_upsell)
    analysis_results['binding_status'] = binding_upsell.to_dict()
    
    # By smartphone brand
    brand_upsell = core_data.groupby('smartphone_brand')['has_done_upselling'].agg(['mean', 'count'])
    logger.info("\nUpsell Rate by Smartphone Brand:")
    logger.info(brand_upsell)
    analysis_results['smartphone_brand'] = brand_upsell.to_dict()
    
    # By available GB
    gb_upsell = core_data.groupby('available_gb')['has_done_upselling'].agg(['mean', 'count'])
    logger.info("\nUpsell Rate by Available GB:")
    logger.info(gb_upsell)
    analysis_results['available_gb'] = gb_upsell.to_dict()
    
    # By special offer status
    special_offer_upsell = core_data.groupby('has_special_offer')['has_done_upselling'].agg(['mean', 'count'])
    logger.info("\nUpsell Rate by Special Offer Status:")
    logger.info(special_offer_upsell)
    
    # By Magenta1 status
    magenta1_upsell = core_data.groupby('is_magenta1_customer')['has_done_upselling'].agg(['mean', 'count'])
    logger.info("\nUpsell Rate by Magenta1 Status:")
    logger.info(magenta1_upsell)
    
    return analysis_results