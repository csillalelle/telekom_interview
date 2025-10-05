"""
Advanced XGBoost Optimization - Early Stopping & Grid Search
Goal: Maximize test performance while minimizing overfitting
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold, 
    GridSearchCV, RandomizedSearchCV
)
from sklearn.metrics import roc_auc_score, make_scorer
import xgboost as xgb
from dagster import asset, get_dagster_logger

logger = get_dagster_logger(__name__)
group_name = "training"


# ============================================================================
# METHOD 1: EARLY STOPPING (Prevent Overfitting Automatically)
# ============================================================================

@asset(group_name=group_name)
def xgboost_model_early_stopping(X_train, y_train):
    """
    XGBoost with early stopping - automatically stops when validation performance plateaus
    
    Benefits:
    - No manual tuning of n_estimators
    - Automatically prevents overfitting
    - Often achieves better test performance than fixed iterations
    """
    logger.info("Training XGBoost with EARLY STOPPING...")
    logger.info("="*60)
    
    # Split training data into train/validation for early stopping
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, 
        test_size=0.2, 
        random_state=42, 
        stratify=y_train
    )
    
    logger.info(f"Training set: {len(X_tr)} samples")
    logger.info(f"Validation set: {len(X_val)} samples (for early stopping)")
    
    # Calculate scale_pos_weight
    scale_pos_weight = (y_tr == 0).sum() / (y_tr == 1).sum()
    logger.info(f"Scale pos weight: {scale_pos_weight:.2f}")
    
    # Parameters optimized for early stopping
    xgb_params = {
        # Tree structure - moderate complexity
        'max_depth': 4,
        'min_child_weight': 3,
        'gamma': 0.05,
        
        # Learning parameters - moderate learning rate
        'learning_rate': 0.05,
        'n_estimators': 500,              # Set high, early stopping will prevent overfitting
        
        # Sampling
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'colsample_bylevel': 0.8,
        
        # Regularization
        'reg_alpha': 0.5,
        'reg_lambda': 3.0,
        
        # Other
        'objective': 'binary:logistic',
        'scale_pos_weight': scale_pos_weight,
        'random_state': 42,
        'eval_metric': 'auc',
        
        # Early stopping configuration
        'early_stopping_rounds': 30       # Stop if no improvement for 30 rounds
    }
    
    logger.info("\nModel Parameters:")
    for key, value in xgb_params.items():
        if key != 'early_stopping_rounds':
            logger.info(f"  {key}: {value}")
    logger.info(f"  early_stopping_rounds: {xgb_params['early_stopping_rounds']}")
    
    # Create model
    xgb_model = xgb.XGBClassifier(**xgb_params)
    
    # Fit with early stopping
    logger.info("\nTraining with early stopping...")
    logger.info("Monitoring validation AUC...")
    
    xgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_tr, y_tr), (X_val, y_val)],
        eval_metric='auc',
        verbose=False
    )
    
    # Report early stopping results
    logger.info("\n" + "="*60)
    logger.info("EARLY STOPPING RESULTS")
    logger.info("="*60)
    logger.info(f"Best iteration: {xgb_model.best_iteration}")
    logger.info(f"Best validation AUC: {xgb_model.best_score:.4f}")
    logger.info(f"Total rounds trained: {xgb_model.n_estimators}")
    logger.info(f"Stopped early: {'Yes' if xgb_model.best_iteration < xgb_model.n_estimators else 'No'}")
    
    if xgb_model.best_iteration < xgb_model.n_estimators:
        rounds_saved = xgb_model.n_estimators - xgb_model.best_iteration
        logger.info(f"Rounds saved by early stopping: {rounds_saved}")
    
    # Evaluate on full training set
    train_pred_proba = xgb_model.predict_proba(X_train)[:, 1]
    train_auc = roc_auc_score(y_train, train_pred_proba)
    logger.info(f"\nFull training set AUC: {train_auc:.4f}")
    
    # Cross-validation on full training set
    cv_scores = cross_val_score(
        xgb_model, X_train, y_train,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='roc_auc'
    )
    
    logger.info(f"Cross-validation AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logger.info("\nTop 15 Most Important Features:")
    logger.info(feature_importance.head(15).to_string())
    
    return {
        'model': xgb_model,
        'best_iteration': xgb_model.best_iteration,
        'best_score': xgb_model.best_score,
        'cv_scores': cv_scores,
        'feature_importance': feature_importance,
        'params': xgb_params
    }


# ============================================================================
# METHOD 2: GRID SEARCH (Find Optimal Hyperparameters)
# ============================================================================

@asset(group_name=group_name)
def xgboost_model_grid_search(X_train, y_train):
    """
    Comprehensive grid search to find optimal XGBoost hyperparameters
    
    This systematically tests different parameter combinations
    to find the best model configuration
    """
    logger.info("GRID SEARCH FOR OPTIMAL HYPERPARAMETERS")
    logger.info("="*60)
    logger.info("This may take 5-10 minutes...")
    
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    # Define parameter grid - comprehensive search
    param_grid = {
        'max_depth': [3, 4, 5],
        'min_child_weight': [3, 5, 7],
        'gamma': [0, 0.05, 0.1],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'learning_rate': [0.03, 0.05, 0.1],
        'reg_alpha': [0, 0.5, 1.0],
        'reg_lambda': [1, 3, 5]
    }
    
    logger.info(f"\nParameter grid:")
    for param, values in param_grid.items():
        logger.info(f"  {param}: {values}")
    
    total_combinations = np.prod([len(v) for v in param_grid.values()])
    logger.info(f"\nTotal combinations: {total_combinations}")
    logger.info(f"With 5-fold CV: {total_combinations * 5} model fits")
    
    # Base model
    xgb_base = xgb.XGBClassifier(
        n_estimators=150,                  # Moderate number for speed
        objective='binary:logistic',
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='auc'
    )
    
    # Grid search with cross-validation
    grid_search = GridSearchCV(
        estimator=xgb_base,
        param_grid=param_grid,
        scoring='roc_auc',
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        verbose=2,
        n_jobs=-1,                         # Use all CPU cores
        return_train_score=True
    )
    
    logger.info("\nStarting grid search...")
    logger.info("Progress will be shown below:")
    logger.info("-" * 60)
    
    # Fit grid search
    grid_search.fit(X_train, y_train)
    
    # Report results
    logger.info("\n" + "="*60)
    logger.info("GRID SEARCH RESULTS")
    logger.info("="*60)
    
    logger.info("\n🏆 BEST PARAMETERS:")
    for param, value in grid_search.best_params_.items():
        logger.info(f"  {param}: {value}")
    
    logger.info(f"\n🏆 BEST CV SCORE: {grid_search.best_score_:.4f}")
    
    # Get best model
    best_model = grid_search.best_estimator_
    
    # Show top 5 parameter combinations
    results_df = pd.DataFrame(grid_search.cv_results_)
    results_df = results_df.sort_values('rank_test_score')
    
    logger.info("\n📊 TOP 5 PARAMETER COMBINATIONS:")
    logger.info("-" * 60)
    
    for idx in range(min(5, len(results_df))):
        row = results_df.iloc[idx]
        logger.info(f"\nRank {idx + 1}:")
        logger.info(f"  CV Score: {row['mean_test_score']:.4f} (+/- {row['std_test_score']:.4f})")
        logger.info(f"  Train Score: {row['mean_train_score']:.4f}")
        logger.info(f"  Overfitting Gap: {row['mean_train_score'] - row['mean_test_score']:.4f}")
        logger.info(f"  Params: {row['params']}")
    
    # Analyze overfitting
    best_train_score = results_df.iloc[0]['mean_train_score']
    best_test_score = results_df.iloc[0]['mean_test_score']
    gap = best_train_score - best_test_score
    
    logger.info("\n" + "="*60)
    logger.info("OVERFITTING ANALYSIS")
    logger.info("="*60)
    logger.info(f"Best model train score: {best_train_score:.4f}")
    logger.info(f"Best model test score:  {best_test_score:.4f}")
    logger.info(f"Train-test gap:         {gap:.4f}")
    
    if gap < 0.05:
        logger.info("✅ Excellent generalization!")
    elif gap < 0.10:
        logger.info("✅ Good generalization")
    else:
        logger.warning("⚠️  Some overfitting detected")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logger.info("\nTop 15 Most Important Features:")
    logger.info(feature_importance.head(15).to_string())
    
    return {
        'model': best_model,
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'cv_results': results_df,
        'feature_importance': feature_importance,
        'grid_search': grid_search
    }


# ============================================================================
# METHOD 3: RANDOMIZED SEARCH (Faster Alternative to Grid Search)
# ============================================================================

@asset(group_name=group_name)
def xgboost_model_random_search(X_train, y_train):
    """
    Randomized search - faster alternative to grid search
    Tests random combinations of parameters instead of all combinations
    
    Good for: Initial exploration or when grid search is too slow
    """
    logger.info("RANDOMIZED SEARCH FOR HYPERPARAMETERS")
    logger.info("="*60)
    logger.info("Testing 50 random parameter combinations...")
    
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    # Define parameter distributions
    param_distributions = {
        'max_depth': [2, 3, 4, 5, 6],
        'min_child_weight': [1, 3, 5, 7, 10],
        'gamma': [0, 0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.2],
        'reg_alpha': [0, 0.1, 0.5, 1.0, 2.0],
        'reg_lambda': [0.5, 1, 2, 3, 5, 10],
        'n_estimators': [100, 150, 200, 300]
    }
    
    # Base model
    xgb_base = xgb.XGBClassifier(
        objective='binary:logistic',
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='auc'
    )
    
    # Randomized search
    random_search = RandomizedSearchCV(
        estimator=xgb_base,
        param_distributions=param_distributions,
        n_iter=50,                         # Number of random combinations to try
        scoring='roc_auc',
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        verbose=2,
        n_jobs=-1,
        random_state=42,
        return_train_score=True
    )
    
    logger.info("\nStarting randomized search...")
    logger.info("Testing 50 random parameter combinations with 5-fold CV")
    logger.info("-" * 60)
    
    # Fit
    random_search.fit(X_train, y_train)
    
    # Report results
    logger.info("\n" + "="*60)
    logger.info("RANDOMIZED SEARCH RESULTS")
    logger.info("="*60)
    
    logger.info("\n🏆 BEST PARAMETERS:")
    for param, value in random_search.best_params_.items():
        logger.info(f"  {param}: {value}")
    
    logger.info(f"\n🏆 BEST CV SCORE: {random_search.best_score_:.4f}")
    
    # Get best model
    best_model = random_search.best_estimator_
    
    # Show top 5 combinations
    results_df = pd.DataFrame(random_search.cv_results_)
    results_df = results_df.sort_values('rank_test_score')
    
    logger.info("\n📊 TOP 5 PARAMETER COMBINATIONS:")
    for idx in range(min(5, len(results_df))):
        row = results_df.iloc[idx]
        logger.info(f"\nRank {idx + 1}:")
        logger.info(f"  CV Score: {row['mean_test_score']:.4f} (+/- {row['std_test_score']:.4f})")
        logger.info(f"  Gap: {row['mean_train_score'] - row['mean_test_score']:.4f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return {
        'model': best_model,
        'best_params': random_search.best_params_,
        'best_score': random_search.best_score_,
        'cv_results': results_df,
        'feature_importance': feature_importance
    }


# ============================================================================
# METHOD 4: EARLY STOPPING + GRID SEARCH (BEST OF BOTH WORLDS)
# ============================================================================

@asset(group_name=group_name)
def xgboost_model_optimized(X_train, y_train):
    """
    ULTIMATE MODEL: Grid search to find best parameters, then early stopping
    
    This combines the strengths of both methods:
    1. Grid search finds optimal hyperparameters
    2. Early stopping prevents overfitting during training
    
    This is the recommended approach for production!
    """
    logger.info("OPTIMIZED XGBOOST: GRID SEARCH + EARLY STOPPING")
    logger.info("="*60)
    
    # Split for early stopping
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    scale_pos_weight = (y_tr == 0).sum() / (y_tr == 1).sum()
    
    # STEP 1: Grid search with smaller parameter space (faster)
    logger.info("\n📍 STEP 1: Grid search for optimal parameters...")
    
    param_grid = {
        'max_depth': [3, 4, 5],
        'min_child_weight': [3, 5, 7],
        'gamma': [0, 0.05, 0.1],
        'subsample': [0.7, 0.8],
        'colsample_bytree': [0.7, 0.8],
        'learning_rate': [0.05, 0.1],
        'reg_alpha': [0.5, 1.0],
        'reg_lambda': [2, 3, 5]
    }
    
    xgb_base = xgb.XGBClassifier(
        n_estimators=100,
        objective='binary:logistic',
        scale_pos_weight=scale_pos_weight,
        random_state=42
    )
    
    grid_search = GridSearchCV(
        estimator=xgb_base,
        param_grid=param_grid,
        scoring='roc_auc',
        cv=3,  # 3-fold for speed
        verbose=1,
        n_jobs=-1
    )
    
    grid_search.fit(X_tr, y_tr)
    
    best_params = grid_search.best_params_
    logger.info(f"\n✅ Best parameters found:")
    for param, value in best_params.items():
        logger.info(f"   {param}: {value}")
    
    # STEP 2: Train final model with early stopping using best params
    logger.info("\n📍 STEP 2: Training with early stopping using best parameters...")
    
    final_model = xgb.XGBClassifier(
        **best_params,
        n_estimators=500,              # High number, will be stopped early
        objective='binary:logistic',
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        early_stopping_rounds=30
    )
    
    final_model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    logger.info(f"\n✅ Training complete!")
    logger.info(f"   Best iteration: {final_model.best_iteration}")
    logger.info(f"   Best validation AUC: {final_model.best_score:.4f}")
    
    # Evaluate
    train_pred_proba = final_model.predict_proba(X_train)[:, 1]
    train_auc = roc_auc_score(y_train, train_pred_proba)
    
    logger.info(f"\n📊 Full training set AUC: {train_auc:.4f}")
    
    # Cross-validation
    cv_scores = cross_val_score(
        final_model, X_train, y_train,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='roc_auc'
    )
    
    logger.info(f"📊 Cross-validation AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logger.info("\n🎯 Top 15 Most Important Features:")
    logger.info(feature_importance.head(15).to_string())
    
    logger.info("\n" + "="*60)
    logger.info("✅ OPTIMIZED MODEL COMPLETE!")
    logger.info("   This model combines grid search + early stopping")
    logger.info("   Expected to have best generalization performance")
    logger.info("="*60)
    
    return {
        'model': final_model,
        'best_params': best_params,
        'best_iteration': final_model.best_iteration,
        'best_score': final_model.best_score,
        'cv_scores': cv_scores,
        'feature_importance': feature_importance
    }