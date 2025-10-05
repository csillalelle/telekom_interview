import logging
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_auc_score, 
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    f1_score
)
import xgboost as xgb
from dagster import AssetOut, Output, multi_asset, asset, get_dagster_logger

log_fmt = "[%(asctime)s] %(message)s"
log_datefmt = "%Y-%m-%d %H:%M:%S"
logging.basicConfig(stream=sys.stdout, format=log_fmt, datefmt=log_datefmt, level=logging.INFO)
logger = get_dagster_logger(__name__)
group_name = "training"


@multi_asset(
    group_name=group_name,
    outs={
        "X_train": AssetOut(),
        "X_test": AssetOut(),
        "y_train": AssetOut(),
        "y_test": AssetOut(),
        "scaler": AssetOut()
    }
)
def split_and_scale_data(df_model_input, feature_names):
    """
    Split data into train/test sets and scale features
    """
    logger.info("Splitting and scaling data...")
    
    # Separate features and target
    X = df_model_input[feature_names].copy()
    y = df_model_input['has_done_upselling'].copy()
    
    logger.info(f"Total samples: {len(X)}")
    logger.info(f"Positive class (upsell): {y.sum()} ({y.mean():.2%})")
    logger.info(f"Negative class (no upsell): {(1-y).sum()} ({(1-y).mean():.2%})")
    
    # Split into train and test sets (80/20 split, stratified by target)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )
    
    logger.info(f"Train set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")
    logger.info(f"Train positive rate: {y_train.mean():.2%}")
    logger.info(f"Test positive rate: {y_test.mean():.2%}")
    
    # Scale features (for logistic regression)
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    
    logger.info("Data scaling completed")
    
    return (
        Output(X_train_scaled, output_name="X_train"),
        Output(X_test_scaled, output_name="X_test"),
        Output(y_train, output_name="y_train"),
        Output(y_test, output_name="y_test"),
        Output(scaler, output_name="scaler")
    )


@asset(group_name=group_name)
def logistic_regression_model(X_train, y_train):
    """
    Train a Logistic Regression model
    
    Why Logistic Regression?
    - Interpretable: Easy to explain coefficients to business stakeholders
    - Baseline model: Good starting point to understand feature importance
    - Fast to train: Efficient for production deployment
    - Probability outputs: Provides calibrated probability scores for ranking customers
    """
    logger.info("Training Logistic Regression model...")
    
    # Handle class imbalance with class_weight='balanced'
    # This gives more weight to the minority class (upselling customers)
    lr_model = LogisticRegression(
        class_weight='balanced',  # Handles imbalanced data
        max_iter=1000,            # Ensure convergence
        random_state=42,
        solver='lbfgs',           # Good for small to medium datasets
        C=1.0                     # Regularization strength (inverse)
    )
    
    # Train the model
    lr_model.fit(X_train, y_train)
    
    # Cross-validation
    cv_scores = cross_val_score(
        lr_model, X_train, y_train, 
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='roc_auc'
    )
    
    logger.info(f"Cross-validation ROC-AUC scores: {cv_scores}")
    logger.info(f"Mean CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    # Get feature importance (coefficients)
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'coefficient': lr_model.coef_[0],
        'abs_coefficient': np.abs(lr_model.coef_[0])
    }).sort_values('abs_coefficient', ascending=False)
    
    logger.info("\nTop 15 Most Important Features (by absolute coefficient):")
    logger.info(feature_importance.head(15).to_string())
    
    return {
        'model': lr_model,
        'cv_scores': cv_scores,
        'feature_importance': feature_importance
    }


@asset(group_name=group_name)
def xgboost_model(X_train, y_train):
    """
    Train an XGBoost model optimized for PR-AUC (Precision-Recall AUC)
    
    Why XGBoost?
    - Handles non-linear relationships: Captures complex patterns in data
    - Feature interactions: Automatically learns interactions between features
    - Robust to outliers: Tree-based models are less sensitive to extreme values
    - High performance: Often achieves better predictive accuracy than linear models
    - Built-in feature importance: Provides multiple ways to rank features
    
    Why PR-AUC optimization?
    - More informative for imbalanced datasets (7% upsell rate)
    - Focuses on precision-recall tradeoff which is more relevant for business
    - Less sensitive to class imbalance than ROC-AUC
    """
    logger.info("Training XGBoost model optimized for PR-AUC...")
    
    # Calculate scale_pos_weight for class imbalance
    # This parameter helps XGBoost handle imbalanced datasets
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    logger.info(f"Scale pos weight (for class imbalance): {scale_pos_weight:.2f}")
    
    # XGBoost parameters
    xgb_params = {
        # Parameters changed based on GridSearch results
        'max_depth': 3,                    # Increased from 2 for better minority class capture
        'min_child_weight': 5,             # Reduced from 7 to allow more splits on minority class
        'gamma': 0.1,                      # Increased for more conservative splits
        
        # Learning parameters
        'learning_rate': 0.05,             # Lower learning rate for better generalization
        'n_estimators': 400,               # Increased boosting rounds for better minority class learning
        
        # Sampling to reduce overfitting while capturing minority class
        'subsample': 0.75,                 # Slightly increased to see more minority samples
        'colsample_bytree': 0.85,          # Reduced to add more randomness
        
        # Regularization - prevent overfitting
        'reg_alpha': 1.5,                  # Reduced L1 for more flexibility
        'reg_lambda': 1.0,                 # Increased L2 for better generalization
        
        # Other parameters
        'objective': 'binary:logistic',   # Binary classification
        'scale_pos_weight': scale_pos_weight,  # Handle class imbalance
        'random_state': 42,
        'eval_metric': 'aucpr'            # Optimize for PR-AUC instead of ROC-AUC
    }

    # Train the model
    xgb_model = xgb.XGBClassifier(**xgb_params)
    xgb_model.fit(X_train, y_train, verbose=False)
    
    # Cross-validation 
    cv_scores_roc = cross_val_score(
        xgb_model, X_train, y_train,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='roc_auc'
    )
    
    cv_scores_pr = cross_val_score(
        xgb_model, X_train, y_train,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='average_precision'  # PR-AUC metric
    )
    
    logger.info(f"Cross-validation ROC-AUC scores: {cv_scores_roc}")
    logger.info(f"Mean CV ROC-AUC: {cv_scores_roc.mean():.4f} (+/- {cv_scores_roc.std():.4f})")
    
    logger.info(f"\nCross-validation PR-AUC scores: {cv_scores_pr}")
    logger.info(f"Mean CV PR-AUC: {cv_scores_pr.mean():.4f} (+/- {cv_scores_pr.std():.4f})")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logger.info("\nTop 15 Most Important Features (by XGBoost importance):")
    logger.info(feature_importance.head(15).to_string())
    
    return {
        'model': xgb_model,
        'cv_scores': cv_scores_roc,
        'cv_scores_pr': cv_scores_pr, 
        'feature_importance': feature_importance,
        'params': xgb_params
    }


@asset(group_name=group_name)
def evaluate_logistic_regression(logistic_regression_model, X_train, X_test, y_train, y_test):
    """
    Evaluate Logistic Regression model on both training and test sets
    """
    logger.info("Evaluating Logistic Regression model...")
    
    lr_model = logistic_regression_model['model']
    
    # ============ TRAINING SET EVALUATION ============
    logger.info("\n" + "="*60)
    logger.info("TRAINING SET PERFORMANCE")
    logger.info("="*60)
    
    y_train_pred = lr_model.predict(X_train)
    y_train_pred_proba = lr_model.predict_proba(X_train)[:, 1]
    
    train_roc_auc = roc_auc_score(y_train, y_train_pred_proba)
    train_avg_precision = average_precision_score(y_train, y_train_pred_proba)
    train_f1 = f1_score(y_train, y_train_pred)
    
    logger.info(f"ROC-AUC Score: {train_roc_auc:.4f}")
    logger.info(f"Average Precision Score: {train_avg_precision:.4f}")
    logger.info(f"F1 Score: {train_f1:.4f}")
    
    logger.info("\nClassification Report:")
    logger.info(classification_report(y_train, y_train_pred, target_names=['No Upsell', 'Upsell']))
    
    train_cm = confusion_matrix(y_train, y_train_pred)
    logger.info("\nConfusion Matrix:")
    logger.info(train_cm)
    
    # ============ TEST SET EVALUATION ============
    logger.info("\n" + "="*60)
    logger.info("TEST SET PERFORMANCE")
    logger.info("="*60)
    
    y_pred = lr_model.predict(X_test)
    y_pred_proba = lr_model.predict_proba(X_test)[:, 1]
    
    test_roc_auc = roc_auc_score(y_test, y_pred_proba)
    test_avg_precision = average_precision_score(y_test, y_pred_proba)
    test_f1 = f1_score(y_test, y_pred)
    
    logger.info(f"ROC-AUC Score: {test_roc_auc:.4f}")
    logger.info(f"Average Precision Score: {test_avg_precision:.4f}")
    logger.info(f"F1 Score: {test_f1:.4f}")
    
    logger.info("\nClassification Report:")
    logger.info(classification_report(y_test, y_pred, target_names=['No Upsell', 'Upsell']))
    
    test_cm = confusion_matrix(y_test, y_pred)
    logger.info("\nConfusion Matrix:")
    logger.info(test_cm)
    logger.info(f"True Negatives: {test_cm[0,0]}, False Positives: {test_cm[0,1]}")
    logger.info(f"False Negatives: {test_cm[1,0]}, True Positives: {test_cm[1,1]}")
    
    # ============ OVERFITTING CHECK ============
    logger.info("\n" + "="*60)
    logger.info("OVERFITTING ANALYSIS")
    logger.info("="*60)
    
    auc_gap = train_roc_auc - test_roc_auc
    f1_gap = train_f1 - test_f1
    
    logger.info(f"Train ROC-AUC: {train_roc_auc:.4f}")
    logger.info(f"Test ROC-AUC:  {test_roc_auc:.4f}")
    logger.info(f"AUC Gap:       {auc_gap:.4f}")
    
    if auc_gap > 0.05:
        logger.warning("Potential overfitting: Train AUC significantly higher than test AUC")
    else:
        logger.info("Model generalizes well: Minimal train-test gap")
    
    # Business metrics
    precision = test_cm[1,1] / (test_cm[1,1] + test_cm[0,1]) if (test_cm[1,1] + test_cm[0,1]) > 0 else 0
    recall = test_cm[1,1] / (test_cm[1,1] + test_cm[1,0]) if (test_cm[1,1] + test_cm[1,0]) > 0 else 0
    
    logger.info("\n" + "="*60)
    logger.info("BUSINESS INTERPRETATION (TEST SET)")
    logger.info("="*60)
    logger.info(f"Precision: {precision:.2%} - Of customers we target, {precision:.2%} will actually upsell")
    logger.info(f"Recall: {recall:.2%} - We capture {recall:.2%} of all potential upsell customers")
    
    return {
        'train_roc_auc': train_roc_auc,
        'train_avg_precision': train_avg_precision,
        'train_f1_score': train_f1,
        'roc_auc': test_roc_auc,
        'avg_precision': test_avg_precision,
        'f1_score': test_f1,
        'predictions': y_pred,
        'probabilities': y_pred_proba,
        'confusion_matrix': test_cm,
        'auc_gap': auc_gap
    }


@asset(group_name=group_name)
def evaluate_xgboost(xgboost_model, X_train, X_test, y_train, y_test):
    """
    Evaluate XGBoost model on both training and test sets
    """
    logger.info("Evaluating XGBoost model...")
    
    xgb_model = xgboost_model['model']
    
    # ============ TRAINING SET EVALUATION ============
    logger.info("\n" + "="*60)
    logger.info("TRAINING SET PERFORMANCE")
    logger.info("="*60)
    
    y_train_pred = xgb_model.predict(X_train)
    y_train_pred_proba = xgb_model.predict_proba(X_train)[:, 1]
    
    train_roc_auc = roc_auc_score(y_train, y_train_pred_proba)
    train_avg_precision = average_precision_score(y_train, y_train_pred_proba)
    train_f1 = f1_score(y_train, y_train_pred)
    
    logger.info(f"ROC-AUC Score: {train_roc_auc:.4f}")
    logger.info(f"Average Precision Score: {train_avg_precision:.4f}")
    logger.info(f"F1 Score: {train_f1:.4f}")
    
    logger.info("\nClassification Report:")
    logger.info(classification_report(y_train, y_train_pred, target_names=['No Upsell', 'Upsell']))
    
    train_cm = confusion_matrix(y_train, y_train_pred)
    logger.info("\nConfusion Matrix:")
    logger.info(train_cm)
    
    # ============ TEST SET EVALUATION ============
    logger.info("\n" + "="*60)
    logger.info("TEST SET PERFORMANCE")
    logger.info("="*60)
    
    y_pred = xgb_model.predict(X_test)
    y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
    
    test_roc_auc = roc_auc_score(y_test, y_pred_proba)
    test_avg_precision = average_precision_score(y_test, y_pred_proba)
    test_f1 = f1_score(y_test, y_pred)
    
    logger.info(f"ROC-AUC Score: {test_roc_auc:.4f}")
    logger.info(f"Average Precision Score: {test_avg_precision:.4f}")
    logger.info(f"F1 Score: {test_f1:.4f}")
    
    logger.info("\nClassification Report:")
    logger.info(classification_report(y_test, y_pred, target_names=['No Upsell', 'Upsell']))
    
    test_cm = confusion_matrix(y_test, y_pred)
    logger.info("\nConfusion Matrix:")
    logger.info(test_cm)
    logger.info(f"True Negatives: {test_cm[0,0]}, False Positives: {test_cm[0,1]}")
    logger.info(f"False Negatives: {test_cm[1,0]}, True Positives: {test_cm[1,1]}")
    
    # ============ OVERFITTING CHECK ============
    logger.info("\n" + "="*60)
    logger.info("OVERFITTING ANALYSIS")
    logger.info("="*60)
    
    auc_gap = train_roc_auc - test_roc_auc
    f1_gap = train_f1 - test_f1
    
    logger.info(f"Train ROC-AUC: {train_roc_auc:.4f}")
    logger.info(f"Test ROC-AUC:  {test_roc_auc:.4f}")
    logger.info(f"AUC Gap:       {auc_gap:.4f}")
    
    if auc_gap > 0.05:
        logger.warning("Potential overfitting: Train AUC significantly higher than test AUC")
    else:
        logger.info("Model generalizes well: Minimal train-test gap")
    
    # Business metrics
    precision = test_cm[1,1] / (test_cm[1,1] + test_cm[0,1]) if (test_cm[1,1] + test_cm[0,1]) > 0 else 0
    recall = test_cm[1,1] / (test_cm[1,1] + test_cm[1,0]) if (test_cm[1,1] + test_cm[1,0]) > 0 else 0
    
    logger.info("\n" + "="*60)
    logger.info("BUSINESS INTERPRETATION (TEST SET)")
    logger.info("="*60)
    logger.info(f"Precision: {precision:.2%} - Of customers we target, {precision:.2%} will actually upsell")
    logger.info(f"Recall: {recall:.2%} - We capture {recall:.2%} of all potential upsell customers")
    
    return {
        'train_roc_auc': train_roc_auc,
        'train_avg_precision': train_avg_precision,
        'train_f1_score': train_f1,
        'roc_auc': test_roc_auc,
        'avg_precision': test_avg_precision,
        'f1_score': test_f1,
        'predictions': y_pred,
        'probabilities': y_pred_proba,
        'confusion_matrix': test_cm,
        'auc_gap': auc_gap
    }


@asset(group_name=group_name)
def model_comparison(evaluate_logistic_regression, evaluate_xgboost):
    """
    Compare performance of both models on training and test sets
    """
    logger.info("\n" + "="*60)
    logger.info("MODEL COMPARISON - TRAINING VS TEST")
    logger.info("="*60)
    
    # Create comprehensive comparison table
    comparison = pd.DataFrame({
        'LR_Train': {
            'ROC-AUC': evaluate_logistic_regression['train_roc_auc'],
            'Avg Precision': evaluate_logistic_regression['train_avg_precision'],
            'F1 Score': evaluate_logistic_regression['train_f1_score']
        },
        'LR_Test': {
            'ROC-AUC': evaluate_logistic_regression['roc_auc'],
            'Avg Precision': evaluate_logistic_regression['avg_precision'],
            'F1 Score': evaluate_logistic_regression['f1_score']
        },
        'XGB_Train': {
            'ROC-AUC': evaluate_xgboost['train_roc_auc'],
            'Avg Precision': evaluate_xgboost['train_avg_precision'],
            'F1 Score': evaluate_xgboost['train_f1_score']
        },
        'XGB_Test': {
            'ROC-AUC': evaluate_xgboost['roc_auc'],
            'Avg Precision': evaluate_xgboost['avg_precision'],
            'F1 Score': evaluate_xgboost['f1_score']
        }
    }).T
    
    logger.info("\n📊 COMPLETE PERFORMANCE COMPARISON:")
    logger.info("\n" + comparison.to_string())
    
    # Performance gaps
    logger.info("\n" + "="*60)
    logger.info("TRAIN-TEST PERFORMANCE GAPS")
    logger.info("="*60)
    logger.info(f"Logistic Regression AUC Gap: {evaluate_logistic_regression['auc_gap']:.4f}")
    logger.info(f"XGBoost AUC Gap:            {evaluate_xgboost['auc_gap']:.4f}")
    
    # Test set comparison
    logger.info("\n" + "="*60)
    logger.info("TEST SET MODEL COMPARISON")
    logger.info("="*60)
    
    # Determine winner
    if evaluate_xgboost['roc_auc'] > evaluate_logistic_regression['roc_auc']:
        winner = "XGBoost"
        diff = evaluate_xgboost['roc_auc'] - evaluate_logistic_regression['roc_auc']
        logger.info(f"XGBoost achieves higher ROC-AUC score (+{diff:.4f})")
    else:
        winner = "Logistic Regression"
        diff = evaluate_logistic_regression['roc_auc'] - evaluate_xgboost['roc_auc']
        logger.info(f"Logistic Regression achieves higher ROC-AUC score (+{diff:.4f})")
    
    logger.info("\n" + "="*60)
    logger.info("RECOMMENDATIONS")
    logger.info("="*60)
    logger.info("For MAXIMUM PREDICTIVE PERFORMANCE:")
    logger.info(f"   → Use {winner} (higher test set ROC-AUC)")
    logger.info("\nFor INTERPRETABILITY & STAKEHOLDER COMMUNICATION:")
    logger.info("   → Use Logistic Regression (clear feature coefficients)")
    logger.info("\nFor PRODUCTION DEPLOYMENT:")
    logger.info("   → Consider ensemble of both models or stacking")
    logger.info("\nFor OVERFITTING CONCERNS:")
    if evaluate_logistic_regression['auc_gap'] < evaluate_xgboost['auc_gap']:
        logger.info("   → Logistic Regression shows better generalization")
    else:
        logger.info("   → XGBoost shows better generalization")
    
    return {
        'comparison_table': comparison,
        'best_model': winner,
        'test_auc_difference': abs(diff)
    }