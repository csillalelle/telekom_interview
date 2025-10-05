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
    
    # Scale features (important for logistic regression)
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
    
    # Cross-validation to assess model stability
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
    Train an XGBoost model
    
    Why XGBoost?
    - Handles non-linear relationships: Captures complex patterns in data
    - Feature interactions: Automatically learns interactions between features
    - Robust to outliers: Tree-based models are less sensitive to extreme values
    - High performance: Often achieves better predictive accuracy than linear models
    - Built-in feature importance: Provides multiple ways to rank features
    """
    logger.info("Training XGBoost model...")
    
    # Calculate scale_pos_weight for class imbalance
    # This parameter helps XGBoost handle imbalanced datasets
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    logger.info(f"Scale pos weight (for class imbalance): {scale_pos_weight:.2f}")
    
    # XGBoost parameters
    xgb_params = {
        'max_depth': 5,                    # Maximum depth of trees (prevents overfitting)
        'learning_rate': 0.1,              # Step size shrinkage (lower = more robust)
        'n_estimators': 100,               # Number of boosting rounds
        'objective': 'binary:logistic',   # Binary classification
        'scale_pos_weight': scale_pos_weight,  # Handle class imbalance
        'subsample': 0.8,                  # Subsample ratio of training instances
        'colsample_bytree': 0.8,          # Subsample ratio of features
        'random_state': 42,
        'eval_metric': 'auc'              # Evaluation metric
    }
    
    # Train the model
    xgb_model = xgb.XGBClassifier(**xgb_params)
    xgb_model.fit(X_train, y_train, verbose=False)
    
    # Cross-validation
    cv_scores = cross_val_score(
        xgb_model, X_train, y_train,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='roc_auc'
    )
    
    logger.info(f"Cross-validation ROC-AUC scores: {cv_scores}")
    logger.info(f"Mean CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logger.info("\nTop 15 Most Important Features (by XGBoost importance):")
    logger.info(feature_importance.head(15).to_string())
    
    return {
        'model': xgb_model,
        'cv_scores': cv_scores,
        'feature_importance': feature_importance,
        'params': xgb_params
    }


@asset(group_name=group_name)
def evaluate_logistic_regression(logistic_regression_model, X_test, y_test):
    """
    Evaluate Logistic Regression model on test set
    """
    logger.info("Evaluating Logistic Regression model...")
    
    lr_model = logistic_regression_model['model']
    
    # Predictions
    y_pred = lr_model.predict(X_test)
    y_pred_proba = lr_model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    avg_precision = average_precision_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)
    
    logger.info("\n" + "="*60)
    logger.info("LOGISTIC REGRESSION - TEST SET PERFORMANCE")
    logger.info("="*60)
    logger.info(f"ROC-AUC Score: {roc_auc:.4f}")
    logger.info(f"Average Precision Score: {avg_precision:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    
    logger.info("\nClassification Report:")
    logger.info(classification_report(y_test, y_pred, target_names=['No Upsell', 'Upsell']))
    
    logger.info("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    logger.info(cm)
    logger.info(f"True Negatives: {cm[0,0]}, False Positives: {cm[0,1]}")
    logger.info(f"False Negatives: {cm[1,0]}, True Positives: {cm[1,1]}")
    
    # Business metrics
    precision = cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0
    recall = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
    
    logger.info("\nBusiness Interpretation:")
    logger.info(f"Precision: {precision:.2%} - Of customers we target, {precision:.2%} will actually upsell")
    logger.info(f"Recall: {recall:.2%} - We capture {recall:.2%} of all potential upsell customers")
    
    return {
        'roc_auc': roc_auc,
        'avg_precision': avg_precision,
        'f1_score': f1,
        'predictions': y_pred,
        'probabilities': y_pred_proba,
        'confusion_matrix': cm
    }


@asset(group_name=group_name)
def evaluate_xgboost(xgboost_model, X_test, y_test):
    """
    Evaluate XGBoost model on test set
    """
    logger.info("Evaluating XGBoost model...")
    
    xgb_model = xgboost_model['model']
    
    # Predictions
    y_pred = xgb_model.predict(X_test)
    y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    avg_precision = average_precision_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)
    
    logger.info("\n" + "="*60)
    logger.info("XGBOOST - TEST SET PERFORMANCE")
    logger.info("="*60)
    logger.info(f"ROC-AUC Score: {roc_auc:.4f}")
    logger.info(f"Average Precision Score: {avg_precision:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    
    logger.info("\nClassification Report:")
    logger.info(classification_report(y_test, y_pred, target_names=['No Upsell', 'Upsell']))
    
    logger.info("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    logger.info(cm)
    logger.info(f"True Negatives: {cm[0,0]}, False Positives: {cm[0,1]}")
    logger.info(f"False Negatives: {cm[1,0]}, True Positives: {cm[1,1]}")
    
    # Business metrics
    precision = cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0
    recall = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
    
    logger.info("\nBusiness Interpretation:")
    logger.info(f"Precision: {precision:.2%} - Of customers we target, {precision:.2%} will actually upsell")
    logger.info(f"Recall: {recall:.2%} - We capture {recall:.2%} of all potential upsell customers")
    
    return {
        'roc_auc': roc_auc,
        'avg_precision': avg_precision,
        'f1_score': f1,
        'predictions': y_pred,
        'probabilities': y_pred_proba,
        'confusion_matrix': cm
    }


@asset(group_name=group_name)
def model_comparison(evaluate_logistic_regression, evaluate_xgboost):
    """
    Compare performance of both models
    """
    logger.info("\n" + "="*60)
    logger.info("MODEL COMPARISON")
    logger.info("="*60)
    
    comparison = pd.DataFrame({
        'Logistic Regression': {
            'ROC-AUC': evaluate_logistic_regression['roc_auc'],
            'Avg Precision': evaluate_logistic_regression['avg_precision'],
            'F1 Score': evaluate_logistic_regression['f1_score']
        },
        'XGBoost': {
            'ROC-AUC': evaluate_xgboost['roc_auc'],
            'Avg Precision': evaluate_xgboost['avg_precision'],
            'F1 Score': evaluate_xgboost['f1_score']
        }
    }).T
    
    logger.info("\n" + comparison.to_string())
    
    # Determine winner
    if evaluate_xgboost['roc_auc'] > evaluate_logistic_regression['roc_auc']:
        logger.info("\nXGBoost achieves higher ROC-AUC score")
    else:
        logger.info("\nLogistic Regression achieves higher ROC-AUC score")
    
    logger.info("\nRecommendation:")
    logger.info("- Use XGBoost for maximum predictive performance")
    logger.info("- Use Logistic Regression for interpretability and stakeholder communication")
    logger.info("- Consider ensemble or stacking both models for production")
    
    return comparison