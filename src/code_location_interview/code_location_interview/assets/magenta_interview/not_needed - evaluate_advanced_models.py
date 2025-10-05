# """
# Evaluation assets for advanced XGBoost models
# """

# import numpy as np
# import pandas as pd
# from sklearn.metrics import (
#     roc_auc_score, classification_report, confusion_matrix,
#     precision_score, recall_score, f1_score, average_precision_score
# )
# from dagster import asset, get_dagster_logger

# logger = get_dagster_logger(__name__)
# group_name = "training"


# def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
#     """
#     Helper function to evaluate any model
#     """
#     logger.info("\n" + "="*60)
#     logger.info(f"{model_name.upper()} EVALUATION")
#     logger.info("="*60)
    
#     # Training set evaluation
#     train_pred_proba = model.predict_proba(X_train)[:, 1]
#     train_pred = model.predict(X_train)
    
#     train_auc = roc_auc_score(y_train, train_pred_proba)
#     train_precision = precision_score(y_train, train_pred)
#     train_recall = recall_score(y_train, train_pred)
#     train_f1 = f1_score(y_train, train_pred)
#     train_cm = confusion_matrix(y_train, train_pred)
    
#     logger.info("\nTRAINING SET:")
#     logger.info(f"ROC-AUC:    {train_auc:.4f}")
#     logger.info(f"Precision:  {train_precision:.4f}")
#     logger.info(f"Recall:     {train_recall:.4f}")
#     logger.info(f"F1 Score:   {train_f1:.4f}")
    
#     # Test set evaluation
#     test_pred_proba = model.predict_proba(X_test)[:, 1]
#     test_pred = model.predict(X_test)
    
#     test_auc = roc_auc_score(y_test, test_pred_proba)
#     test_avg_precision = average_precision_score(y_test, test_pred_proba)
#     test_precision = precision_score(y_test, test_pred)
#     test_recall = recall_score(y_test, test_pred)
#     test_f1 = f1_score(y_test, test_pred)
#     test_cm = confusion_matrix(y_test, test_pred)
    
#     logger.info("\nTEST SET:")
#     logger.info(f"ROC-AUC:    {test_auc:.4f}")
#     logger.info(f"Precision:  {test_precision:.4f}")
#     logger.info(f"Recall:     {test_recall:.4f}")
#     logger.info(f"F1 Score:   {test_f1:.4f}")
#     logger.info(f"Avg Prec:   {test_avg_precision:.4f}")
    
#     logger.info("\nConfusion Matrix:")
#     logger.info(test_cm)
    
#     # Overfitting analysis
#     auc_gap = train_auc - test_auc
#     logger.info("\nGENERALIZATION:")
#     logger.info(f"Train AUC: {train_auc:.4f}")
#     logger.info(f"Test AUC:  {test_auc:.4f}")
#     logger.info(f"AUC Gap:   {auc_gap:.4f}")
    
#     if auc_gap < 0.03:
#         logger.info("Excellent generalization!")
#     elif auc_gap < 0.06:
#         logger.info("Good generalization")
#     elif auc_gap < 0.10:
#         logger.info("Acceptable generalization")
#     else:
#         logger.warning("Overfitting detected")
    
#     return {
#         'train_auc': train_auc,
#         'train_precision': train_precision,
#         'train_recall': train_recall,
#         'train_f1': train_f1,
#         'test_auc': test_auc,
#         'test_avg_precision': test_avg_precision,
#         'test_precision': test_precision,
#         'test_recall': test_recall,
#         'test_f1': test_f1,
#         'auc_gap': auc_gap,
#         'confusion_matrix': test_cm,
#         'predictions': test_pred,
#         'probabilities': test_pred_proba
#     }


# @asset(group_name=group_name)
# def evaluate_early_stopping(xgboost_model_early_stopping, X_train, X_test, y_train, y_test):
#     """Evaluate early stopping model"""
#     model = xgboost_model_early_stopping['model']
#     return evaluate_model(model, X_train, X_test, y_train, y_test, "Early Stopping XGBoost")


# @asset(group_name=group_name)
# def evaluate_grid_search(xgboost_model_grid_search, X_train, X_test, y_train, y_test):
#     """Evaluate grid search model"""
#     model = xgboost_model_grid_search['model']
#     return evaluate_model(model, X_train, X_test, y_train, y_test, "Grid Search XGBoost")


# @asset(group_name=group_name)
# def evaluate_random_search(xgboost_model_random_search, X_train, X_test, y_train, y_test):
#     """Evaluate random search model"""
#     model = xgboost_model_random_search['model']
#     return evaluate_model(model, X_train, X_test, y_train, y_test, "Random Search XGBoost")


# @asset(group_name=group_name)
# def evaluate_optimized(xgboost_model_optimized, X_train, X_test, y_train, y_test):
#     """Evaluate optimized (grid search + early stopping) model"""
#     model = xgboost_model_optimized['model']
#     return evaluate_model(model, X_train, X_test, y_train, y_test, "Optimized XGBoost")


# @asset(group_name=group_name)
# def compare_all_xgboost_models(
#     evaluate_xgboost,  # Original regularized model
#     evaluate_early_stopping,
#     evaluate_grid_search,
#     evaluate_optimized
# ):
#     """
#     Comprehensive comparison of all XGBoost model variants
#     """
#     logger.info("\n" + "="*60)
#     logger.info("COMPREHENSIVE MODEL COMPARISON")
#     logger.info("="*60)
    
#     models = {
#         'Regularized (Original)': evaluate_xgboost,
#         'Early Stopping': evaluate_early_stopping,
#         'Grid Search': evaluate_grid_search,
#         'Optimized (Grid+ES)': evaluate_optimized
#     }
    
#     # Create comparison table
#     comparison_data = {}
#     for name, results in models.items():
#         comparison_data[name] = {
#             'Train AUC': results['train_auc'],
#             'Test AUC': results['test_auc'],
#             'AUC Gap': results['auc_gap'],
#             'Test Precision': results['test_precision'],
#             'Test Recall': results['test_recall'],
#             'Test F1': results['test_f1']
#         }
    
#     comparison_df = pd.DataFrame(comparison_data).T
    
#     logger.info("\nPERFORMANCE COMPARISON:")
#     logger.info("\n" + comparison_df.to_string())
    
#     # Find best models
#     best_test_auc = comparison_df['Test AUC'].idxmax()
#     best_generalization = comparison_df['AUC Gap'].idxmin()
#     best_f1 = comparison_df['Test F1'].idxmax()
    
#     logger.info("\n" + "="*60)
#     logger.info("BEST MODELS BY METRIC")
#     logger.info("="*60)
#     logger.info(f"\nBest Test AUC: {best_test_auc}")
#     logger.info(f"   Score: {comparison_df.loc[best_test_auc, 'Test AUC']:.4f}")
    
#     logger.info(f"\nBest Generalization (smallest gap): {best_generalization}")
#     logger.info(f"   Gap: {comparison_df.loc[best_generalization, 'AUC Gap']:.4f}")
    
#     logger.info(f"\nBest F1 Score: {best_f1}")
#     logger.info(f"   Score: {comparison_df.loc[best_f1, 'Test F1']:.4f}")
    
#     # Overall recommendation
#     logger.info("\n" + "="*60)
#     logger.info("RECOMMENDATION")
#     logger.info("="*60)
    
#     # Score each model (test AUC weight 50%, generalization 30%, F1 20%)
#     comparison_df['Score'] = (
#         0.50 * (comparison_df['Test AUC'] / comparison_df['Test AUC'].max()) +
#         0.30 * (1 - comparison_df['AUC Gap'] / comparison_df['AUC Gap'].max()) +
#         0.20 * (comparison_df['Test F1'] / comparison_df['Test F1'].max())
#     )
    
#     best_overall = comparison_df['Score'].idxmax()
    
#     logger.info(f"\nBEST OVERALL MODEL: {best_overall}")
#     logger.info(f"   Test AUC: {comparison_df.loc[best_overall, 'Test AUC']:.4f}")
#     logger.info(f"   AUC Gap:  {comparison_df.loc[best_overall, 'AUC Gap']:.4f}")
#     logger.info(f"   Test F1:  {comparison_df.loc[best_overall, 'Test F1']:.4f}")
    
#     logger.info("\nPRODUCTION RECOMMENDATION:")
#     if best_overall == 'Optimized (Grid+ES)':
#         logger.info("   Use Optimized model (Grid Search + Early Stopping)")
#         logger.info("   This combines hyperparameter tuning with overfitting prevention")
#         logger.info("   Best balance of performance and generalization")
#     elif best_overall == 'Grid Search':
#         logger.info("   Use Grid Search model")
#         logger.info("   Systematically optimized hyperparameters")
#     elif best_overall == 'Early Stopping':
#         logger.info("   Use Early Stopping model")
#         logger.info("   Automatically prevents overfitting")
#     else:
#         logger.info("   Use Regularized model")
#         logger.info("   Simple and effective regularization")
    
#     # Performance improvements over original
#     if 'Regularized (Original)' in comparison_data:
#         original_test_auc = comparison_data['Regularized (Original)']['Test AUC']
#         best_test_auc_value = comparison_df.loc[best_overall, 'Test AUC']
#         improvement = ((best_test_auc_value - original_test_auc) / original_test_auc) * 100
        
#         logger.info(f"\nIMPROVEMENT OVER BASELINE:")
#         logger.info(f"   Test AUC improvement: {improvement:+.2f}%")
#         logger.info(f"   From {original_test_auc:.4f} to {best_test_auc_value:.4f}")
    
#     return {
#         'comparison_table': comparison_df,
#         'best_overall': best_overall,
#         'best_test_auc': best_test_auc,
#         'best_generalization': best_generalization
#     }