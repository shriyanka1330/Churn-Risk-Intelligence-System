import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

import shap

# Set styling for plots
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.titlesize': 18
})

def build_features(df):
    """
    Perform feature engineering on the bank churn dataset.
    """
    df = df.copy()
    
    # 1. Balance-to-Salary Ratio
    df['Balance_Salary_Ratio'] = df['Balance'] / (df['EstimatedSalary'] + 1e-5)
    
    # 2. Product Density Indicator (Products per year of tenure)
    df['Product_Density'] = df['NumOfProducts'] / (df['Tenure'] + 1)
    
    # 3. Engagement-Product Interaction
    df['Active_Product_Interaction'] = df['IsActiveMember'] * df['NumOfProducts']
    
    # 4. Age-Tenure Interaction
    df['Age_Tenure_Interaction'] = df['Age'] * df['Tenure']
    
    return df

def main():
    print("Step 1: Loading Dataset...")
    # Load dataset
    data_path = 'European_Bank.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    
    raw_df = pd.read_csv(data_path)
    print(f"Dataset loaded successfully with shape: {raw_df.shape}")
    
    # Preprocessing: Drop non-informative features
    # Check if 'Year' is in the dataset
    cols_to_drop = ['CustomerId', 'Surname']
    if 'Year' in raw_df.columns:
        cols_to_drop.append('Year')
    
    print(f"Dropping non-informative columns: {cols_to_drop}")
    df_cleaned = raw_df.drop(columns=cols_to_drop)
    
    print("Step 2: Feature Engineering...")
    df_featured = build_features(df_cleaned)
    print(f"Features created. New shape: {df_featured.shape}")
    
    # Step 3: Train-Test Split
    X = df_featured.drop(columns=['Exited'])
    y = df_featured['Exited']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Split data into train (size={len(X_train)}) and test (size={len(X_test)}) with stratified target distribution.")
    
    # Step 4: Define preprocessing pipelines
    categorical_cols = ['Geography', 'Gender']
    numerical_cols = [col for col in X.columns if col not in categorical_cols]
    
    print(f"Categorical features: {categorical_cols}")
    print(f"Numerical features: {numerical_cols}")
    
    # One-hot encoder without drop='first' is safer for Streamlit handling
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ]
    )
    
    # Step 5: Initialize Models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
        'Decision Tree': DecisionTreeClassifier(max_depth=6, random_state=42, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42, class_weight='balanced'),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=4, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.1, random_state=42, use_label_encoder=False, eval_metric='logloss')
    }
    
    results = {}
    trained_pipelines = {}
    
    print("Step 6: Training and Evaluating Models...")
    plt.figure(figsize=(10, 8))
    
    for name, model in models.items():
        print(f"  Training {name}...")
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        # Fit model
        pipeline.fit(X_train, y_train)
        trained_pipelines[name] = pipeline
        
        # Predictions
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        
        y_train_pred = pipeline.predict(X_train)
        y_train_proba = pipeline.predict_proba(X_train)[:, 1]
        
        # Metrics
        metrics = {
            'Train Accuracy': float(accuracy_score(y_train, y_train_pred)),
            'Test Accuracy': float(accuracy_score(y_test, y_pred)),
            'Train Precision': float(precision_score(y_train, y_train_pred)),
            'Test Precision': float(precision_score(y_test, y_pred)),
            'Train Recall': float(recall_score(y_train, y_train_pred)),
            'Test Recall': float(recall_score(y_test, y_pred)),
            'Train F1': float(f1_score(y_train, y_train_pred)),
            'Test F1': float(f1_score(y_test, y_pred)),
            'Train ROC-AUC': float(roc_auc_score(y_train, y_train_proba)),
            'Test ROC-AUC': float(roc_auc_score(y_test, y_pred_proba))
        }
        results[name] = metrics
        
        print(f"    Test Accuracy: {metrics['Test Accuracy']:.4f} | Test F1: {metrics['Test F1']:.4f} | Test ROC-AUC: {metrics['Test ROC-AUC']:.4f}")
        
        # Plot ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {metrics['Test ROC-AUC']:.3f})")
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('roc_curves.png', dpi=300)
    plt.close()
    print("Saved ROC curves comparison plot to 'roc_curves.png'")
    
    # Save metrics table
    metrics_df = pd.DataFrame(results).T
    metrics_df.to_csv('model_comparison_metrics.csv')
    print("\nModel Comparison Table:")
    print(metrics_df[['Test Accuracy', 'Test Precision', 'Test Recall', 'Test F1', 'Test ROC-AUC']])
    
    # Step 7: Select Best Model
    # We will select the best model based on Test ROC-AUC
    best_model_name = metrics_df['Test ROC-AUC'].idxmax()
    print(f"\nBest Model Selected based on Test ROC-AUC: {best_model_name}")
    best_pipeline = trained_pipelines[best_model_name]
    
    # Save the pipeline
    joblib.dump(best_pipeline, 'best_model.joblib')
    print(f"Saved the best pipeline model to 'best_model.joblib'")
    
    # Save metadata about features and metrics
    metadata = {
        'best_model_name': best_model_name,
        'features': {
            'numerical': numerical_cols,
            'categorical': categorical_cols
        },
        'best_metrics': results[best_model_name]
    }
    with open('model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)
        
    # Step 8: Explainability (Feature Importance & SHAP)
    print("\nStep 8: Model Explainability Analysis...")
    
    # Get feature names from preprocessor
    ohe = best_pipeline.named_steps['preprocessor'].named_transformers_['cat']
    ohe_features = list(ohe.get_feature_names_out(categorical_cols))
    all_feature_names = numerical_cols + ohe_features
    
    # Feature Importance Plot (for tree-based models)
    classifier = best_pipeline.named_steps['classifier']
    
    if hasattr(classifier, 'feature_importances_'):
        importances = classifier.feature_importances_
        feat_imp = pd.Series(importances, index=all_feature_names).sort_values(ascending=True)
        
        plt.figure(figsize=(10, 6))
        # Use premium colors (cool teal/blue palette)
        colors = sns.color_palette("viridis", len(feat_imp))
        feat_imp.plot(kind='barh', color=colors)
        plt.xlabel('Importance Score')
        plt.ylabel('Feature')
        plt.title(f'Feature Importance: {best_model_name}')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300)
        plt.close()
        print(f"Saved feature importance plot to 'feature_importance.png'")
    elif hasattr(classifier, 'coef_'):
        coefs = classifier.coef_[0]
        feat_coefs = pd.Series(coefs, index=all_feature_names).sort_values(key=abs, ascending=True)
        
        plt.figure(figsize=(10, 6))
        colors = ['red' if c < 0 else 'blue' for c in feat_coefs]
        feat_coefs.plot(kind='barh', color=colors)
        plt.xlabel('Coefficient Value (Impact)')
        plt.ylabel('Feature')
        plt.title(f'Feature Coefficients: {best_model_name}')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300)
        plt.close()
        print(f"Saved feature coefficients plot to 'feature_importance.png'")
        
    # SHAP Explainer
    print("  Generating SHAP values (using a sample of test data for efficiency)...")
    # Transform test features
    X_test_transformed = best_pipeline.named_steps['preprocessor'].transform(X_test)
    X_test_transformed_df = pd.DataFrame(X_test_transformed, columns=all_feature_names)
    
    # Use a background sample of 500 records for the explainer to run quickly and avoid high memory usage
    shap_sample = X_test_transformed_df.sample(500, random_state=42)
    
    # Initialize explainer depending on model type
    if best_model_name in ['Random Forest', 'Gradient Boosting', 'Decision Tree', 'XGBoost']:
        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(shap_sample)
    else:
        explainer = shap.Explainer(classifier, shap_sample)
        shap_values = explainer(shap_sample).values
    
    # Handle SHAP shape differences between packages and models
    # Random Forest in scikit-learn returns a list of arrays for binary classification, index 1 is class 1 (churned)
    if isinstance(shap_values, list):
        # TreeExplainer for sklearn RF returns [shap_values_class0, shap_values_class1]
        shap_values_to_plot = shap_values[1]
    elif len(shap_values.shape) == 3:
        # Some SHAP versions return shape (samples, features, classes)
        shap_values_to_plot = shap_values[:, :, 1]
    else:
        shap_values_to_plot = shap_values
        
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_to_plot, shap_sample, show=False)
    plt.title("SHAP Summary Plot (Model Churn Drivers)", fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig('shap_summary.png', dpi=300)
    plt.close()
    print("Saved SHAP summary plot to 'shap_summary.png'")
    
    # Save a small sample of transformed and raw data for what-if scenarios and testing
    sample_customers = X_test.head(10).copy()
    sample_customers['Exited'] = y_test.head(10).values
    sample_customers.to_csv('sample_customers.csv', index=False)
    print("Saved a test sample of 10 customers to 'sample_customers.csv'")

if __name__ == '__main__':
    main()
