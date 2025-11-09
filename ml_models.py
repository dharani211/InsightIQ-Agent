import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

class MLAnalyzer:
    """Machine Learning analysis toolkit"""
    
    @staticmethod
    def linear_regression(df, features, target, test_size=0.2):
        """Perform linear regression with visualization"""
        try:
            # Prepare data
            X = df[features].dropna()
            y = df[target].loc[X.index]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            # Train model
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            
            # Metrics
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            
            # Coefficients
            coefficients = {
                feature: round(float(coef), 4)
                for feature, coef in zip(features, model.coef_)
            }
            coefficients['intercept'] = round(float(model.intercept_), 4)
            
            # Create plots
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Plot 1: Actual vs Predicted
            axes[0].scatter(y_test, y_pred, alpha=0.6, edgecolors='k', s=60)
            axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                        'r--', lw=2, label='Perfect Prediction')
            axes[0].set_xlabel('Actual Values', fontsize=12, fontweight='bold')
            axes[0].set_ylabel('Predicted Values', fontsize=12, fontweight='bold')
            axes[0].set_title(f'Actual vs Predicted (R² = {r2:.4f})', fontsize=14, fontweight='bold')
            axes[0].legend(fontsize=10)
            axes[0].grid(alpha=0.3)
            
            # Plot 2: Residuals
            residuals = y_test - y_pred
            axes[1].scatter(y_pred, residuals, alpha=0.6, edgecolors='k', s=60)
            axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
            axes[1].set_xlabel('Predicted Values', fontsize=12, fontweight='bold')
            axes[1].set_ylabel('Residuals', fontsize=12, fontweight='bold')
            axes[1].set_title('Residual Plot', fontsize=14, fontweight='bold')
            axes[1].grid(alpha=0.3)
            
            plt.tight_layout()
            
            return {
                'r2_score': round(r2, 4),
                'rmse': round(rmse, 4),
                'mae': round(mae, 4),
                'coefficients': coefficients,
                'plot': fig
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    @staticmethod
    def classification(df, features, target, algorithm='Logistic Regression', test_size=0.2):
        """Perform classification with confusion matrix"""
        try:
            # Prepare data
            X = df[features].dropna()
            y = df[target].loc[X.index]
            
            # Encode target if categorical
            le = LabelEncoder()
            if y.dtype == 'object':
                y_encoded = le.fit_transform(y)
            else:
                y_encoded = y
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
            )
            
            # Select model
            if algorithm == "Logistic Regression":
                model = LogisticRegression(max_iter=1000, random_state=42)
            elif algorithm == "Random Forest":
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:  # SVM
                model = SVC(random_state=42)
            
            # Train
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            # Classification report
            report = classification_report(y_test, y_pred, zero_division=0)
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                       cbar_kws={'label': 'Count'}, linewidths=1, linecolor='white')
            ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
            ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
            ax.set_title(f'Confusion Matrix - {algorithm}\nAccuracy: {accuracy:.4f}', 
                        fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            return {
                'accuracy': round(accuracy, 4),
                'precision': round(precision, 4),
                'recall': round(recall, 4),
                'f1_score': round(f1, 4),
                'classification_report': report,
                'confusion_matrix_plot': fig
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    @staticmethod
    def knn_analysis(df, features, target, k=5, task='Classification', test_size=0.2):
        """K-Nearest Neighbors analysis with visualization"""
        try:
            # Prepare data
            X = df[features].dropna()
            y = df[target].loc[X.index]
            
            if task == 'Classification':
                le = LabelEncoder()
                if y.dtype == 'object':
                    y_encoded = le.fit_transform(y)
                else:
                    y_encoded = y
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
                )
                
                model = KNeighborsClassifier(n_neighbors=k)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                
                # Create accuracy bar chart
                fig, ax = plt.subplots(figsize=(8, 5))
                metrics = ['Accuracy', 'F1 Score']
                values = [accuracy, f1]
                bars = ax.bar(metrics, values, color=['#667eea', '#764ba2'], alpha=0.8, edgecolor='black')
                ax.set_ylim(0, 1)
                ax.set_ylabel('Score', fontsize=12, fontweight='bold')
                ax.set_title(f'KNN Classification Performance (K={k})', fontsize=14, fontweight='bold')
                ax.grid(axis='y', alpha=0.3)
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.4f}', ha='center', va='bottom', fontweight='bold')
                
                plt.tight_layout()
                
                return {
                    'accuracy': round(accuracy, 4),
                    'f1_score': round(f1, 4),
                    'plot': fig
                }
            
            else:  # Regression
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )
                
                model = KNeighborsRegressor(n_neighbors=k)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                
                # Create scatter plot
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.scatter(y_test, y_pred, alpha=0.6, edgecolors='k', s=60)
                ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                       'r--', lw=2, label='Perfect Prediction')
                ax.set_xlabel('Actual Values', fontsize=12, fontweight='bold')
                ax.set_ylabel('Predicted Values', fontsize=12, fontweight='bold')
                ax.set_title(f'KNN Regression (K={k})\nR² = {r2:.4f}', fontsize=14, fontweight='bold')
                ax.legend()
                ax.grid(alpha=0.3)
                plt.tight_layout()
                
                return {
                    'r2_score': round(r2, 4),
                    'rmse': round(rmse, 4),
                    'plot': fig
                }
                
        except Exception as e:
            return {'error': str(e)}
    
    @staticmethod
    def anova_test(df, group_col, value_col):
        """Perform ANOVA test with visualization"""
        try:
            groups = df.groupby(group_col)[value_col].apply(list)
            
            # Perform ANOVA
            f_stat, p_value = stats.f_oneway(*groups)
            
            # Group statistics
            group_stats = {}
            for group_name, group_data in groups.items():
                group_stats[str(group_name)] = {
                    'mean': round(float(np.mean(group_data)), 4),
                    'std': round(float(np.std(group_data)), 4),
                    'count': len(group_data)
                }
            
            # Create boxplot
            fig, ax = plt.subplots(figsize=(10, 6))
            df.boxplot(column=value_col, by=group_col, ax=ax, 
                      patch_artist=True, boxprops=dict(facecolor='lightblue', alpha=0.7))
            ax.set_title(f'ANOVA Test: F={f_stat:.4f}, p={p_value:.6f}\n' + 
                        ('Significant Difference' if p_value < 0.05 else 'No Significant Difference'),
                        fontsize=14, fontweight='bold')
            ax.set_xlabel(group_col, fontsize=12, fontweight='bold')
            ax.set_ylabel(value_col, fontsize=12, fontweight='bold')
            plt.suptitle('')
            ax.grid(alpha=0.3)
            plt.tight_layout()
            
            return {
                'f_statistic': round(float(f_stat), 4),
                'p_value': round(float(p_value), 6),
                'significant': bool(p_value < 0.05),
                'group_stats': group_stats,
                'plot': fig
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    @staticmethod
    def decision_tree(df, features, target, max_depth=5, task='Classification', test_size=0.2):
        """Decision Tree analysis with feature importance plot"""
        try:
            # Prepare data
            X = df[features].dropna()
            y = df[target].loc[X.index]
            
            if task == 'Classification':
                le = LabelEncoder()
                if y.dtype == 'object':
                    y_encoded = le.fit_transform(y)
                else:
                    y_encoded = y
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
                )
                
                model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                
                feature_importance = {
                    feature: round(float(importance), 4)
                    for feature, importance in zip(features, model.feature_importances_)
                }
                
                # Create feature importance plot
                fig, ax = plt.subplots(figsize=(10, 6))
                importance_df = pd.DataFrame([
                    {'Feature': k, 'Importance': v}
                    for k, v in feature_importance.items()
                ]).sort_values('Importance', ascending=True)
                
                ax.barh(importance_df['Feature'], importance_df['Importance'], 
                       color='#667eea', alpha=0.8, edgecolor='black')
                ax.set_xlabel('Importance', fontsize=12, fontweight='bold')
                ax.set_ylabel('Feature', fontsize=12, fontweight='bold')
                ax.set_title(f'Decision Tree Feature Importance\nAccuracy: {accuracy:.4f}', 
                            fontsize=14, fontweight='bold')
                ax.grid(axis='x', alpha=0.3)
                plt.tight_layout()
                
                return {
                    'accuracy': round(accuracy, 4),
                    'f1_score': round(f1, 4),
                    'feature_importance': feature_importance,
                    'plot': fig
                }
            
            else:  # Regression
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )
                
                model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                
                feature_importance = {
                    feature: round(float(importance), 4)
                    for feature, importance in zip(features, model.feature_importances_)
                }
                
                # Create feature importance plot
                fig, ax = plt.subplots(figsize=(10, 6))
                importance_df = pd.DataFrame([
                    {'Feature': k, 'Importance': v}
                    for k, v in feature_importance.items()
                ]).sort_values('Importance', ascending=True)
                
                ax.barh(importance_df['Feature'], importance_df['Importance'], 
                       color='#764ba2', alpha=0.8, edgecolor='black')
                ax.set_xlabel('Importance', fontsize=12, fontweight='bold')
                ax.set_ylabel('Feature', fontsize=12, fontweight='bold')
                ax.set_title(f'Decision Tree Feature Importance\nR²: {r2:.4f}', 
                            fontsize=14, fontweight='bold')
                ax.grid(axis='x', alpha=0.3)
                plt.tight_layout()
                
                return {
                    'r2_score': round(r2, 4),
                    'rmse': round(rmse, 4),
                    'feature_importance': feature_importance,
                    'plot': fig
                }
                
        except Exception as e:
            return {'error': str(e)}
