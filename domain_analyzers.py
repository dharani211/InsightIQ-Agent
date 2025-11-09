import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

class UniversalAnalyzer:
    """Universal data analyzer for multiple domains"""
    
    @staticmethod
    def quick_summary(df):
        """Generate comprehensive data summary"""
        summary = {
            'basic_info': {
                'rows': len(df),
                'columns': len(df.columns),
                'memory_usage_mb': round(df.memory_usage(deep=True).sum() / 1024**2, 2)
            },
            'column_types': df.dtypes.value_counts().to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicates': int(df.duplicated().sum())
        }
        
        # Numeric summary
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            summary['numeric_summary'] = df[numeric_cols].describe().to_dict()
        
        # Categorical summary
        cat_cols = df.select_dtypes(include=['object']).columns
        if len(cat_cols) > 0:
            summary['categorical_summary'] = {
                col: df[col].value_counts().head(5).to_dict()
                for col in cat_cols[:3]
            }
        
        return summary
    
    @staticmethod
    def correlation_analysis(df, method='pearson'):
        """Calculate correlation matrix"""
        numeric_df = df.select_dtypes(include=['number'])
        
        if numeric_df.shape[1] < 2:
            return None, {"error": "Need at least 2 numeric columns"}
        
        corr_matrix = numeric_df.corr(method=method)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   fmt='.2f', square=True, linewidths=0.5, ax=ax)
        ax.set_title(f'Correlation Matrix ({method.capitalize()})', fontsize=14, pad=20)
        plt.tight_layout()
        
        return fig, corr_matrix
    
    @staticmethod
    def distribution_plot(df, column, group_col=None):
        """Create distribution visualization"""
        if column not in df.columns:
            return None, {"error": "Column not found"}
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        if group_col and group_col in df.columns:
            for group in df[group_col].dropna().unique():
                data = df[df[group_col] == group][column].dropna()
                axes[0].hist(data, alpha=0.6, label=str(group), bins=25, edgecolor='black')
            axes[0].legend()
        else:
            axes[0].hist(df[column].dropna(), bins=25, color='steelblue', alpha=0.7, edgecolor='black')
        
        axes[0].set_xlabel(column, fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title(f'Distribution: {column}', fontsize=14)
        axes[0].grid(alpha=0.3)
        
        # Boxplot
        if group_col and group_col in df.columns:
            df.boxplot(column=column, by=group_col, ax=axes[1])
            axes[1].set_title(f'{column} by {group_col}', fontsize=14)
            axes[1].set_xlabel(group_col, fontsize=12)
            axes[1].set_ylabel(column, fontsize=12)
        else:
            df.boxplot(column=column, ax=axes[1])
            axes[1].set_title(f'Boxplot: {column}', fontsize=14)
            axes[1].set_ylabel(column, fontsize=12)
        
        plt.suptitle('')  # Remove default title
        plt.tight_layout()
        return fig, None
    
    @staticmethod
    def group_comparison(df, group_col, measure_col):
        """Statistical comparison between groups"""
        if group_col not in df.columns or measure_col not in df.columns:
            return {"error": "Column not found"}
        
        groups = df[group_col].dropna().unique()
        
        if len(groups) < 2:
            return {"error": "Need at least 2 groups"}
        
        result = {
            'groups': {},
            'total_subjects': len(df)
        }
        
        # Calculate stats for each group
        for group in groups:
            group_data = df[df[group_col] == group][measure_col].dropna()
            result['groups'][str(group)] = {
                'n': len(group_data),
                'mean': round(float(group_data.mean()), 3),
                'std': round(float(group_data.std()), 3),
                'median': round(float(group_data.median()), 3),
                'min': round(float(group_data.min()), 3),
                'max': round(float(group_data.max()), 3)
            }
        
        # Perform statistical test if 2 groups
        if len(groups) == 2:
            g1_data = df[df[group_col] == groups[0]][measure_col].dropna()
            g2_data = df[df[group_col] == groups[1]][measure_col].dropna()
            
            t_stat, p_val = stats.ttest_ind(g1_data, g2_data)
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt((g1_data.std()**2 + g2_data.std()**2) / 2)
            cohens_d = (g1_data.mean() - g2_data.mean()) / pooled_std if pooled_std != 0 else 0
            
            result['statistical_test'] = {
                't_statistic': round(float(t_stat), 4),
                'p_value': round(float(p_val), 4),
                'significant_p05': bool(p_val < 0.05),
                'cohens_d': round(float(cohens_d), 3),
                'effect_size': 'small' if abs(cohens_d) < 0.5 else 
                              ('medium' if abs(cohens_d) < 0.8 else 'large')
            }
        
        return result
    
    @staticmethod
    def categorical_analysis(df, column):
        """Analyze categorical variable"""
        if column not in df.columns:
            return None, {"error": "Column not found"}
        
        value_counts = df[column].value_counts()
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Bar chart
        value_counts.plot(kind='bar', ax=axes[0], color='steelblue', edgecolor='black')
        axes[0].set_title(f'Frequency: {column}', fontsize=14)
        axes[0].set_xlabel(column, fontsize=12)
        axes[0].set_ylabel('Count', fontsize=12)
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(axis='y', alpha=0.3)
        
        # Pie chart
        value_counts.plot(kind='pie', ax=axes[1], autopct='%1.1f%%', startangle=90)
        axes[1].set_title(f'Distribution: {column}', fontsize=14)
        axes[1].set_ylabel('')
        
        plt.tight_layout()
        
        return fig, value_counts.to_dict()
