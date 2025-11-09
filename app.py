import streamlit as st
import pandas as pd
from agent import UniversalDataAgent
from domain_analyzers import UniversalAnalyzer
from ml_models import MLAnalyzer
import os
from pathlib import Path

# Page config
st.set_page_config(
    page_title="InsightIQ Agent",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Create directories
Path('data').mkdir(exist_ok=True)
Path('outputs').mkdir(exist_ok=True)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">InsightIQ Agent</p>', unsafe_allow_html=True)
st.markdown("**AI-powered analysis for CSV/Excel datasets across multiple domains**")

# Sidebar
with st.sidebar:
    st.header("Configuration")
    
    use_ollama = st.radio(
        "Select LLM:",
        ["Ollama (Local)", "Google Gemini (Cloud)"],
        index=0
    ) == "Ollama (Local)"
    
    if not use_ollama:
        api_key = st.text_input("Gemini API Key:", type="password")
    else:
        api_key = None
    
    st.markdown("---")
    
    st.markdown("### Supported Domains")
    domains = [
        "Healthcare/Medical",
        "Sales/E-commerce",
        "Transportation",
        "Telecommunications",
        "Real Estate",
        "Social/Happiness",
        "Finance",
        "Neuroimaging",
        "General"
    ]
    
    for domain in domains:
        st.text(f"{domain}")
    
    st.markdown("---")
    
    st.markdown("### Features")
    st.markdown("""
    - Natural language queries
    - Auto-domain detection
    - Statistical analysis
    - Smart visualizations
    - CSV & Excel support
    - Group comparisons
    - Correlation analysis
    - Machine Learning models
    """)
    
    st.markdown("---")
    
    if st.button("Download Sample Datasets"):
        st.info("Run `python download_datasets.py` in terminal")

# File upload
uploaded_file = st.file_uploader(
    "Upload your data file",
    type=['csv', 'xlsx', 'xls', 'xlsm', 'tsv'],
    help="Supported: CSV, Excel (.xlsx, .xls), TSV"
)

if uploaded_file:
    # Save file
    save_path = os.path.join('data', uploaded_file.name)
    with open(save_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    
    # Initialize agent
    if 'agent' not in st.session_state or st.session_state.get('current_file') != uploaded_file.name:
        st.session_state.current_file = uploaded_file.name
        
        with st.spinner("Initializing agent..."):
            try:
                st.session_state.agent = UniversalDataAgent(use_ollama=use_ollama, api_key=api_key)
                
                # Handle Excel sheets
                if uploaded_file.name.endswith(('.xlsx', '.xls', '.xlsm')):
                    sheets = st.session_state.agent.get_excel_sheets(save_path)
                    
                    if len(sheets) > 1:
                        selected_sheet = st.selectbox("Select Excel sheet:", sheets)
                        preview, shape, cols = st.session_state.agent.load_excel_sheet(save_path, selected_sheet)
                    else:
                        preview, shape, cols, file_type = st.session_state.agent.load_data(save_path)
                else:
                    preview, shape, cols, file_type = st.session_state.agent.load_data(save_path)
                
                st.session_state.df_preview = preview
                st.session_state.shape = shape
                st.session_state.cols = cols
                st.session_state.file_type = st.session_state.agent.file_type
                st.session_state.domain = st.session_state.agent.detected_domain
                
                st.success(f"Loaded {st.session_state.file_type} successfully!")
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.stop()
    
    # Domain badge
    domain_colors = {
        'Healthcare/Medical': '#ff6b6b',
        'Sales/E-commerce': '#4ecdc4',
        'Transportation': '#45b7d1',
        'Telecommunications': '#96ceb4',
        'Real Estate': '#ffeaa7',
        'Social/Happiness': '#dfe6e9',
        'Finance': '#74b9ff',
        'Neuroimaging': '#a29bfe',
        'General': '#636e72'
    }
    
    domain_color = domain_colors.get(st.session_state.domain, '#636e72')
    st.markdown(f"""
    <div style='background-color:{domain_color}; padding:0.5rem; border-radius:0.5rem; text-align:center; margin:1rem 0;'>
        <h3 style='margin:0; color:white;'>Detected Domain: {st.session_state.domain}</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Data preview
    st.subheader("Data Preview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Rows", f"{st.session_state.shape[0]:,}")
    with col2:
        st.metric("Total Columns", st.session_state.shape[1])
    with col3:
        missing = st.session_state.agent.df.isnull().sum().sum()
        st.metric("Missing Values", f"{missing:,}")
    with col4:
        duplicates = st.session_state.agent.df.duplicated().sum()
        st.metric("Duplicates", f"{duplicates:,}")
    
    st.dataframe(st.session_state.df_preview, use_container_width=True, height=200)
    
    # Domain insights
    with st.expander("Smart Insights & Suggestions"):
        insights = st.session_state.agent.get_domain_insights()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Suggested Analyses:**")
            for suggestion in insights['suggestions']:
                st.markdown(f"• {suggestion}")
        
        with col2:
            st.markdown("**Key Metrics:**")
            for metric in insights['key_metrics']:
                st.markdown(f"• {metric}")
        
        with col3:
            st.markdown("**Example Questions:**")
            for question in insights['example_questions'][:3]:
                st.markdown(f"• {question}")
    
    st.markdown("---")
    
    # Main tabs - ADDED MACHINE LEARNING TAB
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Ask Questions",
        "Quick Analysis",
        "Visualizations",
        "Machine Learning",
        "Export Data"
    ])
    
    # Tab 1: Natural Language Queries
    with tab1:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown("### Ask questions in plain English")
            
            # Smart questions
            smart_questions = st.session_state.agent.generate_smart_questions()
            selected_q = st.selectbox(
                "Try these AI-generated questions:",
                [""] + smart_questions,
                key="smart_q"
            )
            
            user_query = st.text_area(
                "Your question:",
                value=selected_q if selected_q else "",
                height=100,
                placeholder="e.g., What is the average value grouped by category?"
            )
            
            col_a, col_b = st.columns([1, 3])
            with col_a:
                run_query = st.button("Analyze", type="primary", use_container_width=True)
            
            if run_query and user_query.strip():
                with st.spinner("Thinking..."):
                    response = st.session_state.agent.query(user_query)
                    st.success("**Answer:**")
                    st.write(response)
        
        with col2:
            st.markdown("### Tips")
            st.info("""
            **Good questions:**
            - "What is the average X?"
            - "Show distribution of Y"
            - "Compare A vs B"
            - "Correlation between X and Y"
            - "How many rows where Z > 100?"
            """)
    
    # Tab 2: Quick Analysis
    with tab2:
        st.markdown("### Quick Data Analysis")
        
        analyzer = UniversalAnalyzer()
        df = st.session_state.agent.df
        
        analysis_type = st.selectbox(
            "Select analysis:",
            ["Data Summary", "Group Comparison", "Categorical Analysis"]
        )
        
        if analysis_type == "Data Summary":
            if st.button("Generate Summary", use_container_width=True):
                with st.spinner("Analyzing..."):
                    summary = analyzer.quick_summary(df)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Basic Info:**")
                        st.json(summary['basic_info'])
                        
                        st.markdown("**Missing Values:**")
                        missing_df = pd.DataFrame([
                            {'Column': k, 'Missing': v}
                            for k, v in summary['missing_values'].items() if v > 0
                        ])
                        if len(missing_df) > 0:
                            st.dataframe(missing_df, use_container_width=True)
                        else:
                            st.success("No missing values!")
                    
                    with col2:
                        if 'numeric_summary' in summary:
                            st.markdown("**Numeric Summary:**")
                            st.dataframe(pd.DataFrame(summary['numeric_summary']), use_container_width=True)
        
        elif analysis_type == "Group Comparison":
            col1, col2 = st.columns(2)
            
            with col1:
                group_col = st.selectbox("Group column:", df.columns, key="grp1")
            with col2:
                numeric_cols = df.select_dtypes(include=['number']).columns
                measure_col = st.selectbox("Measure column:", numeric_cols, key="meas1")
            
            if st.button("🔬 Compare Groups", use_container_width=True):
                with st.spinner("Running analysis..."):
                    result = analyzer.group_comparison(df, group_col, measure_col)
                    
                    if 'error' in result:
                        st.error(result['error'])
                    else:
                        st.json(result)
                        
                        if 'statistical_test' in result:
                            if result['statistical_test']['significant_p05']:
                                st.success(f"Significant difference (p={result['statistical_test']['p_value']})")
                            else:
                                st.info(f"No significant difference (p={result['statistical_test']['p_value']})")
        
        elif analysis_type == "Categorical Analysis":
            cat_cols = df.select_dtypes(include=['object']).columns
            
            if len(cat_cols) == 0:
                st.warning("No categorical columns found")
            else:
                selected_col = st.selectbox("Select column:", cat_cols)
                
                if st.button("Analyze", use_container_width=True):
                    fig, counts = analyzer.categorical_analysis(df, selected_col)
                    if fig:
                        st.pyplot(fig)
                        st.markdown("**Counts:**")
                        st.json(counts)
    
    # Tab 3: Visualizations
    with tab3:
        st.markdown("### Data Visualizations")
        
        viz_type = st.selectbox(
            "Visualization type:",
            ["Correlation Heatmap", "Distribution Plot", "Categorical Plot"]
        )
        
        if viz_type == "Correlation Heatmap":
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            
            if len(numeric_cols) < 2:
                st.warning("Need at least 2 numeric columns")
            else:
                method = st.radio("Correlation method:", ["pearson", "spearman", "kendall"])
                
                if st.button("Generate Heatmap"):
                    fig, corr = analyzer.correlation_analysis(df, method=method)
                    if fig:
                        st.pyplot(fig)
                        
                        with st.expander("View correlation values"):
                            st.dataframe(corr, use_container_width=True)
        
        elif viz_type == "Distribution Plot":
            col1, col2 = st.columns(2)
            
            with col1:
                numeric_cols = df.select_dtypes(include=['number']).columns
                dist_col = st.selectbox("Select column:", numeric_cols, key="dist1")
            
            with col2:
                all_cols = ['None'] + df.columns.tolist()
                group_col = st.selectbox("Group by (optional):", all_cols, key="grp2")
                group_col = None if group_col == 'None' else group_col
            
            if st.button("Generate Plot"):
                fig, err = analyzer.distribution_plot(df, dist_col, group_col)
                if fig:
                    st.pyplot(fig)
                else:
                    st.error(err['error'])
        
        elif viz_type == "Categorical Plot":
            cat_cols = df.select_dtypes(include=['object']).columns
            
            if len(cat_cols) == 0:
                st.warning("No categorical columns found")
            else:
                selected_col = st.selectbox("Select column:", cat_cols, key="cat1")
                
                if st.button("Generate Plot"):
                    fig, counts = analyzer.categorical_analysis(df, selected_col)
                    if fig:
                        st.pyplot(fig)
    
    # Tab 4: MACHINE LEARNING (NEW!)
    with tab4:
        st.markdown("### Machine Learning Models")
        
        ml_analyzer = MLAnalyzer()
        
        ml_task = st.selectbox(
            "Select ML Task:",
            ["Linear Regression", "Classification", "K-Nearest Neighbors (KNN)", "ANOVA Test", "Decision Tree"]
        )
        
        # LINEAR REGRESSION
        if ml_task == "Linear Regression":
            st.markdown("#### Linear Regression")
            st.caption("Predict a continuous target variable")
            
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            
            if len(numeric_cols) < 2:
                st.warning("Need at least 2 numeric columns for regression")
            else:
                col1, col2 = st.columns(2)
                
                with col1:
                    target = st.selectbox("Target (Y):", numeric_cols, key="lr_target")
                
                with col2:
                    available_features = [col for col in numeric_cols if col != target]
                    features = st.multiselect(
                        "Features (X):",
                        available_features,
                        default=available_features[:min(3, len(available_features))]
                    )
                
                test_size = st.slider("Test set size:", 0.1, 0.5, 0.2, 0.05)
                
                if features and st.button("Train Linear Regression", use_container_width=True):
                    with st.spinner("Training model..."):
                        result = ml_analyzer.linear_regression(df, features, target, test_size)
                        
                        if 'error' in result:
                            st.error(result['error'])
                        else:
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("R² Score", f"{result['r2_score']:.4f}")
                            with col2:
                                st.metric("RMSE", f"{result['rmse']:.4f}")
                            with col3:
                                st.metric("MAE", f"{result['mae']:.4f}")
                            
                            st.markdown("**Feature Coefficients:**")
                            coef_df = pd.DataFrame([
                                {'Feature': k, 'Coefficient': v}
                                for k, v in result['coefficients'].items()
                            ])
                            st.dataframe(coef_df, use_container_width=True)
                            
                            if result['plot']:
                                st.pyplot(result['plot'])
        
        # CLASSIFICATION
        elif ml_task == "Classification":
            st.markdown("#### Classification")
            st.caption("Predict categorical outcomes")
            
            cat_cols = df.select_dtypes(include=['object']).columns.tolist()
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            binary_cols = [col for col in numeric_cols if df[col].nunique() <= 10]
            all_target_cols = cat_cols + binary_cols
            
            if not all_target_cols:
                st.warning("No categorical columns found for classification")
            else:
                col1, col2 = st.columns(2)
                
                with col1:
                    target = st.selectbox("Target (Y):", all_target_cols, key="clf_target")
                
                with col2:
                    available_features = [col for col in numeric_cols if col != target]
                    features = st.multiselect(
                        "Features (X):",
                        available_features,
                        default=available_features[:min(3, len(available_features))]
                    )
                
                algorithm = st.selectbox(
                    "Algorithm:",
                    ["Logistic Regression", "Random Forest", "Support Vector Machine"]
                )
                
                test_size = st.slider("Test set size:", 0.1, 0.5, 0.2, 0.05, key="clf_test")
                
                if features and st.button("Train Classifier", use_container_width=True):
                    with st.spinner("Training model..."):
                        result = ml_analyzer.classification(df, features, target, algorithm, test_size)
                        
                        if 'error' in result:
                            st.error(result['error'])
                        else:
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Accuracy", f"{result['accuracy']:.4f}")
                            with col2:
                                st.metric("Precision", f"{result['precision']:.4f}")
                            with col3:
                                st.metric("Recall", f"{result['recall']:.4f}")
                            
                            st.markdown("**Classification Report:**")
                            st.text(result['classification_report'])
                            
                            if result['confusion_matrix_plot']:
                                st.pyplot(result['confusion_matrix_plot'])
        
        # KNN
        elif ml_task == "K-Nearest Neighbors (KNN)":
            st.markdown("#### K-Nearest Neighbors")
            st.caption("Classification or regression using nearest neighbors")
            
            task_type = st.radio("Task:", ["Classification", "Regression"], key="knn_task")
            
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            cat_cols = df.select_dtypes(include=['object']).columns.tolist()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if task_type == "Classification":
                    binary_cols = [col for col in numeric_cols if df[col].nunique() <= 10]
                    all_targets = cat_cols + binary_cols
                    if all_targets:
                        target = st.selectbox("Target:", all_targets, key="knn_target")
                    else:
                        st.error("No categorical columns found")
                        target = None
                else:
                    if numeric_cols:
                        target = st.selectbox("Target:", numeric_cols, key="knn_target_reg")
                    else:
                        st.error("No numeric columns found")
                        target = None
            
            if target:
                with col2:
                    available_features = [col for col in numeric_cols if col != target]
                    features = st.multiselect(
                        "Features:",
                        available_features,
                        default=available_features[:min(3, len(available_features))],
                        key="knn_features"
                    )
                
                with col3:
                    k_neighbors = st.slider("K Neighbors:", 1, 20, 5)
                
                test_size = st.slider("Test set size:", 0.1, 0.5, 0.2, 0.05, key="knn_test")
                
                if features and st.button("Train KNN", use_container_width=True):
                    with st.spinner("Training KNN model..."):
                        result = ml_analyzer.knn_analysis(df, features, target, k_neighbors, task_type, test_size)
                        
                        if 'error' in result:
                            st.error(result['error'])
                        else:
                            if task_type == "Classification":
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Accuracy", f"{result['accuracy']:.4f}")
                                with col2:
                                    st.metric("F1 Score", f"{result['f1_score']:.4f}")
                            else:
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("R² Score", f"{result['r2_score']:.4f}")
                                with col2:
                                    st.metric("RMSE", f"{result['rmse']:.4f}")
                            
                            if 'plot' in result and result['plot']:
                                st.pyplot(result['plot'])
        
        # ANOVA
        elif ml_task == "ANOVA Test":
            st.markdown("#### ANOVA (Analysis of Variance)")
            st.caption("Test if there are significant differences between group means")
            
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            cat_cols = df.select_dtypes(include=['object']).columns.tolist()
            
            col1, col2 = st.columns(2)
            
            with col1:
                if cat_cols:
                    group_col = st.selectbox("Groups:", cat_cols, key="anova_group")
                else:
                    st.error("No categorical columns found")
                    group_col = None
            
            with col2:
                if group_col and numeric_cols:
                    value_col = st.selectbox("Values:", numeric_cols, key="anova_value")
                elif not numeric_cols:
                    st.error("No numeric columns found")
                    value_col = None
                else:
                    value_col = None
            
            if group_col and value_col and st.button("Run ANOVA", use_container_width=True):
                with st.spinner("Running ANOVA test..."):
                    result = ml_analyzer.anova_test(df, group_col, value_col)
                    
                    if 'error' in result:
                        st.error(result['error'])
                    else:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("F-Statistic", f"{result['f_statistic']:.4f}")
                        with col2:
                            st.metric("P-Value", f"{result['p_value']:.6f}")
                        
                        if result['significant']:
                            st.success("Significant difference between groups (p < 0.05)")
                        else:
                            st.info("No significant difference between groups (p ≥ 0.05)")
                        
                        st.markdown("**Group Statistics:**")
                        group_df = pd.DataFrame(result['group_stats']).T
                        st.dataframe(group_df, use_container_width=True)
                        
                        if result['plot']:
                            st.pyplot(result['plot'])
        
        # DECISION TREE
        elif ml_task == "Decision Tree":
            st.markdown("#### Decision Tree")
            st.caption("Tree-based model for classification or regression")
            
            task_type = st.radio("Task:", ["Classification", "Regression"], key="dt_task")
            
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            cat_cols = df.select_dtypes(include=['object']).columns.tolist()
            
            col1, col2 = st.columns(2)
            
            with col1:
                if task_type == "Classification":
                    binary_cols = [col for col in numeric_cols if df[col].nunique() <= 10]
                    all_targets = cat_cols + binary_cols
                    if all_targets:
                        target = st.selectbox("Target:", all_targets, key="dt_target")
                    else:
                        st.error("No categorical columns found")
                        target = None
                else:
                    if numeric_cols:
                        target = st.selectbox("Target:", numeric_cols, key="dt_target_reg")
                    else:
                        st.error("No numeric columns found")
                        target = None
            
            if target:
                with col2:
                    available_features = [col for col in numeric_cols if col != target]
                    features = st.multiselect(
                        "Features:",
                        available_features,
                        default=available_features[:min(3, len(available_features))],
                        key="dt_features"
                    )
                
                max_depth = st.slider("Max Depth:", 1, 20, 5)
                test_size = st.slider("Test set size:", 0.1, 0.5, 0.2, 0.05, key="dt_test")
                
                if features and st.button("Train Decision Tree", use_container_width=True):
                    with st.spinner("Training decision tree..."):
                        result = ml_analyzer.decision_tree(df, features, target, max_depth, task_type, test_size)
                        
                        if 'error' in result:
                            st.error(result['error'])
                        else:
                            if task_type == "Classification":
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Accuracy", f"{result['accuracy']:.4f}")
                                with col2:
                                    st.metric("F1 Score", f"{result['f1_score']:.4f}")
                            else:
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("R² Score", f"{result['r2_score']:.4f}")
                                with col2:
                                    st.metric("RMSE", f"{result['rmse']:.4f}")
                            
                            st.markdown("**Feature Importance:**")
                            importance_df = pd.DataFrame([
                                {'Feature': k, 'Importance': v}
                                for k, v in result['feature_importance'].items()
                            ]).sort_values('Importance', ascending=False)
                            st.dataframe(importance_df, use_container_width=True)
                            
                            if 'plot' in result and result['plot']:
                                st.pyplot(result['plot'])
    
    # Tab 5: Export Data (formerly Tab 4)
    with tab5:
        st.markdown("###Export Data")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Full Dataset**")
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"export_{uploaded_file.name}",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            st.markdown("**Summary Statistics**")
            summary = df.describe().to_csv()
            st.download_button(
                label="Download Stats",
                data=summary,
                file_name="summary_statistics.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col3:
            st.markdown("**Correlation Matrix**")
            numeric_df = df.select_dtypes(include=['number'])
            if len(numeric_df.columns) >= 2:
                corr = numeric_df.corr().to_csv()
                st.download_button(
                    label="Download Correlations",
                    data=corr,
                    file_name="correlation_matrix.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.info("Need at least 2 numeric columns")

else:
    # Landing page
    st.info("**Upload a CSV or Excel file to begin analysis**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Supported File Formats")
        st.markdown("""
        - **CSV** (.csv)
        - **Excel** (.xlsx, .xls, .xlsm)
        - **TSV** (.tsv)
        """)
        
        st.markdown("### Supported Domains")
        st.markdown("""
        - Healthcare & Medical
        - Sales & E-commerce
        - Transportation
        - Telecommunications
        - Real Estate
        - Social & Happiness
        - Finance
        - Neuroimaging
        - General Analysis
        """)
    
    with col2:
        st.markdown("### Quick Start")
        st.markdown("""
        1. **Download sample datasets**
           ```
           python download_datasets.py
           ```
        
        2. **Upload a file** using the uploader above
        
        3. **Ask questions** in plain English
        
        4. **Explore** visualizations and insights
        """)
        
        st.markdown("### Sample Datasets Available")
        st.markdown("""
        - `titanic.csv` - Passenger survival data
        - `customer_churn.csv` - Telecom churn analysis
        - `superstore_sales.csv` - Retail sales data
        - `world_happiness.csv` - Country happiness scores
        - `insurance.csv` - Medical insurance costs
        - `real_estate.csv` - Property listings
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    Designed by Dharani Mandla
</div>
""", unsafe_allow_html=True)
