"""
╔══════════════════════════════════════════════════════════════════════╗
║           SALARY PREDICTION USING MULTIPLE REGRESSION MODELS          ║
║                     Professional ML Dashboard                          ║
║                        Built with Streamlit                            ║
╚══════════════════════════════════════════════════════════════════════╝

Author: [Janhavi Maurya]
Project: Salary Prediction System
Technologies: Python, Streamlit, Scikit-learn, Pandas, Plotly

Models Used:
- Linear Regression
- Ridge Regression
- Lasso Regression
- Decision Tree
- Random Forest
- K-Neighbors
- Support Vector Regressor (SVR)
"""

# ============================================
# IMPORT LIBRARIES
# ============================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import warnings
warnings.filterwarnings('ignore')

# Sklearn imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import (r2_score, mean_squared_error, mean_absolute_error, 
                             mean_absolute_percentage_error)


# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Salary Prediction ML Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS STYLING
# ============================================
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Sub header styling */
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Metric card styling */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Info box */
    .info-box {
        background-color: #e7f3ff;
        border: 1px solid #b6d4fe;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
</style>
""", unsafe_allow_html=True)


# ============================================
# HELPER FUNCTIONS
# ============================================

@st.cache_data
def load_data(file_path):
    """Load and cache the dataset"""
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        st.error(f"File not found: {file_path}")
        return None


def get_regression_models():
    """
    Return dictionary of selected regression models:
    - Linear Regression
    - Ridge Regression  
    - Lasso Regression
    - Decision Tree
    - Random Forest
    - K-Neighbors
    - SVR
    """
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=1.0),
        'Decision Tree': DecisionTreeRegressor(random_state=42, max_depth=10),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'K-Neighbors': KNeighborsRegressor(n_neighbors=5),
        'SVR': SVR(kernel='rbf', C=1.0)
    }
    return models


def evaluate_model(model, X_train, X_test, y_train, y_test):
    """Train and evaluate a model"""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    metrics = {
        'R2 Score': r2_score(y_test, y_pred),
        'MAE': mean_absolute_error(y_test, y_pred),
        'MSE': mean_squared_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'MAPE': mean_absolute_percentage_error(y_test, y_pred) * 100
    }
    
    return metrics, y_pred, model


def get_feature_importance(model, feature_names):
    """Get feature importance from model"""
    if hasattr(model, 'feature_importances_'):
        return pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
    elif hasattr(model, 'coef_'):
        return pd.DataFrame({
            'Feature': feature_names,
            'Importance': np.abs(model.coef_)
        }).sort_values('Importance', ascending=False)
    return None


# ============================================
# SIDEBAR
# ============================================
st.sidebar.markdown("## 🎛️ Navigation")

# Page selection
page = st.sidebar.radio(
    "Select Page",
    ["🏠 Home", "📊 Data Exploration", "🤖 Model Training", 
     "📈 Model Comparison", "🔮 Predict Salary", "📋 About"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📁 Dataset")

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=['csv'])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
else:
    data = load_data("Salary Data.csv")


# ============================================
# DATA CLEANING
# ============================================
if data is not None:
    # Check for missing values
    missing_count = data.isnull().sum().sum()
    
    if missing_count > 0:
        st.sidebar.warning(f"⚠️ Found {missing_count} missing values")
        
        # Fill missing values for categorical columns
        for col in data.select_dtypes(include=['object']).columns:
            data[col] = data[col].fillna('Unknown')
        
        # Fill missing values for numerical columns with median
        for col in data.select_dtypes(include=[np.number]).columns:
            data[col] = data[col].fillna(data[col].median())
        
        st.sidebar.success("✅ Missing values handled!")


# Sidebar info
st.sidebar.markdown("---")
st.sidebar.markdown("### 📌 Quick Stats")
if data is not None:
    st.sidebar.write(f"📊 Rows: {data.shape[0]}")
    st.sidebar.write(f"📋 Columns: {data.shape[1]}")
    st.sidebar.write(f"💰 Avg Salary: ${data['Salary'].mean():,.2f}")

# Show available models
st.sidebar.markdown("---")
st.sidebar.markdown("### 🤖 Available Models")
st.sidebar.markdown("""
1. Linear Regression
2. Ridge Regression
3. Lasso Regression
4. Decision Tree
5. Random Forest
6. K-Neighbors
7. SVR
""")


# ============================================
# PAGE: HOME
# ============================================
if page == "🏠 Home":
    
    # Header
    st.markdown('<h1 class="main-header">💰 Salary Prediction Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Machine Learning Powered Salary Prediction System</p>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Project Overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h2>🤖 7</h2>
            <p>ML Models</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h2>📊 Interactive</h2>
            <p>Visualizations</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h2>🎯 Real-time</h2>
            <p>Predictions</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # About the project
    st.subheader("📖 About This Project")
    
    st.markdown("""
    <div class="info-box">
    <h4>🎯 Project Objective</h4>
    <p>This project predicts employee salaries based on various factors like age, gender, 
    education level, job title, and years of experience using multiple machine learning 
    regression algorithms.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Technologies and Models
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🛠️ Technologies Used")
        st.markdown("""
        - **Python** - Programming Language
        - **Streamlit** - Web Application Framework
        - **Pandas & NumPy** - Data Manipulation
        - **Scikit-learn** - Machine Learning
        - **Plotly & Matplotlib** - Visualization
        """)
    
    with col2:
        st.subheader("📊 ML Models Implemented")
        st.markdown("""
        | # | Model | Type |
        |---|-------|------|
        | 1 | Linear Regression | Linear |
        | 2 | Ridge Regression | Linear (L2) |
        | 3 | Lasso Regression | Linear (L1) |
        | 4 | Decision Tree | Tree-based |
        | 5 | Random Forest | Ensemble |
        | 6 | K-Neighbors | Instance-based |
        | 7 | SVR | Kernel-based |
        """)
    
    # Dataset preview
    st.markdown("---")
    st.subheader("📋 Dataset Preview")
    
    if data is not None:
        st.dataframe(data.head(10), use_container_width=True)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Records", f"{len(data):,}")
        col2.metric("Features", f"{len(data.columns)-1}")
        col3.metric("Min Salary", f"${data['Salary'].min():,.0f}")
        col4.metric("Max Salary", f"${data['Salary'].max():,.0f}")


# ============================================
# PAGE: DATA EXPLORATION
# ============================================
elif page == "📊 Data Exploration":
    
    st.markdown('<h1 class="main-header">📊 Data Exploration</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    if data is not None:
        
        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📋 Overview", "📈 Statistics", "🔍 Distributions", "🔗 Correlations"])
        
        # TAB 1: Overview
        with tab1:
            st.subheader("Dataset Overview")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("##### 📊 Dataset Shape")
                st.info(f"Rows: {data.shape[0]} | Columns: {data.shape[1]}")
                
                st.markdown("##### 📋 Column Information")
                st.dataframe(pd.DataFrame({
                    'Column': data.columns,
                    'Data Type': data.dtypes.values,
                    'Non-Null Count': data.notnull().sum().values,
                    'Null Count': data.isnull().sum().values
                }), use_container_width=True)
            
            with col2:
                st.markdown("##### 🔍 Missing Values")
                missing = data.isnull().sum()
                if missing.sum() == 0:
                    st.success("✅ No missing values found!")
                else:
                    st.warning(f"⚠️ {missing.sum()} missing values found")
                    st.dataframe(missing[missing > 0])
                
                st.markdown("##### 📝 Unique Values")
                unique_df = pd.DataFrame({
                    'Column': data.columns,
                    'Unique Values': [data[col].nunique() for col in data.columns]
                })
                st.dataframe(unique_df, use_container_width=True)
        
        # TAB 2: Statistics
        with tab2:
            st.subheader("Statistical Summary")
            
            st.markdown("##### 🔢 Numerical Columns")
            st.dataframe(data.describe(), use_container_width=True)
            
            st.markdown("##### 📝 Categorical Columns")
            categorical_cols = data.select_dtypes(include=['object']).columns
            
            for col in categorical_cols:
                with st.expander(f"📊 {col} - Value Counts"):
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        value_counts = data[col].dropna().value_counts()
                        fig = px.bar(
                            x=value_counts.index[:15],
                            y=value_counts.values[:15],
                            labels={'x': col, 'y': 'Count'},
                            title=f'Distribution of {col}'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    with col2:
                        st.dataframe(data[col].value_counts().head(15))
        
        # TAB 3: Distributions
        with tab3:
            st.subheader("Data Distributions")
            
            # Salary Distribution
            st.markdown("##### 💰 Salary Distribution")
            fig = make_subplots(rows=1, cols=2, subplot_titles=('Histogram', 'Box Plot'))
            
            fig.add_trace(
                go.Histogram(x=data['Salary'], name='Salary', marker_color='#667eea'),
                row=1, col=1
            )
            fig.add_trace(
                go.Box(y=data['Salary'], name='Salary', marker_color='#764ba2'),
                row=1, col=2
            )
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Salary by Categories
            st.markdown("##### 📊 Salary by Categories")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.box(data, x='Education Level', y='Salary', 
                            color='Education Level',
                            title='Salary by Education Level')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.box(data, x='Gender', y='Salary', 
                            color='Gender',
                            title='Salary by Gender')
                st.plotly_chart(fig, use_container_width=True)
            
            # Salary vs Experience
            st.markdown("##### 📈 Salary vs Experience")
            fig = px.scatter(data, x='Years of Experience', y='Salary', 
                           color='Education Level',
                           size='Age',
                           hover_data=['Job Title'],
                           title='Salary vs Years of Experience')
            st.plotly_chart(fig, use_container_width=True)
        
        # TAB 4: Correlations
        with tab4:
            st.subheader("Correlation Analysis")
            
            # Encode categorical variables for correlation
            df_encoded = data.copy()
            for col in df_encoded.select_dtypes(include=['object']).columns:
                df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col].astype(str))
            
            # Correlation matrix
            corr_matrix = df_encoded.corr()
            
            # Heatmap
            fig = px.imshow(corr_matrix, 
                           text_auto=True, 
                           aspect='auto',
                           color_continuous_scale='RdBu_r',
                           title='Correlation Heatmap')
            st.plotly_chart(fig, use_container_width=True)
            
            # Correlation with Salary
            st.markdown("##### 🎯 Correlation with Salary")
            salary_corr = corr_matrix['Salary'].sort_values(ascending=False)
            
            fig = px.bar(x=salary_corr.values, y=salary_corr.index,
                        orientation='h',
                        title='Feature Correlation with Salary',
                        labels={'x': 'Correlation', 'y': 'Feature'})
            st.plotly_chart(fig, use_container_width=True)


# ============================================
# PAGE: MODEL TRAINING
# ============================================
elif page == "🤖 Model Training":
    
    st.markdown('<h1 class="main-header">🤖 Model Training</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    if data is not None:
        
        # Preprocessing
        st.subheader("⚙️ Data Preprocessing")
        
        with st.expander("📋 View Preprocessing Steps", expanded=True):
            
            # Handle missing values
            df = data.dropna()
            st.write(f"✅ Removed missing values: {len(data) - len(df)} rows")
            
            # Features and Target
            X = df.drop('Salary', axis=1)
            y = df['Salary']
            
            # One-hot encoding
            X_encoded = pd.get_dummies(X, drop_first=True)
            st.write(f"✅ One-hot encoding applied: {X_encoded.shape[1]} features")
            
            # Train-test split
            test_size = st.slider("Select Test Size", 0.1, 0.4, 0.2, 0.05)
            X_train, X_test, y_train, y_test = train_test_split(
                X_encoded, y, test_size=test_size, random_state=42
            )
            
            col1, col2 = st.columns(2)
            col1.metric("Training Samples", len(X_train))
            col2.metric("Testing Samples", len(X_test))
            
            # Feature Scaling
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            st.write("✅ Feature scaling applied (StandardScaler)")
        
        st.markdown("---")
        
        # Model Selection
        st.subheader("🎯 Select Models to Train")
        
        models_dict = get_regression_models()
        
        col1, col2 = st.columns(2)
        
        with col1:
            selected_models = st.multiselect(
                "Choose Models",
                list(models_dict.keys()),
                default=['Linear Regression', 'Random Forest', 'Ridge Regression']
            )
        
        with col2:
            use_scaling = st.checkbox("Use Feature Scaling", value=True)
            st.info("💡 Scaling is recommended for Linear, Ridge, Lasso, K-Neighbors, and SVR models")
        
        # Model descriptions
        with st.expander("📖 Model Descriptions"):
            st.markdown("""
            | Model | Description | Best For |
            |-------|-------------|----------|
            | **Linear Regression** | Fits a linear relationship between features and target | Simple linear relationships |
            | **Ridge Regression** | Linear regression with L2 regularization | Multicollinearity issues |
            | **Lasso Regression** | Linear regression with L1 regularization | Feature selection |
            | **Decision Tree** | Tree-based model that splits data | Non-linear patterns |
            | **Random Forest** | Ensemble of multiple decision trees | Complex patterns, overfitting prevention |
            | **K-Neighbors** | Predicts based on k nearest neighbors | Similar data points |
            | **SVR** | Support Vector Machine for regression | High-dimensional data |
            """)
        
        # Train Models
        if st.button("🚀 Train Selected Models", use_container_width=True):
            
            if len(selected_models) == 0:
                st.warning("⚠️ Please select at least one model!")
            else:
                results = {}
                trained_models = {}
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, model_name in enumerate(selected_models):
                    status_text.text(f"🔄 Training {model_name}...")
                    
                    # Create a fresh instance of the model
                    model = get_regression_models()[model_name]
                    
                    if use_scaling:
                        metrics, y_pred, trained_model = evaluate_model(
                            model, X_train_scaled, X_test_scaled, y_train, y_test
                        )
                    else:
                        metrics, y_pred, trained_model = evaluate_model(
                            model, X_train, X_test, y_train, y_test
                        )
                    
                    results[model_name] = metrics
                    trained_models[model_name] = {
                        'model': trained_model,
                        'predictions': y_pred
                    }
                    
                    progress_bar.progress((i + 1) / len(selected_models))
                
                status_text.text("✅ Training Complete!")
                
                # Store in session state
                st.session_state['results'] = results
                st.session_state['trained_models'] = trained_models
                st.session_state['X_train'] = X_train
                st.session_state['X_test'] = X_test
                st.session_state['y_test'] = y_test
                st.session_state['scaler'] = scaler
                st.session_state['feature_columns'] = X_encoded.columns.tolist()
                st.session_state['use_scaling'] = use_scaling
                
                # Display Results
                st.markdown("---")
                st.subheader("📊 Training Results")
                
                # Results Table
                results_df = pd.DataFrame(results).T
                results_df = results_df.round(4)
                st.dataframe(results_df, use_container_width=True)
                
                # Best Model
                best_model = max(results.keys(), key=lambda x: results[x]['R2 Score'])
                best_r2 = results[best_model]['R2 Score']
                
                st.success(f"🏆 Best Model: **{best_model}** with R² Score: **{best_r2:.4f}** ({best_r2*100:.2f}%)")
                
                # Visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    # R2 Score Comparison
                    fig = px.bar(
                        x=list(results.keys()),
                        y=[results[m]['R2 Score'] for m in results.keys()],
                        title='R² Score Comparison',
                        labels={'x': 'Model', 'y': 'R² Score'},
                        color=[results[m]['R2 Score'] for m in results.keys()],
                        color_continuous_scale='Viridis'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # RMSE Comparison
                    fig = px.bar(
                        x=list(results.keys()),
                        y=[results[m]['RMSE'] for m in results.keys()],
                        title='RMSE Comparison (Lower is Better)',
                        labels={'x': 'Model', 'y': 'RMSE'},
                        color=[results[m]['RMSE'] for m in results.keys()],
                        color_continuous_scale='Reds_r'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Actual vs Predicted for Best Model
                st.subheader(f"📈 Actual vs Predicted ({best_model})")
                
                y_pred_best = trained_models[best_model]['predictions']
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=y_test.values, y=y_pred_best,
                    mode='markers',
                    name='Predictions',
                    marker=dict(color='#667eea', size=8, opacity=0.6)
                ))
                fig.add_trace(go.Scatter(
                    x=[y_test.min(), y_test.max()],
                    y=[y_test.min(), y_test.max()],
                    mode='lines',
                    name='Perfect Prediction',
                    line=dict(color='red', dash='dash')
                ))
                fig.update_layout(
                    xaxis_title='Actual Salary',
                    yaxis_title='Predicted Salary',
                    title=f'Actual vs Predicted Salary - {best_model}'
                )
                st.plotly_chart(fig, use_container_width=True)


# ============================================
# PAGE: MODEL COMPARISON
# ============================================
elif page == "📈 Model Comparison":
    
    st.markdown('<h1 class="main-header">📈 Model Comparison</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    if 'results' in st.session_state and 'trained_models' in st.session_state:
        
        results = st.session_state['results']
        trained_models = st.session_state['trained_models']
        y_test = st.session_state['y_test']
        
        # Comparison Table
        st.subheader("📊 Performance Metrics Comparison")
        
        results_df = pd.DataFrame(results).T
        results_df = results_df.round(4)
        results_df = results_df.sort_values('R2 Score', ascending=False)
        
        # Add ranking
        results_df['Rank'] = range(1, len(results_df) + 1)
        results_df = results_df[['Rank'] + [col for col in results_df.columns if col != 'Rank']]
        
        st.dataframe(results_df.style.background_gradient(cmap='RdYlGn', subset=['R2 Score'])
                     .background_gradient(cmap='RdYlGn_r', subset=['RMSE', 'MAE', 'MSE', 'MAPE']),
                     use_container_width=True)
        
        st.markdown("---")
        
        # Visualizations
        st.subheader("📊 Visual Comparison")
        
        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Metrics", "📈 Predictions", "🎯 Residuals", "📋 Feature Importance"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                # R2 Score
                fig = px.bar(
                    results_df,
                    x=results_df.index,
                    y='R2 Score',
                    title='R² Score by Model (Higher is Better)',
                    color='R2 Score',
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(xaxis_title='Model')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # RMSE
                fig = px.bar(
                    results_df,
                    x=results_df.index,
                    y='RMSE',
                    title='RMSE by Model (Lower is Better)',
                    color='RMSE',
                    color_continuous_scale='Reds_r'
                )
                fig.update_layout(xaxis_title='Model')
                st.plotly_chart(fig, use_container_width=True)
            
            # All Metrics Comparison
            st.markdown("##### 📊 All Metrics Comparison")
            
            metrics_to_plot = ['R2 Score', 'MAE', 'RMSE']
            fig = go.Figure()
            
            for metric in metrics_to_plot:
                fig.add_trace(go.Bar(
                    name=metric,
                    x=results_df.index,
                    y=results_df[metric],
                    text=results_df[metric].round(2),
                    textposition='auto'
                ))
            
            fig.update_layout(
                barmode='group',
                title='Metrics Comparison Across Models',
                xaxis_title='Model',
                yaxis_title='Value'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.markdown("##### 📈 Actual vs Predicted Comparison")
            
            selected_model = st.selectbox(
                "Select Model",
                list(trained_models.keys()),
                key='pred_model'
            )
            
            y_pred = trained_models[selected_model]['predictions']
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Scatter plot
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=y_test.values, y=y_pred, mode='markers',
                    marker=dict(color='blue', opacity=0.5, size=8),
                    name='Predictions'
                ))
                fig.add_trace(go.Scatter(
                    x=[y_test.min(), y_test.max()],
                    y=[y_test.min(), y_test.max()],
                    mode='lines', line=dict(color='red', dash='dash'),
                    name='Perfect Prediction'
                ))
                fig.update_layout(
                    title=f'Actual vs Predicted - {selected_model}',
                    xaxis_title='Actual Salary',
                    yaxis_title='Predicted Salary'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Error distribution
                errors = y_test.values - y_pred
                fig = px.histogram(x=errors, nbins=30,
                                  title='Prediction Error Distribution',
                                  labels={'x': 'Error (Actual - Predicted)'})
                fig.add_vline(x=0, line_dash="dash", line_color="red")
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.markdown("##### 🎯 Residual Analysis")
            
            selected_model_res = st.selectbox(
                "Select Model for Residuals",
                list(trained_models.keys()),
                key='residual_model'
            )
            
            y_pred_res = trained_models[selected_model_res]['predictions']
            residuals = y_test.values - y_pred_res
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.scatter(x=y_pred_res, y=residuals,
                               labels={'x': 'Predicted Salary', 'y': 'Residuals'},
                               title='Residuals vs Predicted Values')
                fig.add_hline(y=0, line_dash="dash", line_color="red")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.histogram(x=residuals, nbins=30,
                                  labels={'x': 'Residuals'},
                                  title='Residual Distribution')
                fig.add_vline(x=0, line_dash="dash", line_color="red")
                st.plotly_chart(fig, use_container_width=True)
            
            # Residual statistics
            st.markdown("##### 📊 Residual Statistics")
            res_stats = pd.DataFrame({
                'Metric': ['Mean', 'Std Dev', 'Min', 'Max', 'Median'],
                'Value': [
                    f"${np.mean(residuals):,.2f}",
                    f"${np.std(residuals):,.2f}",
                    f"${np.min(residuals):,.2f}",
                    f"${np.max(residuals):,.2f}",
                    f"${np.median(residuals):,.2f}"
                ]
            })
            st.dataframe(res_stats, use_container_width=True)
        
        with tab4:
            st.markdown("##### 📋 Feature Importance")
            
            # Models that support feature importance
            importance_models = [m for m in trained_models.keys() 
                               if hasattr(trained_models[m]['model'], 'feature_importances_') 
                               or hasattr(trained_models[m]['model'], 'coef_')]
            
            if importance_models:
                selected_model_imp = st.selectbox(
                    "Select Model",
                    importance_models,
                    key='importance_model'
                )
                
                feature_names = st.session_state['feature_columns']
                importance_df = get_feature_importance(
                    trained_models[selected_model_imp]['model'],
                    feature_names
                )
                
                if importance_df is not None:
                    fig = px.bar(importance_df.head(15), 
                               x='Importance', y='Feature',
                               orientation='h',
                               title=f'Top 15 Feature Importances - {selected_model_imp}',
                               color='Importance',
                               color_continuous_scale='Viridis')
                    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show data
                    with st.expander("📋 View All Feature Importances"):
                        st.dataframe(importance_df, use_container_width=True)
            else:
                st.info("ℹ️ Train Decision Tree, Random Forest, or Linear models to see feature importance.")
        
        # Best Model Summary
        st.markdown("---")
        st.subheader("🏆 Best Model Summary")
        
        best_model = results_df.index[0]
        best_metrics = results[best_model]
        
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("🏆 Model", best_model)
        col2.metric("R² Score", f"{best_metrics['R2 Score']:.4f}")
        col3.metric("RMSE", f"${best_metrics['RMSE']:,.2f}")
        col4.metric("MAE", f"${best_metrics['MAE']:,.2f}")
        col5.metric("MAPE", f"{best_metrics['MAPE']:.2f}%")
    
    else:
        st.warning("⚠️ Please train models first in the 'Model Training' page!")
        st.info("👉 Go to 🤖 Model Training → Select models → Click 'Train Selected Models'")


# ============================================
# PAGE: PREDICT SALARY
# ============================================
elif page == "🔮 Predict Salary":
    
    st.markdown('<h1 class="main-header">🔮 Predict Salary</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    if data is not None and 'trained_models' in st.session_state:
        
        trained_models = st.session_state['trained_models']
        scaler = st.session_state['scaler']
        feature_columns = st.session_state['feature_columns']
        use_scaling = st.session_state.get('use_scaling', True)
        
        st.subheader("📝 Enter Employee Details")
        
        # Clean data for dropdown options
        clean_data = data.dropna()
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("👤 Age", min_value=18, max_value=70, value=30)
            
            gender_options = clean_data['Gender'].unique().tolist()
            gender = st.selectbox("⚧️ Gender", gender_options)
            
            education_options = clean_data['Education Level'].unique().tolist()
            education = st.selectbox("🎓 Education Level", education_options)
        
        with col2:
            job_title_options = sorted(clean_data['Job Title'].unique().tolist())
            job_title = st.selectbox("💼 Job Title", job_title_options)
            
            experience = st.number_input("📅 Years of Experience", 
                                        min_value=0, max_value=50, value=5)
            
            model_choice = st.selectbox("🤖 Select Model for Prediction", 
                                       list(trained_models.keys()))
        
        st.markdown("---")
        
        # Prediction Button
        if st.button("💰 Predict Salary", use_container_width=True):
            
            # Create input dataframe
            input_data = pd.DataFrame({
                'Age': [age],
                'Gender': [gender],
                'Education Level': [education],
                'Job Title': [job_title],
                'Years of Experience': [experience]
            })
            
            # Display input
            st.markdown("##### 📋 Input Summary")
            st.dataframe(input_data, use_container_width=True)
            
            # One-hot encoding
            input_encoded = pd.get_dummies(input_data, drop_first=True)
            input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)
            
            # Scale features if needed
            if use_scaling:
                input_processed = scaler.transform(input_encoded)
            else:
                input_processed = input_encoded.values
            
            # Get the selected model
            model = trained_models[model_choice]['model']
            
            # Predict
            prediction = model.predict(input_processed)[0]
            
            # Display prediction
            st.markdown("---")
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 2rem; border-radius: 15px; text-align: center; color: white;">
                <h2>💰 Predicted Salary</h2>
                <h1 style="font-size: 3rem;">${prediction:,.2f}</h1>
                <p>Using {model_choice}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Additional insights
            st.markdown("---")
            st.subheader("📊 Prediction Insights")
            
            col1, col2, col3 = st.columns(3)
            
            # Compare with average
            avg_salary = clean_data['Salary'].mean()
            diff_from_avg = prediction - avg_salary
            diff_percent = (diff_from_avg / avg_salary) * 100
            
            col1.metric(
                "vs Average Salary",
                f"${diff_from_avg:,.2f}",
                f"{diff_percent:+.1f}%"
            )
            
            # Compare with same education level
            edu_data = clean_data[clean_data['Education Level'] == education]
            if len(edu_data) > 0:
                edu_avg = edu_data['Salary'].mean()
                edu_diff = prediction - edu_avg
                col2.metric(
                    f"vs {education} Avg",
                    f"${edu_diff:,.2f}",
                    f"{(edu_diff/edu_avg)*100:+.1f}%"
                )
            
            # Compare with similar experience
            exp_data = clean_data[
                (clean_data['Years of Experience'] >= experience - 2) & 
                (clean_data['Years of Experience'] <= experience + 2)
            ]
            if len(exp_data) > 0:
                exp_avg = exp_data['Salary'].mean()
                exp_diff = prediction - exp_avg
                col3.metric(
                    f"vs ~{experience}yr Exp Avg",
                    f"${exp_diff:,.2f}",
                    f"{(exp_diff/exp_avg)*100:+.1f}%"
                )
            
            # Predict with all models
            st.markdown("---")
            st.subheader("🔄 Predictions from All Models")
            
            all_predictions = {}
            for model_name, model_data in trained_models.items():
                try:
                    pred = model_data['model'].predict(input_processed)[0]
                    all_predictions[model_name] = pred
                except Exception as e:
                    pass
            
            if all_predictions:
                # Sort by prediction
                all_predictions = dict(sorted(all_predictions.items(), key=lambda x: x[1], reverse=True))
                
                # Create dataframe
                pred_df = pd.DataFrame({
                    'Model': list(all_predictions.keys()),
                    'Predicted Salary': list(all_predictions.values())
                })
                pred_df['Predicted Salary'] = pred_df['Predicted Salary'].apply(lambda x: f"${x:,.2f}")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    fig = px.bar(
                        x=list(all_predictions.keys()),
                        y=list(all_predictions.values()),
                        title='Salary Prediction by All Models',
                        labels={'x': 'Model', 'y': 'Predicted Salary'},
                        color=list(all_predictions.values()),
                        color_continuous_scale='Viridis'
                    )
                    fig.add_hline(y=prediction, line_dash="dash", line_color="red",
                                 annotation_text=f"Selected: ${prediction:,.2f}")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("##### 📋 All Predictions")
                    st.dataframe(pred_df, use_container_width=True)
                
                # Statistics
                col1, col2, col3 = st.columns(3)
                col1.metric("Min Prediction", f"${min(all_predictions.values()):,.2f}")
                col2.metric("Avg Prediction", f"${np.mean(list(all_predictions.values())):,.2f}")
                col3.metric("Max Prediction", f"${max(all_predictions.values()):,.2f}")
    
    else:
        st.warning("⚠️ Please train models first!")
        st.info("👉 Go to 🤖 Model Training → Select models → Click 'Train Selected Models'")


# ============================================
# PAGE: ABOUT
# ============================================
elif page == "📋 About":
    
    st.markdown('<h1 class="main-header">📋 About This Project</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🎯 Project Overview")
        st.markdown("""
        This is a **Machine Learning Dashboard** for predicting employee salaries based on 
        various factors. The project implements multiple regression algorithms and provides 
        comprehensive model comparison and visualization tools.
        
        ### 🌟 Key Features
        
        - **7 Regression Models** - Compare different ML algorithms
        - **Interactive Visualizations** - Plotly-powered charts
        - **Real-time Predictions** - Instant salary predictions
        - **Model Comparison** - Side-by-side performance analysis
        - **Feature Importance** - Understand what drives salaries
        - **Residual Analysis** - Validate model assumptions
        """)
        
        st.subheader("🛠️ Technical Stack")
        
        tech_col1, tech_col2, tech_col3 = st.columns(3)
        
        with tech_col1:
            st.markdown("""
            **Frontend**
            - Streamlit
            - Plotly
            - Custom CSS
            """)
        
        with tech_col2:
            st.markdown("""
            **Backend**
            - Python 3.x
            - Pandas
            - NumPy
            """)
        
        with tech_col3:
            st.markdown("""
            **ML Libraries**
            - Scikit-learn
            - Matplotlib
            - Seaborn
            """)
    
    with col2:
        st.subheader("📊 Models Used")
        st.markdown("""
        | # | Model |
        |---|-------|
        | 1 | Linear Regression |
        | 2 | Ridge Regression |
        | 3 | Lasso Regression |
        | 4 | Decision Tree |
        | 5 | Random Forest |
        | 6 | K-Neighbors |
        | 7 | SVR |
        """)
    
    st.markdown("---")
    
    # Model explanations
    st.subheader("📖 Model Explanations")
    
    with st.expander("🔵 Linear Regression"):
        st.markdown("""
        **Linear Regression** finds a linear relationship between input features and the target variable.
        
        - **Equation**: y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
        - **Pros**: Simple, interpretable, fast
        - **Cons**: Assumes linear relationship
        """)
    
    with st.expander("🔵 Ridge Regression (L2)"):
        st.markdown("""
        **Ridge Regression** adds L2 regularization to prevent overfitting.
        
        - **Regularization**: Adds penalty = α × Σ(β²)
        - **Pros**: Handles multicollinearity, prevents overfitting
        - **Cons**: Keeps all features (no selection)
        """)
    
    with st.expander("🔵 Lasso Regression (L1)"):
        st.markdown("""
        **Lasso Regression** adds L1 regularization for feature selection.
        
        - **Regularization**: Adds penalty = α × Σ|β|
        - **Pros**: Performs feature selection, sparse models
        - **Cons**: May drop important features
        """)
    
    with st.expander("🟢 Decision Tree"):
        st.markdown("""
        **Decision Tree** creates a tree structure by splitting data based on features.
        
        - **Method**: Recursive binary splits
        - **Pros**: Non-linear, interpretable, no scaling needed
        - **Cons**: Prone to overfitting
        """)
    
    with st.expander("🟢 Random Forest"):
        st.markdown("""
        **Random Forest** is an ensemble of multiple decision trees.
        
        - **Method**: Bagging (Bootstrap Aggregating)
        - **Pros**: Reduces overfitting, handles non-linearity
        - **Cons**: Less interpretable, slower
        """)
    
    with st.expander("🟣 K-Neighbors Regressor"):
        st.markdown("""
        **K-Neighbors** predicts based on the k nearest data points.
        
        - **Method**: Instance-based learning
        - **Pros**: Simple, no training needed
        - **Cons**: Slow for large datasets, sensitive to scale
        """)
    
    with st.expander("🟣 Support Vector Regressor (SVR)"):
        st.markdown("""
        **SVR** uses kernel tricks to find optimal hyperplane.
        
        - **Method**: Kernel-based mapping
        - **Pros**: Effective in high dimensions
        - **Cons**: Slow for large datasets, needs scaling
        """)
    
    st.markdown("---")
    
    # How to Use
    st.subheader("📖 How to Use")
    
    st.markdown("""
    <div class="info-box">
    <h4>Step-by-Step Guide</h4>
    <ol>
        <li><b>Data Exploration:</b> Explore the dataset, view statistics, and analyze distributions</li>
        <li><b>Model Training:</b> Select regression models and train them on the dataset</li>
        <li><b>Model Comparison:</b> Compare performance metrics across all trained models</li>
        <li><b>Predict Salary:</b> Enter employee details to get salary predictions</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Contact
    st.subheader("👤 Developer Info")
    
    st.markdown("""
    **[Janhavi Maurya]**  
    - 🔗 LinkedIn: [https://www.linkedin.com/in/janhavi-maurya-8430442b5/]
    - 💻 GitHub: [https://github.com/janhavi-2011]
    - 📧 Email: [janhavimaurya8738@gmail.com]
    
    ---
    
    *This project was developed as a demonstration of machine learning skills 
    for salary prediction using multiple regression algorithms.*
    """)
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: gray;">
        <p>Made with ❤️ using Streamlit | © 2024</p>
    </div>
    """, unsafe_allow_html=True)


# ============================================
# FOOTER
# ============================================
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="text-align: center; font-size: 0.8rem; color: gray;">
    <p>💰 Salary Prediction Dashboard</p>
    <p>Built with Streamlit</p>
</div>

""", unsafe_allow_html=True)
