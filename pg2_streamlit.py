import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Streamlit Page Configurations
st.set_page_config(page_title="Advanced EDA for Genomic Data Analysis", layout="wide")
st.title("🔬 Advanced EDA for Genomic Data Analysis")
st.markdown("### Upload your dataset to begin analysis")

# File Upload
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.write(df.head())
    
    # Structure Investigation
    st.subheader("Dataset Structure")
    st.write("Shape of Dataset:", df.shape)
    st.write("Data Types:", df.dtypes)
    
    # Quality Investigation
    st.subheader("Data Quality Check")
    st.write("Missing Values:")
    st.write(df.isnull().sum())
    st.write("Duplicate Rows:", df.duplicated().sum())
    
    # Feature Distribution
    st.subheader("Feature Distribution")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if not numeric_cols.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        df[numeric_cols].hist(ax=ax, bins=20, edgecolor='black', alpha=0.7)
        st.pyplot(fig)
    else:
        st.warning("No numeric columns found for distribution plots.")
    
    # Correlation Matrix
    st.subheader("Feature Correlation")
    if not numeric_cols.empty:
        corr_matrix = df[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", linewidths=0.5, ax=ax)
        st.pyplot(fig)
    else:
        st.warning("No numeric columns found for correlation analysis.")
    
    # Outliers Detection
    st.subheader("Outlier Detection")
    for col in numeric_cols:
        fig, ax = plt.subplots(figsize=(8,5))
        sns.boxplot(x=df[col], ax=ax, color='skyblue')
        st.pyplot(fig)
    
    # Categorical Value Count
    st.subheader("Categorical Feature Distribution")
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        fig, ax = plt.subplots()
        sns.countplot(y=df[col], palette="viridis", ax=ax)
        st.pyplot(fig)
    
    # Data Summary
    st.subheader("Data Summary Statistics")
    st.write(df.describe())
    
    st.success("EDA Completed Successfully! 🎉")
