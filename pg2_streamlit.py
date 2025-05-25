import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

st.set_page_config(page_title="Genetic Variation Visualizer", layout="wide")
st.title("🔬 Identifying Genetic Variations Through Visualization")

# Upload file
uploaded_file = st.file_uploader("Upload Breast Cancer Dataset CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df_original = df.copy()  # for PCA or other use

    # Identify numeric and categorical
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include='object').columns.tolist()

    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Dataset Overview", "Quality & Summary", "Visualizations",
        "Target Analysis", "Model Building", "Export Results"
    ])

    # TAB 1 - Dataset Overview
    with tab1:
        st.subheader("Dataset Preview")
        st.write(df.head())
        st.write("Shape:", df.shape)
        st.write("Data Types:")
        st.write(df.dtypes)

    # TAB 2 - Data Quality & Summary
    with tab2:
        st.subheader("Missing Values")
        st.write(df.isnull().sum())
        st.subheader("Duplicate Rows")
    
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            if st.button("🧹 Drop Duplicate Rows"):
                df.drop_duplicates(inplace=True)
                st.success(f"✅ {duplicate_count} duplicate rows dropped.")
            else:
                st.info("Click the button above to remove duplicates.")
        st.write(f"Total Duplicate Rows: {duplicate_count}")
        st.subheader("Descriptive Statistics")
        st.write(df.describe())

    # TAB 3 - Visualizations
    with tab3:
        st.subheader("Feature Distributions")
        for col in numeric_cols[:5]:
            fig, ax = plt.subplots()
            sns.histplot(df[col], bins=30, kde=True, ax=ax)
            st.pyplot(fig)

        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

        st.subheader("Outlier Detection")
        for col in numeric_cols[:3]:
            fig, ax = plt.subplots()
            sns.boxplot(x=df[col], ax=ax)
            st.pyplot(fig)

        st.subheader("Categorical Distributions")
        for col in categorical_cols:
            fig, ax = plt.subplots()
            sns.countplot(y=col, data=df, palette="Set2", ax=ax)
            st.pyplot(fig)

    # TAB 4 - Target Column Analysis
# TAB 4 - Target Column Analysis
    with tab4:
        st.subheader("Target Column Selection")
        target_col = st.selectbox("Select target (e.g., diagnosis)", df.columns)

        if target_col and df[target_col].nunique() <= 2:
            df_encoded = df.copy()

        # Label encode the target if needed
            if df_encoded[target_col].dtype == 'object':
                df_encoded[target_col] = LabelEncoder().fit_transform(df_encoded[target_col].astype(str))

        # Drop non-numeric columns for correlation
            df_numeric = df_encoded.select_dtypes(include=[np.number])

            if target_col not in df_numeric.columns:
                st.error("Target column must be numeric or convertible to numeric.")
            else:
            # Correlation with target
                corr = df_numeric.corr()[target_col].abs().sort_values(ascending=False)
                top_features = corr[1:6].index.tolist()

            st.write("Top 5 Correlated Features with Target:")
            st.write(top_features)

            for col in top_features:
                fig, ax = plt.subplots()
                sns.boxplot(x=target_col, y=col, data=df_encoded, palette="pastel", ax=ax)
                st.pyplot(fig)
        else:
            st.warning("Please select a binary target column.")


    # TAB 5 - Model Building
    with tab5:
        st.subheader("Train a Random Forest Classifier")

        if 'target_col' in locals() and target_col and df[target_col].nunique() <= 2:
            df_model = df.copy()

            # Encode all categorical columns (including target)
            for col in df_model.select_dtypes(include='object').columns:
                df_model[col] = LabelEncoder().fit_transform(df_model[col].astype(str))

            corr = df_model.corr()[target_col].abs().sort_values(ascending=False)
            top_features = corr[1:6].index.tolist()

            X = df_model[top_features]
            y = df_model[target_col]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42)

            model = RandomForestClassifier(random_state=42)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            st.subheader("Model Performance")
            st.write("Accuracy Score:", accuracy_score(y_test, y_pred))
            st.write("Classification Report:")
            st.text(classification_report(y_test, y_pred))
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots()
            sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap="Blues", ax=ax)
            st.pyplot(fig)
        
        # Feature importance
            st.subheader("📊 Feature Importance (Top 10)")
            importances = model.feature_importances_
            feat_importance = pd.Series(importances, index=X.columns).sort_values(ascending=False)[:10]
            fig, ax = plt.subplots()
            feat_importance.plot(kind='barh', ax=ax, color='purple')
            st.pyplot(fig)
        else:
            st.warning("You must choose a binary target column in the 'Target Analysis' tab.")


    # TAB 6 - Export
    with tab6:
        st.subheader("Export Results")

        csv_summary = df.describe().to_csv().encode('utf-8')
        st.download_button("📥 Download Summary Stats CSV", csv_summary, "eda_summary.csv", "text/csv")

        if 'df_encoded' in locals():
            csv_clean = df_encoded.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Processed Dataset", csv_clean, "cleaned_dataset.csv", "text/csv")

        st.success("✅ You can now download the cleaned & analyzed results.")
