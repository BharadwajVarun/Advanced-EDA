#import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
import numpy as np

# Load the dataset
file_path = "Breast_Cancer.csv"
df = pd.read_csv(file_path)

# 1. Remove Duplicates
df.drop_duplicates(inplace=True)

# 2. Summary Statistics and Skewness/Kurtosis
print("\nSummary Statistics:\n", df.describe())
numerical_cols = df.select_dtypes(include=[np.number])  # Select only numerical columns
print("\nSkewness:\n", numerical_cols.skew())
print("\nKurtosis:\n", numerical_cols.apply(kurtosis))

# 3. Feature Distribution Analysis
plt.figure(figsize=(15, 10))
for i, col in enumerate(["Age", "Tumor Size", "Regional Node Examined", "Reginol Node Positive", "Survival Months"]):
    plt.subplot(2, 3, i+1)
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f"Distribution of {col}")
plt.tight_layout()
plt.show()

# 4. Boxplot for Outlier Detection
plt.figure(figsize=(15, 5))
sns.boxplot(data=df[["Age", "Tumor Size", "Regional Node Examined", "Reginol Node Positive", "Survival Months"]])
plt.title("Boxplot of Numeric Features")
plt.show()

# 5. Correlation Heatmap
plt.figure(figsize=(10, 6))
# Select only numerical features for correlation analysis
numerical_df = df.select_dtypes(include=np.number)  
corr_matrix = numerical_df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Feature Correlation Heatmap")
plt.show()

df.filter(like="T Stage").head()
df = df.rename(columns=lambda x: x.strip())

# 6. Relationship between Categorical and Numeric Features
plt.figure(figsize=(15, 10))
for i, col in enumerate(["Race", "Marital Status", "T Stage", "N Stage", "6th Stage", "Status"]):
    plt.subplot(2, 3, i+1)
    # Replace spaces in column names with underscores before accessing them if they exist in the name
    cleaned_col = col.replace(" ", "_") if " " in col else col
    # Use the original column name or the cleaned one depending on presence of space
    sns.boxplot(x=df[col if " " in col else cleaned_col], y=df["Survival Months"], data=df)  
    plt.xticks(rotation=45)
    plt.title(f"Survival Months vs {col}")
plt.tight_layout()
plt.show()

# 7. Pairplot to Examine Relationships
sns.pairplot(df[["Age", "Tumor Size", "Regional Node Examined", "Reginol Node Positive", "Survival Months"]])
plt.show()

# 8. Outlier Detection using Z-score
from scipy.stats import zscore
df_zscore = df.select_dtypes(include=[np.number]).apply(zscore)
outliers = (df_zscore.abs() > 3).sum()
print("\nOutliers per column (Z-score > 3):\n", outliers)

# 9. Missing Data Visualization
plt.figure(figsize=(12, 6))
sns.heatmap(df.isnull(), cmap='viridis', cbar=False, yticklabels=False)
plt.title("Missing Values Heatmap")
plt.show()

# 10. Feature Importance using Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Encode categorical variables
df_encoded = df.copy()
categorical_cols = df_encoded.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col])

X = df_encoded.drop(columns=['Status'])  # Assuming 'Status' is the target variable
y = df_encoded['Status']
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

feature_importances = pd.Series(rf.feature_importances_, index=X.columns)
plt.figure(figsize=(10, 5))
feature_importances.nlargest(10).plot(kind='barh')
plt.title("Top 10 Feature Importances")
plt.show()


print("Advanced EDA for Genomic Data Aanlysis is finished") 