# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load dataset (assuming a CSV file)
# For demonstration purposes, replace 'your_dataset.csv' with your actual dataset file path
data = pd.read_csv('your_dataset.csv')

# Preview the data
print(data.head())

# Handle missing values
# We'll use SimpleImputer to replace missing values with the mean for numerical features
numerical_features = data.select_dtypes(include=['int64', 'float64']).columns
categorical_features = data.select_dtypes(include=['object']).columns

# Preprocessing for numerical data
numerical_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='mean'))   # Impute missing values with mean
])

# Preprocessing for categorical data
categorical_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='most_frequent')),  # Impute missing with most frequent value
    ('onehot', OneHotEncoder(handle_unknown='ignore'))    # One-hot encode categorical features
])

# Combine pipelines
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ])

# Full pipeline including scaling and dimensionality reduction
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('scaler', StandardScaler()),               # Standardize features by removing the mean and scaling to unit variance
    ('pca', PCA(n_components=0.95))             # Reduce dimensionality while preserving 95% variance
])

# Split dataset into training and testing sets
X = data.drop('target', axis=1)  # Here 'target' is a placeholder; replace with your actual target column
y = data['target']               # Replace with your actual target column name
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the pipeline on the training data
pipeline.fit(X_train)

# Transform both the training and testing data
X_train_processed = pipeline.transform(X_train)
X_test_processed = pipeline.transform(X_test)

# Inspect the processed data shape
print('Processed training data shape:', X_train_processed.shape)
print('Processed testing data shape:', X_test_processed.shape)