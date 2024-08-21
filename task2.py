
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


data = pd.read_csv('your_dataset.csv')


print(data.head())


numerical_features = data.select_dtypes(include=['int64', 'float64']).columns
categorical_features = data.select_dtypes(include=['object']).columns


numerical_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='mean'))   # Impute missing values with mean
])


categorical_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='most_frequent')),  # Impute missing with most frequent value
    ('onehot', OneHotEncoder(handle_unknown='ignore'))    # One-hot encode categorical features
])


preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ])


pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('scaler', StandardScaler()),               # Standardize features by removing the mean and scaling to unit variance
    ('pca', PCA(n_components=0.95))             # Reduce dimensionality while preserving 95% variance
])


X = data.drop('target', axis=1)  # Here 'target' is a placeholder; replace with your actual target column
y = data['target']               # Replace with your actual target column name
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


pipeline.fit(X_train)


X_train_processed = pipeline.transform(X_train)
X_test_processed = pipeline.transform(X_test)

# Inspect the processed data shape
print('Processed training data shape:', X_train_processed.shape)
print('Processed testing data shape:', X_test_processed.shape)
