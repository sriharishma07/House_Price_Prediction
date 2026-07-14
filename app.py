import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ==========================================
# STEP 1: DUMMY DATA CREATION (Replace with your actual CSV file)
# ==========================================
# Real-world usecase equivalent data structure
np.random.seed(42)
n_samples = 500

data = pd.DataFrame({
    'area': np.random.randint(500, 5000, n_samples),
    'bedrooms': np.random.randint(1, 6, n_samples),
    'bathrooms': np.random.randint(1, 4, n_samples),
    'location': np.random.choice(['Chennai-Central', 'Adyar', 'Velachery', 'Tambaram'], n_samples),
    'price': 0 # Dummy initialization
})

# Missing value rendering for simulation purpose (handling in step 2)
data.loc[np.random.choice(data.index, 15), 'area'] = np.nan 

# Calculating actual price logic with some noise
data['price'] = (data['area'].fillna(2000) * 3000) + (data['bedrooms'] * 50000) + \
                (data['bathrooms'] * 30000) + np.random.normal(0, 50000, n_samples)

print("--- Initial Raw Data Glimpse ---")
print(data.head())
print("\nMissing values check before pipeline:\n", data.isnull().sum())

# ==========================================
# STEP 2: DATA PREPROCESSING & MODEL PIPELINE
# ==========================================
# Independent & Dependent features separation
X = data.drop('price', axis=1)
y = data['price']

# Handling Numerical & Categorical boundaries separately
numeric_features = ['area', 'bedrooms', 'bathrooms']
categorical_features = ['location']

# Fill Missing Values explicitly for 'area' before scaling
X['area'] = X['area'].fillna(X['area'].median())

# Transformer setups
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ])

# Creating end-to-end Pipeline with Linear Regression Regressor Model
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Train-Test Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# STEP 3: TRAINING THE MODEL
# ==========================================
model_pipeline.fit(X_train, y_train)
print("\n>>> Model Training Completed successfully! <<<")

# ==========================================
# STEP 4: EVALUATION
# ==========================================
y_pred = model_pipeline.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n--- Model Evaluation Metrics ---")
print(f"Mean Squared Error (MSE)   : {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R²) Score       : {r2:.4f} (Accuracy Benchmark)")

# ==========================================
# STEP 5: DATA TRENDS & PREDICTIONS VISUALIZATION
# ==========================================
plt.figure(figsize=(12, 5))

# Plot 1: Actual vs Predicted Prices
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, color='blue', alpha=0.6, edgecolors='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title('Actual vs Predicted House Prices')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.grid(True)

# Plot 2: Residual Distribution
plt.subplot(1, 2, 2)
residuals = y_test - y_pred
sns.histplot(residuals, kde=True, color='purple')
plt.title('Residuals Error Distribution')
plt.xlabel('Prediction Error')
plt.ylabel('Count')
plt.grid(True)

plt.tight_layout()
print("\nDisplaying graphs... Close the plot window to terminate the script.")
plt.show()
