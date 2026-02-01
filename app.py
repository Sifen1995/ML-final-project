import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# 1. Load Dataset
# Note: UCI dataset uses semicolons ';' as separators
try:
    df = pd.read_csv('student-mat.csv', sep=';')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: Please ensure 'student-mat.csv' is in your directory.")

# --- 2. BASIC INSPECTION ---
print("\n--- Dataset Info ---")
print(df.info())
print("\n--- Missing Values ---")
print(df.isnull().sum())

# --- 3. TARGET DISTRIBUTION (G3) ---
plt.figure(figsize=(8, 5))
sns.histplot(df['G3'], kde=True, color='teal', bins=20)
plt.title('Distribution of Final Grades (G3)')
plt.xlabel('Final Grade')
plt.ylabel('Frequency')
plt.savefig('g3_distribution.png')
plt.show()


# --- 4. CORRELATION HEATMAP ---
#We only calculate correlation for numeric columns

plt.figure(figsize=(12, 10))
numeric_df = df.select_dtypes(include=[np.number])
correlation_matrix = numeric_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap of Numeric Features')
plt.savefig('correlation_heatmap.png')
plt.show()


# student one ----------

# --- 5. KEY RELATIONSHIPS ---

# A. Study Time vs G3 (Box Plot)

plt.figure(figsize=(8, 6))
sns.boxplot(x='studytime', y='G3', data=df, hue='studytime', palette='Set2', legend=False)
plt.title('Impact of Weekly Study Time on Final Grade')
plt.xlabel('Study Time (1: <2hrs, 2: 2-5hrs, 3: 5-10hrs, 4: >10hrs)')
plt.show()

#B. Absences vs G3 (Scatter Plot)
plt.figure(figsize=(8, 6))
sns.scatterplot(x='absences', y='G3', data=df, alpha=0.6, color='red')
plt.title('Absences vs Final Grade')
plt.savefig('absences_vs_g3.png')
plt.show()

#C. G1 & G2 vs G3 (Regression Plots)
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
sns.regplot(ax=axes[0], x='G1', y='G3', data=df, scatter_kws={'alpha':0.3})
axes[0].set_title('First Period Grade (G1) vs G3')
sns.regplot(ax=axes[1], x='G2', y='G3', data=df, scatter_kws={'alpha':0.3}, color='green')
axes[1].set_title('Second Period Grade (G2) vs G3')
plt.savefig('grades_progression.png')
plt.show()

#D. Categorical Impact (e.g., Internet Access)
plt.figure(figsize=(8, 6))
sns.barplot(x='internet', y='G3', data=df, capsize=.1)
plt.title('Internet Access vs Average Final Grade')
plt.savefig('internet_impact.png')
plt.show()

print("\nEDA Complete. Plots saved to local directory.")

#student two -----------

# --- 6. DATA PREPROCESSING ---


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

# Reload data to ensure a fresh start
df = pd.read_csv('student-mat.csv', sep=';')

# 5.1 Cleaning
# Check and remove duplicates
initial_count = len(df)
df = df.drop_duplicates()
print(f"Removed {initial_count - len(df)} duplicate rows.")

# Handle missing values (UCI is clean, but we show the logic)
df = df.fillna(df.median(numeric_only=True))


# 5.2 Feature Engineering
# Create a Pass/Fail label for classification (Bonus/Alternative)
# Passing grade is usually 10/20
df['pass_fail'] = df['G3'].apply(lambda x: 1 if x >= 10 else 0)

# Dropping irrelevant features
# We might drop 'address' if we believe it doesn't impact performance, 
# but usually, we keep most in this dataset.
# IMPORTANT: If you want a 'Predictive' model that works before the final exam,
# consider dropping G1 and G2. For now, we keep them.


# 5.3 Encoding
# Binary Encoding (0 and 1)
le = LabelEncoder()
binary_cols = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'schoolsup', 
               'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']

for col in binary_cols:
    df[col] = le.fit_transform(df[col])

# One-Hot Encoding for multi-category features (Mjob, Fjob, reason, guardian)
df = pd.get_dummies(df, columns=['Mjob', 'Fjob', 'reason', 'guardian'], drop_first=True)


# 5.4 & 5.5 Train/Test Split and Scaling
# Define Features (X) and Target (y)
X = df.drop(['G3', 'pass_fail'], axis=1) # Features
y = df['G3']                             # Regression Target

# 80/20 Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training set size: {X_train.shape}")
print(f"Testing set size: {X_test.shape}")


#student three -----------

# --- 7. MODEL TRAINING & EVALUATION ---



from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np

# Define the models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "SVM Regressor": SVR(kernel='rbf')
}

# Dictionary to store results
results = {}

print("--- Starting Model Training with Cross-Validation ---")

for name, model in models.items():
    # Perform 5-Fold Cross-Validation
    # We use 'neg_mean_absolute_error' because sklearn maximizes scores
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='neg_mean_absolute_error')
    avg_mae = -cv_scores.mean()
    
    # Fit on the full training set
    model.fit(X_train_scaled, y_train)
    
    # Predict on test set
    predictions = model.predict(X_test_scaled)
    
    # Calculate Metrics
    test_mae = mean_absolute_error(y_test, predictions)
    test_r2 = r2_score(y_test, predictions)
    
    results[name] = {'CV MAE': avg_mae, 'Test MAE': test_mae, 'Test R2': test_r2}
    
    print(f"{name}: CV MAE = {avg_mae:.3f} | Test MAE = {test_mae:.3f} | Test R2 = {test_r2:.3f}")

#student four -----------

import pandas as pd
results_df = pd.DataFrame(results).T
print("\nFinal Comparison Table:")
print(results_df)



import matplotlib.pyplot as plt
import seaborn as sns

rf_model = models["Random Forest"]
rf_preds = rf_model.predict(X_test_scaled)

# 1. Visualization: Actual vs Predicted (Best Model - likely RF or Linear)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y= rf_preds, alpha=0.6, color='blue', label='Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
plt.title('Random Forest: Actual vs Predicted Grades (G3)')
plt.xlabel('Actual Grade')
plt.ylabel('Predicted Grade')
plt.legend()
plt.savefig('actual_vs_predicted.png')
plt.show()

# 2. Feature Importance (Random Forest)
importances = rf_model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, hue='Feature', palette='viridis', legend=False)
plt.title('Top 10 Most Important Features predicting G3')
plt.savefig('feature_importance.png')
plt.show()

#student five -----------