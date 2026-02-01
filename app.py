import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns












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




import joblib

# Save the model and the scaler
joblib.dump(rf_model, 'student_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(X.columns.tolist(), 'features.pkl') # Save feature names