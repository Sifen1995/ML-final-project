import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns














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