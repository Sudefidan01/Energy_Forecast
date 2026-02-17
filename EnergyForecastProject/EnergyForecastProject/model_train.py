import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import matplotlib.pyplot as plt
import numpy as np
import os


os.makedirs('images', exist_ok=True)


print("--- 1. DATA LOADING AND PREPROCESSING STARTED ---")


try:
    df = pd.read_csv("dataset.csv")
    print("Dataset loaded successfully. First 5 rows:")
    print(df.head())
except FileNotFoundError:
    print("ERROR: 'dataset.csv' not found. Please ensure the file is in the project folder.")
   
    exit()


df = df.rename(columns={'Unnamed: 0': 'RecordID'})


df['TxnDatetime'] = pd.to_datetime(df['TxnDate'] + ' ' + df['TxnTime'], format='%d %b %Y %H:%M:%S')


plt.figure(figsize=(15, 6))


df_plot = df.set_index('TxnDatetime').head(5000)

plt.plot(df_plot['Consumption'], label='Raw Consumption Data', color='red', alpha=0.7)
plt.title('Figure 3: Energy consumption time series before data cleaning and preprocessing')
plt.xlabel('Timestamp')
plt.ylabel('Consumption (kWh)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()


plt.savefig('images/ts_before_cleaning_f3.png') 
plt.close() 
print(f"Visualization saved: images/ts_before_cleaning_f3.png")


df['Hour'] = df['TxnDatetime'].dt.hour
df['DayOfWeek'] = df['TxnDatetime'].dt.dayofweek  
df['DayOfMonth'] = df['TxnDatetime'].dt.day


df_processed = df[['RecordID', 'Hour', 'DayOfWeek', 'DayOfMonth', 'Consumption']].copy()

print("\nPreprocessed data (features extracted):")
print(df_processed.head())


processed_data_file = "processed_consumption_data.csv"
df_processed.to_csv(processed_data_file, index=False)
print(f"\nProcessed data saved to '{processed_data_file}'.")

print("\n--- 2. MACHINE LEARNING MODEL TRAINING STARTED ---")


X = df_processed[['Hour', 'DayOfWeek', 'DayOfMonth']]
y = df_processed['Consumption']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
    "K-Nearest Neighbors Regressor": KNeighborsRegressor(n_neighbors=5),
    "Random Forest Regressor": RandomForestRegressor(random_state=42)
}

results = {}


for name, model in models.items():
  
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results[name] = {'Model': model, 'MAE': mae, 'R2': r2}


results_df = pd.DataFrame([
    (name, res['MAE'], res['R2'])
    for name, res in results.items()
], columns=['Model', 'Mean Absolute Error (MAE)', 'R-squared (R2)'])

print("\nModel Performance Results:")
print(results_df.to_markdown(index=False, floatfmt=".4f"))


best_model_name = results_df.sort_values(by='Mean Absolute Error (MAE)').iloc[0]['Model']
best_model = results[best_model_name]['Model']
model_filename = 'best_consumption_model.pkl'

joblib.dump(best_model, model_filename)

print(f"\nBest model ({best_model_name}) saved to '{model_filename}'.")
print("\n--- PROCESS COMPLETED. PROCEED TO THE FASTAPI APPLICATION. ---")


print("\n--- 3. MODEL VISUALIZATIONS STARTED (Actual vs Predicted) ---")


sample_size = 200
np.random.seed(42)
sample_indices = np.random.choice(len(X_test), sample_size, replace=False)

X_sample = X_test.iloc[sample_indices]
y_actual_sample = y_test.iloc[sample_indices]

plt.figure(figsize=(15, 8))
plt.plot(y_actual_sample.values, label='Actual Consumption', color='black', linewidth=2, alpha=0.7)

colors = {'Random Forest Regressor': 'blue', 'Linear Regression': 'red', 'Decision Tree Regressor': 'green', 'K-Nearest Neighbors Regressor': 'orange'}
i = 0
for name, res in results.items():
    model = res['Model']
    y_pred_sample = model.predict(X_sample)
    
    
    plt.plot(y_pred_sample, label=f'{name} Prediction', color=colors[name], linestyle='--', alpha=0.6)
    i += 1

plt.title('Figure 8: Actual vs Predicted Energy Consumption (Sample)')
plt.xlabel('Test Data Sample Index')
plt.ylabel('Consumption Value')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('images/actual_vs_pred_f8.png') # Saving to images folder and renaming for consistency
plt.close()
print("\n'images/actual_vs_pred_f8.png' file saved. Include this visualization in your report.")


print("\n--- 4. ERROR DISTRIBUTION VISUALIZATION STARTED (Figure 9) ---")


error_data = {}
for name, res in results.items():
    model = res['Model']
    y_pred = model.predict(X_test)
    
    residuals = y_test - y_pred 
    error_data[name] = residuals

error_df = pd.DataFrame(error_data)


plt.figure(figsize=(12, 6))

error_df.boxplot(vert=False, patch_artist=True) 

plt.title('Figure 9: Error Distribution Comparison Between Models (Residuals)')
plt.xlabel('Prediction Error (Actual - Predicted)')
plt.ylabel('Machine Learning Model')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('images/model_error_comparison_f9.png') # Saving to images folder and renaming for consistency
plt.close()
print("\n'images/model_error_comparison_f9.png' file saved. Include this visualization in your report.")


print("\n--- 5. FIGURE 5 VISUALIZATION STARTED (2 Models) ---")

try:
    lin_reg_model = results['Linear Regression']['Model']
    rf_model = results['Random Forest Regressor']['Model']
except KeyError as e:
 
    print(f"ERROR: Model key {e} not found in the 'results' dictionary. Please check model names.")
    exit()


sample_size = 200
X_sample_f5 = X_test.head(sample_size)
y_actual_sample_f5 = y_test.head(sample_size)


y_pred_lr = lin_reg_model.predict(X_sample_f5)
y_pred_rf = rf_model.predict(X_sample_f5)


plt.figure(figsize=(15, 6))


plt.plot(y_actual_sample_f5.values, label='Actual Consumption', color='black', linewidth=2, alpha=0.7)


plt.plot(y_pred_lr, label='Linear Regression Prediction', color='red', linestyle='--')


plt.plot(y_pred_rf, label='Random Forest Prediction', color='blue', linestyle='-.')


plt.title('Figure 5: Energy Consumption: Actual vs Predicted (Linear Regression vs Random Forest)')
plt.xlabel('Test Data Sample Index')
plt.ylabel('Consumption Value')
plt.legend()
plt.grid(True, alpha=0.3)


plt.savefig('images/actual_vs_predicted_2models_f5.png') 
plt.close()
print("\n'images/actual_vs_predicted_2models_f5.png' file saved. Use this visualization for Figure 5 in your report.")