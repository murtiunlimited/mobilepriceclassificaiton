from pathlib import Path
from src.a_load_data import load_datasets
from src.b_eda import (
    categorical_summary, numerical_summary,
    missing_values, calculate_vif
)
from src.c_preprocess import clean_data
from src.d_train_model import train_svm
from src.e_evaluate import evaluate_model
from src.f_predict import predict_test
from src.utils import save_predictions, save_model

from sklearn.model_selection import train_test_split

base_dir = Path.cwd()


df_train, df_test = load_datasets(base_dir)

cat_cols = ['price_range', 'n_cores', 'blue', 'dual_sim',
            'five_g', 'four_g', 'touch_screen', 'wifi']

# EDA text outputs
print(categorical_summary(df_train))
print(numerical_summary(df_train, cat_cols))
print("Missing (train):", missing_values(df_train))
print("Missing (test):", missing_values(df_test))

# Clean data
df_train, df_test = clean_data(df_train, df_test)

# Split
X = df_train.drop(columns=['price_range'])
y = df_train['price_range']
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# VIF
print(calculate_vif(X_train))

# Train model
model, best_params = train_svm(X_train, y_train)
print("Best Params:", best_params)

# Evaluate
eval_results = evaluate_model(model, X_val, y_val)
print(eval_results)

# Predict on test
preds = predict_test(model, df_test, list(X.columns))

# Save
results_dir = base_dir / "results"
results_dir.mkdir(exist_ok=True)

pred_path = save_predictions(df_test, preds, results_dir)
model_path = save_model(model, list(X.columns), results_dir)

print("Predictions saved to:", pred_path)
print("Model saved:", model_path)
