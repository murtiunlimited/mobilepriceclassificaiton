import joblib
import pandas as pd
from datetime import datetime

def save_model(model, feature_order, results_dir):
    model_path = results_dir / "model.joblib"
    joblib.dump(model, model_path)

    pd.Series(feature_order).to_csv(
        results_dir / "feature.csv",
        index=False,
        header=False
    )

    return model_path


def save_predictions(df_test, preds, results_dir):
    df_test["predicted_price_range"] = preds

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = results_dir / f"prediction{timestamp}.csv"

    df_test[["id", "predicted_price_range"]].to_csv(output_path, index=False)
    return output_path
