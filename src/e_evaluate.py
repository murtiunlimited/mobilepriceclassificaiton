from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def evaluate_model(model, X_val, y_val):
    y_pred = model.predict(X_val)

    results = {
        "accuracy": accuracy_score(y_val, y_pred),
        "report": classification_report(y_val, y_pred),
        "confusion_matrix": confusion_matrix(y_val, y_pred)
    }

    return results
