import pandas as pd

def predict_test(model, df_test, feature_order):
    X_test = df_test[feature_order]
    preds = model.predict(X_test)
    return preds


def predict_from_list(model, features):
    return model.predict([features])[0]


def predict_from_dict(model, feature_dict, feature_order):
    arr = [feature_dict[col] for col in feature_order]
    return model.predict([arr])[0]
