import os

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

try:
    from src.nlp_bilstm import (
        build_texts_from_dataframe,
        infer_nlp_signals_batch,
        train_and_save_nlp_models,
        weak_issue_label,
        weak_sentiment_label,
    )
except ModuleNotFoundError:
    from nlp_bilstm import (
        build_texts_from_dataframe,
        infer_nlp_signals_batch,
        train_and_save_nlp_models,
        weak_issue_label,
        weak_sentiment_label,
    )

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE, "data", "Customer-Churn.csv")
MODELS_DIR = os.path.join(BASE, "models")


def load_and_prepare_data(data_path: str = DATA_PATH):
    df = pd.read_csv(data_path).drop_duplicates()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    for col in df.select_dtypes(include=np.number).columns:
        df[col] = df[col].fillna(df[col].mean())

    df["AvgMonthlySpend"] = df["TotalCharges"] / (df["tenure"] + 1)
    return df


def add_nlp_features(df, nlp_bundle=None):
    if nlp_bundle:
        texts = build_texts_from_dataframe(df)
        signals = infer_nlp_signals_batch(nlp_bundle["model"], nlp_bundle["meta"], texts)
        signal_df = pd.DataFrame(signals)
        df["nlp_sentiment_score"] = signal_df["sentiment_score"].astype(float)
        df["nlp_issue_code"] = signal_df["issue_code"].astype(float)
    else:
        df["nlp_sentiment_score"] = [
            {"negative": -1.0, "neutral": 0.0, "positive": 1.0}[weak_sentiment_label(row.to_dict())]
            for _, row in df.iterrows()
        ]
        df["nlp_issue_code"] = [
            ["billing", "service", "network", "support", "account", "general"].index(
                weak_issue_label(row.to_dict())
            )
            for _, row in df.iterrows()
        ]
    return df


def encode_for_training(df):
    encoded = df.copy()
    encoded["Churn"] = LabelEncoder().fit_transform(encoded["Churn"])
    encoded = pd.get_dummies(encoded, drop_first=True)
    x = encoded.drop("Churn", axis=1)
    y = encoded["Churn"]
    return x, y, x.columns.tolist()


def calc_metrics(y_true, y_pred):
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
        "f1": round(f1_score(y_true, y_pred), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def train_models(x, y, random_state: int = 42):
    scaler = MinMaxScaler()
    x_scaled = scaler.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(
        x_scaled, y, test_size=0.3, random_state=random_state
    )

    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(x_train, y_train)
    y_pred_log = log_model.predict(x_test)

    dt_model = DecisionTreeClassifier(max_depth=5, random_state=random_state)
    dt_model.fit(x_train, y_train)
    y_pred_dt = dt_model.predict(x_test)

    dt_full = DecisionTreeClassifier(random_state=random_state)
    dt_full.fit(x_train, y_train)
    top_features = (
        pd.Series(dt_full.feature_importances_, index=x.columns)
        .sort_values(ascending=False)
        .head(10)
        .to_dict()
    )

    metrics = {
        "logistic_regression": calc_metrics(y_test, y_pred_log),
        "decision_tree": calc_metrics(y_test, y_pred_dt),
        "feature_importance": top_features,
    }
    return log_model, dt_model, scaler, metrics


def save_artifacts(log_model, dt_model, scaler, feature_columns, metrics):
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(log_model, os.path.join(MODELS_DIR, "churn_log_model.joblib"))
    joblib.dump(dt_model, os.path.join(MODELS_DIR, "churn_dt_model.joblib"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))
    joblib.dump(feature_columns, os.path.join(MODELS_DIR, "feature_columns.joblib"))
    joblib.dump(metrics, os.path.join(MODELS_DIR, "model_metrics.joblib"))


def main():
    print("📂 Loading data...")
    df = load_and_prepare_data(DATA_PATH)
    nlp_bundle = train_and_save_nlp_models(df, MODELS_DIR)
    df = add_nlp_features(df, nlp_bundle)
    x, y, feature_columns = encode_for_training(df)
    print(f"   {len(x)} rows, {len(feature_columns)} features")

    print("🔬 Training models...")
    log_model, dt_model, scaler, metrics = train_models(x, y)

    for name, m in [
        ("Logistic Regression", metrics["logistic_regression"]),
        ("Decision Tree", metrics["decision_tree"]),
    ]:
        print(
            f"   {name}: Acc={m['accuracy']}, Prec={m['precision']}, "
            f"Rec={m['recall']}, F1={m['f1']}"
        )

    print("💾 Saving artifacts...")
    save_artifacts(log_model, dt_model, scaler, feature_columns, metrics)
    print("✅ Done!")


if __name__ == "__main__":
    main()
