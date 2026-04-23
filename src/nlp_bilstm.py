import os
import re
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np

TEXT_COLUMNS = [
    "SupportText",
    "TicketText",
    "ComplaintText",
    "FeedbackText",
    "CustomerFeedback",
]
SENTIMENT_LABELS = ["negative", "neutral", "positive"]
ISSUE_LABELS = ["billing", "service", "network", "support", "account", "general"]


def clean_text(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def weak_issue_label(row: Dict) -> str:
    payment = str(row.get("PaymentMethod", "")).lower()
    internet = str(row.get("InternetService", "")).lower()
    tech_support = str(row.get("TechSupport", "")).lower()
    support_calls = int(row.get("SupportCalls", row.get("support_calls", 0)) or 0)
    monthly = float(row.get("MonthlyCharges", 0) or 0)
    contract = str(row.get("Contract", "")).lower()

    if "electronic check" in payment or monthly > 80:
        return "billing"
    if "fiber" in internet and tech_support == "no":
        return "network"
    if support_calls >= 3:
        return "support"
    if "month-to-month" in contract:
        return "service"
    if "bank transfer" in payment or "credit card" in payment:
        return "account"
    return "general"


def weak_sentiment_label(row: Dict) -> str:
    support_calls = int(row.get("SupportCalls", row.get("support_calls", 0)) or 0)
    churn_val = str(row.get("Churn", "")).lower()
    monthly = float(row.get("MonthlyCharges", 0) or 0)
    if support_calls >= 3 or churn_val == "yes":
        return "negative"
    if support_calls >= 1 or monthly > 75:
        return "neutral"
    return "positive"


def synthesize_text(row: Dict) -> str:
    tenure = row.get("tenure", 0)
    monthly = row.get("MonthlyCharges", 0)
    support_calls = int(row.get("SupportCalls", row.get("support_calls", 0)) or 0)
    contract = row.get("Contract", "unknown")
    issue = weak_issue_label(row)
    sentiment = weak_sentiment_label(row)

    return (
        f"customer tenure {tenure} months with {contract} contract. "
        f"monthly charges {monthly}. support calls {support_calls}. "
        f"primary issue {issue}. interaction sentiment {sentiment}."
    )


def build_texts_from_dataframe(df) -> List[str]:
    texts = []
    for _, row in df.iterrows():
        parts = []
        for col in TEXT_COLUMNS:
            if col in df.columns and str(row.get(col, "")).strip():
                parts.append(str(row.get(col, "")))
        if not parts:
            parts.append(synthesize_text(row.to_dict()))
        texts.append(clean_text(" ".join(parts)))
    return texts


def build_text_for_inference(customer_profile: Dict, user_text: str) -> str:
    parts = []
    if user_text.strip():
        parts.append(user_text)
    else:
        parts.append(synthesize_text(customer_profile))
    return clean_text(" ".join(parts))


def _build_and_train_bilstm(
    texts: List[str],
    sentiment_labels: List[str],
    issue_labels: List[str],
    max_words: int = 5000,
    max_len: int = 80,
):
    import tensorflow as tf

    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    x_padded = tf.keras.preprocessing.sequence.pad_sequences(
        sequences, maxlen=max_len, padding="post", truncating="post"
    )

    sentiment_map = {label: idx for idx, label in enumerate(SENTIMENT_LABELS)}
    issue_map = {label: idx for idx, label in enumerate(ISSUE_LABELS)}
    y_sent = np.array([sentiment_map[label] for label in sentiment_labels], dtype=np.int32)
    y_issue = np.array([issue_map[label] for label in issue_labels], dtype=np.int32)

    input_layer = tf.keras.layers.Input(shape=(max_len,))
    emb = tf.keras.layers.Embedding(input_dim=max_words, output_dim=64, mask_zero=True)(input_layer)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=False))(emb)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)

    sent_head = tf.keras.layers.Dense(len(SENTIMENT_LABELS), activation="softmax", name="sentiment")(x)
    issue_head = tf.keras.layers.Dense(len(ISSUE_LABELS), activation="softmax", name="issue")(x)

    model = tf.keras.Model(inputs=input_layer, outputs=[sent_head, issue_head])
    model.compile(
        optimizer="adam",
        loss={"sentiment": "sparse_categorical_crossentropy", "issue": "sparse_categorical_crossentropy"},
        metrics={"sentiment": "accuracy", "issue": "accuracy"},
    )
    model.fit(
        x_padded,
        {"sentiment": y_sent, "issue": y_issue},
        epochs=6,
        batch_size=64,
        validation_split=0.2,
        verbose=0,
    )
    return model, tokenizer, max_len


def train_and_save_nlp_models(df, models_dir: str) -> Optional[Dict]:
    try:
        texts = build_texts_from_dataframe(df)
        sentiment_labels = [weak_sentiment_label(row.to_dict()) for _, row in df.iterrows()]
        issue_labels = [weak_issue_label(row.to_dict()) for _, row in df.iterrows()]
        model, tokenizer, max_len = _build_and_train_bilstm(texts, sentiment_labels, issue_labels)
    except Exception as exc:
        print(f"⚠️ NLP BiLSTM training skipped: {exc}")
        return None

    os.makedirs(models_dir, exist_ok=True)
    nlp_model_path = os.path.join(models_dir, "nlp_bilstm.keras")
    model.save(nlp_model_path)
    meta = {
        "tokenizer_json": tokenizer.to_json(),
        "max_len": max_len,
        "sentiment_labels": SENTIMENT_LABELS,
        "issue_labels": ISSUE_LABELS,
    }
    joblib.dump(meta, os.path.join(models_dir, "nlp_meta.joblib"))
    return {"model": model, "meta": meta}


def load_nlp_models(models_dir: str) -> Tuple[Optional[object], Optional[Dict]]:
    try:
        import tensorflow as tf

        model = tf.keras.models.load_model(os.path.join(models_dir, "nlp_bilstm.keras"))
        meta = joblib.load(os.path.join(models_dir, "nlp_meta.joblib"))
        return model, meta
    except Exception:
        return None, None


def infer_nlp_signals(model, meta: Dict, text: str) -> Dict:
    import tensorflow as tf

    tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(meta["tokenizer_json"])
    seq = tokenizer.texts_to_sequences([clean_text(text)])
    x = tf.keras.preprocessing.sequence.pad_sequences(
        seq, maxlen=meta["max_len"], padding="post", truncating="post"
    )
    sent_probs, issue_probs = model.predict(x, verbose=0)
    sent_probs = sent_probs[0]
    issue_probs = issue_probs[0]

    sentiment_idx = int(np.argmax(sent_probs))
    issue_idx = int(np.argmax(issue_probs))
    sentiment_label = meta["sentiment_labels"][sentiment_idx]
    issue_label = meta["issue_labels"][issue_idx]

    sentiment_score = float(sent_probs[2] - sent_probs[0])
    return {
        "sentiment_label": sentiment_label,
        "issue_label": issue_label,
        "sentiment_score": sentiment_score,
        "issue_code": issue_idx,
    }


def infer_nlp_signals_batch(model, meta: Dict, texts: List[str]) -> List[Dict]:
    import tensorflow as tf

    tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(meta["tokenizer_json"])
    seq = tokenizer.texts_to_sequences([clean_text(text) for text in texts])
    x = tf.keras.preprocessing.sequence.pad_sequences(
        seq, maxlen=meta["max_len"], padding="post", truncating="post"
    )
    sent_probs, issue_probs = model.predict(x, verbose=0)
    out = []
    for sent_vec, issue_vec in zip(sent_probs, issue_probs):
        sentiment_idx = int(np.argmax(sent_vec))
        issue_idx = int(np.argmax(issue_vec))
        out.append(
            {
                "sentiment_label": meta["sentiment_labels"][sentiment_idx],
                "issue_label": meta["issue_labels"][issue_idx],
                "sentiment_score": float(sent_vec[2] - sent_vec[0]),
                "issue_code": issue_idx,
            }
        )
    return out
