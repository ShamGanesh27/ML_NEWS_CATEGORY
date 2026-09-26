import streamlit as st
from transformers import pipeline


# =========================================================
# MULTILINGUAL ZERO-SHOT MODEL
# =========================================================

@st.cache_resource
def load_multilingual_model():

    classifier = pipeline(
        "zero-shot-classification",
        model="joeddav/xlm-roberta-large-xnli"
    )

    return classifier


# =========================================================
# CATEGORY LABELS
# =========================================================

candidate_labels = [
    "sports news",
    "business news",
    "politics news",
    "entertainment news",
    "technology news"
]


# =========================================================
# CATEGORY MAPPING
# =========================================================

category_mapping = {
    "sports news": "SPORTS",
    "business news": "BUSINESS",
    "politics news": "POLITICS",
    "entertainment news": "ENTERTAINMENT",
    "technology news": "TECHNOLOGY"
}


# =========================================================
# PREDICT MULTILINGUAL NEWS CATEGORY
# =========================================================

def predict_multilingual(text):

    classifier = load_multilingual_model()

    result = classifier(
        text,
        candidate_labels,
        multi_label=False
    )

    predicted_label = result["labels"][0]

    confidence = result["scores"][0]

    display_category = category_mapping.get(
        predicted_label,
        predicted_label.upper()
    )

    return display_category, confidence, result