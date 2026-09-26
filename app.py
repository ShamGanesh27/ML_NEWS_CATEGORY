import streamlit as st
import pickle
import re

from multilingual_predictor import predict_multilingual


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="News Category Classifier",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #0b0f14;
    }

    .block-container {
        max-width: 1150px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    section[data-testid="stSidebar"] {
        background-color: #080b0f;
    }

    h1, h2, h3 {
        color: #ffffff !important;
    }

    .subtitle {
        color: #aeb7c4;
        font-size: 18px;
        margin-bottom: 25px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# LOAD ORIGINAL ML MODEL
# =========================================================

@st.cache_resource
def load_model():

    with open("model.pkl", "rb") as file:
        model = pickle.load(file)

    with open("vectorizer.pkl", "rb") as file:
        vectorizer = pickle.load(file)

    return model, vectorizer


model, vectorizer = load_model()


# =========================================================
# TEXT CLEANING
# =========================================================

def clean_text(text):

    text = str(text).lower()

    text = re.sub(
        r"http\S+|www\S+|https\S+",
        "",
        text
    )

    text = re.sub(
        r"<.*?>",
        "",
        text
    )

    text = re.sub(
        r"[^a-zA-Z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# =========================================================
# CATEGORY MAPPING
# =========================================================

category_names = {
    "sport": "SPORTS",
    "politics": "POLITICS",
    "business": "BUSINESS",
    "tech": "TECHNOLOGY",
    "entertainment": "ENTERTAINMENT"
}

category_icons = {
    "SPORTS": "⚽",
    "POLITICS": "🏛️",
    "BUSINESS": "💼",
    "TECHNOLOGY": "💻",
    "ENTERTAINMENT": "🎬"
}


# =========================================================
# MAIN HEADER
# =========================================================

st.title("📰 NEWS CATEGORY CLASSIFIER")

st.markdown(
    '<p class="subtitle">'
    'AI-powered news classification using Machine Learning'
    '</p>',
    unsafe_allow_html=True
)


# =========================================================
# CLASSIFICATION MODE
# =========================================================

st.markdown("### 🤖 Classification Mode")

mode = st.radio(
    "Choose how the article should be classified:",
    [
        "Standard ML",
        "Multilingual AI"
    ],
    horizontal=True
)

if mode == "Standard ML":

    st.caption(
        "TF-IDF + Logistic Regression trained on the News dataset."
    )

else:

    st.caption(
        "Multilingual zero-shot AI using XLM-RoBERTa. "
        "Supports English, Hindi, Tamil, Malayalam and Telugu."
    )


# =========================================================
# INTRODUCTION
# =========================================================

st.markdown("### ✨ Classify News Articles Instantly")

if mode == "Standard ML":

    st.info(
        "Enter a news headline and article content below. "
        "The trained machine learning model analyzes the text "
        "using TF-IDF features and predicts the relevant news category."
    )

else:

    st.info(
        "Enter a news headline and article content in English, "
        "Hindi, Tamil, Malayalam or Telugu. "
        "The multilingual AI model predicts the relevant news category."
    )


# =========================================================
# INPUT SECTION
# =========================================================

st.markdown("### 📝 Enter News Information")

news_title = st.text_input(
    "News Title",
    placeholder="Enter the news headline..."
)

news_article = st.text_area(
    "News Article",
    placeholder="Paste or type the complete news article here...",
    height=230
)


# =========================================================
# INPUT STATISTICS
# =========================================================

word_count = len(news_article.split())

character_count = len(news_article)

stat1, stat2, stat3 = st.columns(3)

with stat1:

    st.metric(
        "📝 Words",
        word_count
    )

with stat2:

    st.metric(
        "🔤 Characters",
        character_count
    )

with stat3:

    if news_title.strip() or news_article.strip():
        status = "Ready"
    else:
        status = "Waiting"

    st.metric(
        "📌 Status",
        status
    )


# =========================================================
# PREDICTION BUTTON
# =========================================================

st.write("")

predict_clicked = st.button(
    "✨ CLASSIFY ARTICLE",
    type="primary",
    use_container_width=True
)


# =========================================================
# PREDICTION
# =========================================================

if predict_clicked:

    if (
        news_title.strip() == ""
        and
        news_article.strip() == ""
    ):

        st.warning(
            "⚠️ Please enter a news title or article before classification."
        )

    else:

        # Combine title and article
        combined_text = (
            news_title
            + " "
            + news_article
        )


        # =================================================
        # STANDARD ML MODE
        # =================================================

        if mode == "Standard ML":

            # Clean text
            cleaned_text = clean_text(
                combined_text
            )

            # Convert text into TF-IDF
            text_vector = vectorizer.transform(
                [cleaned_text]
            )

            # Predict
            prediction = model.predict(
                text_vector
            )[0]

            # Display category
            display_category = category_names.get(
                prediction,
                str(prediction).upper()
            )

            icon = category_icons.get(
                display_category,
                "📰"
            )

            # Result
            st.markdown("### 🎯 Prediction Result")

            st.success(
                f"{icon} Predicted Category: **{display_category}**"
            )

            # Confidence
            if hasattr(
                model,
                "predict_proba"
            ):

                probabilities = model.predict_proba(
                    text_vector
                )[0]

                confidence = max(
                    probabilities
                )

                st.markdown(
                    "### 📊 Prediction Confidence"
                )

                st.metric(
                    "Model Confidence",
                    f"{confidence * 100:.2f}%"
                )

                st.progress(
                    float(confidence)
                )

                # Probability breakdown
                st.markdown(
                    "### 🔎 Category Probability"
                )

                probability_data = []

                for class_name, probability in zip(
                    model.classes_,
                    probabilities
                ):

                    display_name = category_names.get(
                        class_name,
                        str(class_name).upper()
                    )

                    probability_data.append(
                        (
                            display_name,
                            float(probability)
                        )
                    )

                probability_data.sort(
                    key=lambda item: item[1],
                    reverse=True
                )

                for category, probability in probability_data:

                    col1, col2 = st.columns(
                        [1, 4]
                    )

                    with col1:

                        st.write(
                            f"**{category}**"
                        )

                    with col2:

                        st.progress(
                            probability
                        )

                    st.caption(
                        f"{probability * 100:.2f}%"
                    )


        # =================================================
        # MULTILINGUAL AI MODE
        # =================================================

        else:

            with st.spinner(
                "🤖 Analyzing the article with multilingual AI..."
            ):

                display_category, confidence, result = (
                    predict_multilingual(
                        combined_text
                    )
                )

            icon = category_icons.get(
                display_category,
                "📰"
            )

            # Result
            st.markdown("### 🎯 Multilingual AI Prediction")

            st.success(
                f"{icon} Predicted Category: **{display_category}**"
            )

            # Confidence
            st.markdown(
                "### 📊 AI Confidence"
            )

            st.metric(
                "Model Confidence",
                f"{confidence * 100:.2f}%"
            )

            st.progress(
                float(confidence)
            )

            # Probability breakdown
            st.markdown(
                "### 🔎 Category Scores"
            )

            for label, score in zip(
                result["labels"],
                result["scores"]
            ):

                display_name = category_names.get(
                    label.replace(" news", ""),
                    label.upper()
                )

                col1, col2 = st.columns(
                    [1, 4]
                )

                with col1:

                    st.write(
                        f"**{display_name}**"
                    )

                with col2:

                    st.progress(
                        float(score)
                    )

                st.caption(
                    f"{score * 100:.2f}%"
                )


# =========================================================
# SUPPORTED CATEGORIES
# =========================================================

st.markdown(
    "### 🗂️ Supported Categories"
)

category_columns = st.columns(5)

categories = [
    ("⚽", "SPORTS"),
    ("💼", "BUSINESS"),
    ("🏛️", "POLITICS"),
    ("🎬", "ENTERTAINMENT"),
    ("💻", "TECHNOLOGY")
]

for column, (icon, category) in zip(
    category_columns,
    categories
):

    with column:

        st.info(
            f"{icon}\n\n**{category}**"
        )


# =========================================================
# MACHINE LEARNING PIPELINE
# =========================================================

st.markdown(
    "### ⚙️ How the System Works"
)

pipeline1, pipeline2, pipeline3, pipeline4, pipeline5 = st.columns(5)

with pipeline1:

    st.info(
        "📰 **News Text**\n\n"
        "Article input"
    )

with pipeline2:

    st.info(
        "🧹 **Text Cleaning**\n\n"
        "Clean and normalize"
    )

with pipeline3:

    st.info(
        "📐 **TF-IDF**\n\n"
        "Extract text features"
    )

with pipeline4:

    st.info(
        "🤖 **ML Model**\n\n"
        "Classify the article"
    )

with pipeline5:

    st.info(
        "🎯 **Prediction**\n\n"
        "Display category"
    )


# =========================================================
# MULTILINGUAL AI INFORMATION
# =========================================================

st.markdown(
    "### 🌐 Multilingual AI"
)

multi1, multi2, multi3 = st.columns(3)

with multi1:

    st.subheader("🌍 Languages")

    st.write(
        "English • Hindi • Tamil • Malayalam • Telugu"
    )

with multi2:

    st.subheader("🤖 AI Model")

    st.write(
        "XLM-RoBERTa multilingual zero-shot classification"
    )

with multi3:

    st.subheader("🎯 Categories")

    st.write(
        "Sports • Business • Politics • Entertainment • Technology"
    )


# =========================================================
# ABOUT THE PROJECT
# =========================================================

st.markdown(
    "### 📚 About the Project"
)

about1, about2, about3 = st.columns(3)

with about1:

    st.subheader("🤖 Machine Learning")

    st.write(
        "Multiple classification algorithms were evaluated "
        "for automatic news category prediction."
    )


with about2:

    st.subheader("📐 TF-IDF")

    st.write(
        "News text is converted into numerical TF-IDF "
        "features before classification."
    )


with about3:

    st.subheader("🌐 Multilingual AI")

    st.write(
        "A multilingual zero-shot model enables news "
        "classification across multiple Indian languages."
    )


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("📰 News Classifier")

    st.write(
        "A machine learning application that classifies "
        "news articles into five categories."
    )

    st.divider()

    st.subheader("📂 Categories")

    st.write("⚽ Sports")
    st.write("💼 Business")
    st.write("🏛️ Politics")
    st.write("🎬 Entertainment")
    st.write("💻 Technology")

    st.divider()

    st.subheader("🤖 Models Evaluated")

    st.write("• Logistic Regression")
    st.write("• Decision Tree")
    st.write("• Random Forest")

    st.divider()

    st.subheader("🌐 Multilingual AI")

    st.write("• XLM-RoBERTa")
    st.write("• Zero-shot Classification")
    st.write("• 5 Languages")

    st.divider()

    st.subheader("🔧 Technology")

    st.write("• Python")
    st.write("• Scikit-learn")
    st.write("• Transformers")
    st.write("• PyTorch")
    st.write("• Streamlit")

    st.divider()

    st.caption(
        "Machine Learning • NLP • Multilingual AI"
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    "---"
)

st.caption(
    "📰 News Category Classifier • Machine Learning Application"
)