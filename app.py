import streamlit as st
import pickle
import numpy as np
import re
import time

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="AI Misinformation Detection System",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ------------------ LOAD MODEL ------------------
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# ------------------ TEXT CLEANING ------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

# ------------------ SUMMARIZATION ------------------
def summarize_text(text, num_sentences=3):
    sentences = text.split(". ")
    if len(sentences) <= num_sentences:
        return text
    return ". ".join(sentences[:num_sentences])

# ------------------ CUSTOM CSS ------------------
st.markdown("""
<style>

/* Hide Sidebar */
[data-testid="stSidebar"] { display: none; }
[data-testid="collapsedControl"] { display: none; }

/* Background Gradient */
.stApp {
    background: linear-gradient(135deg, #fbc2eb, #a6c1ee);
    min-height: 100vh;
}

/* Reduce top spacing */
.block-container {
    padding-top: 2rem;
}

/* Hero Section */
.hero {
    text-align: center;
    padding: 40px 20px;
}

/* Main Title */
.hero-title {
    font-size: 52px;
    font-weight: 800;
    color: #4b3f72;
}

/* Subtitle */
.hero-subtitle {
    font-size: 18px;
    color: #333;
    margin-top: 10px;
    margin-bottom:5px;
}

/* Description */
.hero-desc {
    max-width: 700px;
    margin: 5px auto;
    font-size: 16px;
    color: #444;
}

/* Text Area */
.stTextArea textarea {
    background-color: white;
    color: #222 !important;
    border-radius: 10px;
    border: 1px solid #ccc;
}

/* Button */
.stButton>button {
    background-color: #6c63ff;
    color: white;
    border-radius: 10px;
    padding: 5px 10px;
    font-size: 16px;
    border: none;
    transition: 0.3s;
}

.stButton>button:hover {
    background-color: #574fd6;
    transform: scale(1.05);
}

</style>
""", unsafe_allow_html=True)

# ------------------ HERO SECTION ------------------
st.markdown("""
<div class="hero">
    <div class="hero-title">🧠 AI Misinformation Detection System</div>
    <div class="hero-subtitle">
        Detect Fake News • Assess Credibility • Generate Summary
    </div>
    <div class="hero-desc">
        This AI system analyzes online news articles to detect misinformation,
        evaluate credibility using probability scores, and generate concise summaries
        to help prevent the spread of false information.
    </div>
</div>
""", unsafe_allow_html=True)

st.write("")

# ------------------ INPUT SECTION ------------------
user_input = st.text_area("Paste News Article:", height=200)

if st.button("Analyze Article"):
    if user_input.strip() == "":
        st.warning("Please enter article text.")
    else:
        with st.spinner("Analyzing article using AI model..."):
            time.sleep(2)

        cleaned = clean_text(user_input)
        transformed = vectorizer.transform([cleaned])

        prediction = model.predict(transformed)
        probability = model.predict_proba(transformed)

        real_prob = probability[0][1] * 100
        fake_prob = probability[0][0] * 100
        confidence = np.max(probability) * 100

        st.write("---")

        if prediction[0] == 1:
            st.success("Prediction: REAL News")
        else:
            st.error("Prediction: FAKE News")

        # Credibility Score
        st.subheader("Credibility Score")
        st.progress(int(real_prob))
        st.write(f"{real_prob:.2f}% Credible")

        # Fake Probability
        st.subheader("Fake News Probability")
        st.progress(int(fake_prob))
        st.write(f"{fake_prob:.2f}% Likely Fake")

        # Model Confidence
        st.subheader("Model Confidence")
        st.write(f"{confidence:.2f}%")

        # Summary
        st.subheader("Article Summary")
        summary = summarize_text(user_input)
        st.info(summary)

st.markdown("<br><center>Internship Project | Artificial Intelligence & Machine Learning</center>", unsafe_allow_html=True)