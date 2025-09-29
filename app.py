import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import shutil

# -------------------------------
# Cleanup old NLTK cache (fixes punkt_tab issue)
# -------------------------------
shutil.rmtree("/home/appuser/nltk_data", ignore_errors=True)
shutil.rmtree("/home/adminuser/venv/nltk_data", ignore_errors=True)

# -------------------------------
# Download required NLTK resources
# -------------------------------
for resource in ["punkt", "stopwords"]:
    try:
        nltk.data.find(f"{'tokenizers' if resource=='punkt' else 'corpora'}/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)

# -------------------------------
# Initialize stemmer
# -------------------------------
ps = PorterStemmer()

# -------------------------------
# Text preprocessing
# -------------------------------
def transform_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)  # Correct punkt tokenizer

    # Keep only alphanumeric tokens
    tokens = [i for i in tokens if i.isalnum()]

    # Remove stopwords and punctuation
    tokens = [i for i in tokens if i not in stopwords.words("english") and i not in string.punctuation]

    # Stem the tokens
    tokens = [ps.stem(i) for i in tokens]

    return " ".join(tokens)

# -------------------------------
# Load vectorizer and model
# -------------------------------
with open("vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter your message here:")

if st.button("Predict"):
    if input_sms.strip() == "":
        st.warning("⚠️ Please enter a message first.")
    else:
        transformed_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]

        if result == 1:
            st.success("🚨 Spam")
        else:
            st.success("✅ Not Spam")
