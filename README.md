## Email/SMS Spam Classifier:

**Project Overview**
This project is a machine learning-based web application that classifies messages as Spam or Not Spam. It demonstrates how NLP preprocessing combined with a Naive Bayes classifier can effectively detect spam in emails or SMS messages.
The app is built with Python, Streamlit, and scikit-learn, and provides a simple web interface for end users.

**Dataset:**
Dataset used: spam.csv (available on Kaggle)

**Contents:**
label: Indicates whether the message is spam or ham (not spam)
message: The text of the SMS/email

**Preprocessing:**
Lowercasing
Tokenization using NLTK
Stopword removal
Stemming using PorterStemmer

**Model:**
Algorithm: Naive Bayes (MultinomialNB)
Vectorization: TF-IDF (Term Frequency–Inverse Document Frequency)

**Pipeline:**
Preprocess the message text
Convert to TF-IDF vector
Predict using Naive Bayes classifier

**Tools & Technologies:**
Jupyter Notebook – for initial data exploration, preprocessing, and model training
PyCharm – for creating a Python pipeline and preparing the project for deployment
Python Libraries:
pandas – for data handling
nltk – for text preprocessing (tokenization, stopwords, stemming)
scikit-learn – for TF-IDF vectorization and Naive Bayes classifier
pickle – for saving the trained model and vectorizer
Streamlit – for building a web interface and deploying the app locally

**Project Pipeline:**
Data Loading: Load spam.csv dataset
Preprocessing:
Lowercase text
Remove punctuation
Tokenize text
Remove stopwords
Apply stemming
Feature Extraction: Convert text messages to TF-IDF vectors
Model Training: Train a Naive Bayes classifier on the vectorized dataset
Saving Model & Vectorizer: Use pickle to save trained model and TF-IDF vectorizer

**Web App:**
Built using Streamlit
Accepts user input for a message
Preprocesses input and predicts Spam/Not Spam
