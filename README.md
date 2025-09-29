# Email/SMS Spam Classifier

## **Project Overview**

This project is a **machine learning-based web application** that classifies messages as **Spam** or **Not Spam**. It demonstrates how **NLP preprocessing** combined with a **Naive Bayes classifier** can effectively detect spam in emails or SMS messages.

The app is built with **Python**, **Streamlit**, and **scikit-learn**, and provides a simple **web interface** for end users.

---

## **Dataset**

* **Dataset used:** `spam.csv` (available on [Kaggle](https://www.kaggle.com/uciml/sms-spam-collection-dataset))

* **Contents:**

  * `label`: Indicates whether the message is **spam** or **ham** (not spam)
  * `message`: The text of the SMS/email

* **Preprocessing applied:**

  * Lowercasing
  * Tokenization using **NLTK**
  * Stopword removal
  * Stemming using **PorterStemmer**

---

## **Model**

* **Algorithm:** Naive Bayes (MultinomialNB)
* **Vectorization:** TF-IDF (Term Frequency–Inverse Document Frequency)

### **Pipeline**

1. Preprocess the message text
2. Convert to TF-IDF vector
3. Predict using the Naive Bayes classifier

---

## **Tools & Technologies**

* **Jupyter Notebook** – for initial data exploration, preprocessing, and model training
* **PyCharm** – for creating a Python pipeline and preparing the project for deployment
* **Python Libraries:**

  * `pandas` – for data handling
  * `nltk` – for text preprocessing (tokenization, stopwords, stemming)
  * `scikit-learn` – for TF-IDF vectorization and Naive Bayes classifier
  * `pickle` – for saving the trained model and vectorizer
* **Streamlit** – for building a web interface and deploying the app locally

---

## **Project Pipeline**

1. **Data Loading:** Load the `spam.csv` dataset
2.** Data Cleaning:**
Remove missing or null values
Remove duplicates
Remove unnecessary columns
3. **Preprocessing:**

   * Lowercase text
   * Remove punctuation
   * Tokenize text
   * Remove stopwords
   * Apply stemming
4. **Feature Extraction:** Convert text messages to **TF-IDF vectors**
5. **Model Training:** Train a **Naive Bayes classifier** on the vectorized dataset
6. **Saving Model & Vectorizer:** Use `pickle` to save the trained model and TF-IDF vectorizer

---

## **Web App**

* Built using **Streamlit**
* Accepts **user input** for a message
* Preprocesses the input and predicts **Spam/Not Spam**

---


