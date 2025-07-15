# SMS-classification.

# 📩 Spam SMS Detection using NLP and Naive Bayes

This project uses **Natural Language Processing (NLP)** and **Machine Learning** to classify SMS messages as **spam** or **ham** (not spam). It extracts linguistic and statistical features from text and applies a **Multinomial Naive Bayes** classifier.

---

## 📌 Problem Statement

Spam messages are a common nuisance. The goal of this project is to automatically identify spam SMS messages using text classification techniques and reduce the number of unwanted messages received by users.

---

## 🧠 Model Overview

- **Text Cleaning** using SpaCy and regex
- **Stopword Removal**, **Lemmatization**, **Tokenization**
- **Feature Engineering**:
  - Word count
  - Character count
  - Digit count
  - Part-of-speech (POS) counts (nouns, verbs)
- **Text Vectorization** using TF-IDF
- **Classifier**: Multinomial Naive Bayes
- **Evaluation**: Accuracy on both train and validation sets

---

## 📁 Dataset

- File: `spamdata.csv`
- Columns:
  - `label`: "spam" or "ham"
  - `text`: SMS content

---

## 🧹 Text Preprocessing

- Convert to lowercase
- Remove punctuation
- Tokenize using `spaCy`
- Remove stopwords
- Lemmatize words

```python
def clean_text(text):
    cleaned = text.lower()
    cleaned = "".join([char for char in cleaned if char not in string.punctuation])
    my_doc = nlp(cleaned)
    token_list = [token.text for token in my_doc]
    return " ".join([word for word in token_list if not nlp.vocab[word].is_stop])
