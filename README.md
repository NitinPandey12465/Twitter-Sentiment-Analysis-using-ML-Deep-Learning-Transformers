#  Multi-Model Twitter Sentiment Analysis using ML, DL & Transformers

##  Project Overview

This project performs **sentiment analysis** on Twitter-like text data, comparing multiple modeling approaches:
- Traditional **Machine Learning**
- **Deep Learning** with word embeddings
- **Transformer-based models** using HuggingFace

It classifies tweets into **Fresh** or **Rotten** categories, evaluates model performance using standard NLP metrics, and visualizes insights using plots like confusion matrices, class distributions, and word clouds.

---

## Problem Statement

The aim is to identify the sentiment of a tweet using its text content, leveraging both classical and modern NLP techniques.

---

##  Technologies & Libraries

- **Python**
- **scikit-learn** (ML models, metrics)
- **Keras / TensorFlow** (LSTM, CNN)
- **PyTorch** (used optionally for Transformers)
- **HuggingFace Transformers** (BERT, RoBERTa, ALBERT)
- **NLTK** / **spaCy** (text preprocessing)
- **Pandas / NumPy** (data wrangling)
- **Matplotlib / Seaborn** (visualization)
- **WordCloud** (text visualization)

---

##  Data Preprocessing

Performed on all input text before training any model:
- Text cleaning (punctuation, URLs, mentions, hashtags)
- Lowercasing
- Stopword removal
- Lemmatization
- Tokenization
- Padding/truncation for deep learning input
- Label encoding (e.g., Fresh = 1, Rotten = 0)

---

##  Models Used

###  Traditional ML Models
Implemented using **TF-IDF** and **BoW** features:
- Logistic Regression
- Support Vector Machine (SVM)
- Naive Bayes
- Random Forest

 These models serve as baselines to compare against deep models.

---

###  Deep Learning Models

#### 1. **LSTM (Long Short-Term Memory)**
- Sequential model with word embedding layer (GloVe/Word2Vec)
- Captures long-term dependencies in sequences

#### 2. **CNN (1D Convolutional Neural Network)**
- Uses 1D filters to extract spatial patterns in text
- Faster than LSTM, good for local feature extraction

 Both models used with:
- Embedding layer
- Dropout for regularization
- Binary crossentropy loss

---

### 🔹 Transformer Models (Using HuggingFace)

#### 1. **BERT (bert-base-uncased)**
- Bidirectional transformer that understands context deeply

#### 2. **RoBERTa (roberta-base)**
- Robust version of BERT with better training and data

#### 3. **ALBERT (albert-base-v2)**
- Lighter and faster than BERT, with parameter sharing

 All transformer models:
- Fine-tuned using HuggingFace `Trainer`
- Used pre-tokenized sequences
- Trained with learning rate scheduling and early stopping

---

##  Evaluation Metrics

Each model is evaluated using:
- Accuracy
- Precision
- Recall
- F1-score (macro, micro, weighted)
- Confusion Matrix

 Detailed classification reports are saved in:  
`classification_reports.txt`

---

##  Visualizations

| Visualization | Description |
|---------------|-------------|
|  Bar plots | Sentiment class distribution before training |
|  Confusion Matrix | To check false positives/negatives |
|  Word Cloud | Most common words in Fresh/Rotten classes |
|  Loss & Accuracy | Training vs Validation curves for DL/Transformer models |
|  F1-score Charts | Comparison across models |

###  Example: Class Distribution
```python
sns.countplot(df['sentiment_label'])
